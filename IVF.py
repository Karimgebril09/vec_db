import numpy as np
from sklearn.cluster import KMeans
import os

class IVFIndex:
    """
    Optimized IVFIndex:
      - cluster_centers saved in prefix_centers.npy
      - concatenated ids saved in prefix_ids.npy (dtype=uint32)
      - sizes saved in prefix_sizes.npy (dtype=uint32)
    Retrieval uses get_row(i) to fetch vectors on demand (memory-efficient).
    """

    def __init__(self, n_clusters=100, random_state=42):
        self.n_clusters = n_clusters
        self.cluster_centers = None              # (n_clusters, dim) float32
        # Compact representation of inverted lists:
        self.ids = None                          # 1D uint32 (all ids concatenated)
        self.sizes = None                        # 1D uint32 (size per cluster)
        self.random_state = random_state
        self.fitted = False

    # -------------------------
    # Fit & build compact index
    # -------------------------
    def fit(self, vectors):
        """
        vectors: (N, D) numpy array in memory (only needed to build index)
        After fit, builds:
          - self.cluster_centers (float32)
          - self.ids (uint32) concatenated
          - self.sizes (uint32) per cluster
        """
        print("Fitting KMeans centroids...")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        labels = kmeans.fit_predict(vectors)
        self.cluster_centers = kmeans.cluster_centers_.astype(np.float32)

        # Build lists for each cluster ID (0..n_clusters-1)
        lists = [[] for _ in range(self.n_clusters)]
        for idx, lab in enumerate(labels):
            lists[lab].append(idx)

        # Convert to uint32 arrays and concatenate
        sizes = np.empty(self.n_clusters, dtype=np.uint32)
        id_chunks = []
        for c in range(self.n_clusters):
            arr = np.array(lists[c], dtype=np.uint32)
            id_chunks.append(arr)
            sizes[c] = arr.size

        if len(id_chunks):
            ids = np.concatenate(id_chunks).astype(np.uint32)
        else:
            ids = np.empty(0, dtype=np.uint32)

        self.ids = ids
        self.sizes = sizes
        self.fitted = True
        print("Index built: n_clusters =", self.n_clusters, "total_ids =", self.ids.size)

    # -------------------------
    # Save / Load (binary .npy files)
    # -------------------------
    def save(self, index_dir):
        """Save IVF index in optimized compact format:
           - centers.npy  : (n_clusters, dim) float32
           - sizes.npy    : (n_clusters,) uint32
           - ids.dat      : all cluster ids concatenated as raw uint32
        """
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)

        # Save cluster centers
        np.save(
            os.path.join(index_dir, "centers.npy"),
            self.cluster_centers.astype(np.float32)
        )

        # Save sizes
        np.save(
            os.path.join(index_dir, "sizes.npy"),
            self.sizes.astype(np.uint32)
        )

        # Save concatenated ids as binary file
        self.ids.astype(np.uint32).tofile(
            os.path.join(index_dir, "ids.dat")
        )

        print(f"[OK] Index saved to folder: {index_dir}")


    def load(self, index_dir, mmap=False):
        """Load IVF index saved in compact format."""

        # Load cluster centers
        self.cluster_centers = np.load(
            os.path.join(index_dir, "centers.npy"),
            mmap_mode='r' if mmap else None
        )

        # Load sizes
        self.sizes = np.load(
            os.path.join(index_dir, "sizes.npy"),
            mmap_mode='r' if mmap else None
        )

        # Load concatenated ids
        ids_path = os.path.join(index_dir, "ids.dat")
        if mmap:
            self.ids = np.memmap(ids_path, dtype=np.uint32, mode='r')
        else:
            self.ids = np.fromfile(ids_path, dtype=np.uint32)

        self.n_clusters = self.sizes.size
        self.fitted = True

        print(f"[OK] Index loaded from folder: {index_dir}")

    # -------------------------
    # Helper to get cluster slice
    # -------------------------
    def _cluster_slice(self, cluster_id):
        """
        Return (start, size) for cluster_id, and a view into ids array for that cluster.
        """
        if self.sizes is None:
            raise ValueError("Index not loaded/fitted.")
        if cluster_id < 0 or cluster_id >= self.n_clusters:
            return np.empty(0, dtype=np.uint32)

        # compute offsets on the fly: offsets = cumsum(sizes) - sizes
        # For performance we can compute offsets once externally if needed.
        offsets = np.cumsum(self.sizes, dtype=np.uint64) - self.sizes.astype(np.uint64)
        start = int(offsets[cluster_id])
        size = int(self.sizes[cluster_id])
        if size == 0:
            return np.empty(0, dtype=np.uint32)
        return self.ids[start: start + size]

    # -------------------------
    # Retrieval
    # -------------------------
    def retrieve(self, query_vector, n_clusters, n_arrays, cosine_similarity, get_row, sort_results=True):
        """
        query_vector: 1D array (D,)
        n_clusters: number of nearest clusters to probe (n_probe)
        n_arrays: number of desired neighbors (top-k)
        cosine_similarity: function(a,b)->float
        get_row: function(i)->vector
        Returns: numpy array of top indices sorted by descending similarity.
        """
        if not self.fitted:
            raise ValueError("Index not fitted or loaded.")

        # 1) similarity to cluster centers
        sims_centers = np.array([cosine_similarity(query_vector, c) for c in self.cluster_centers]).squeeze()
        # select n_clusters clusters (fast)
        if n_clusters >= self.n_clusters:
            nearest_clusters = np.arange(self.n_clusters)
        else:
            nearest_clusters = np.argpartition(sims_centers, -n_clusters)[-n_clusters:]

        # 2) gather candidate indices by slicing ids using sizes/offsets
        # compute offsets once to avoid repeated cumsum calls
        offsets = np.cumsum(self.sizes, dtype=np.uint64) - self.sizes.astype(np.uint64)
        candidate_ids_list = []
        for c in nearest_clusters:
            start = int(offsets[c])
            size = int(self.sizes[c])
            if size > 0:
                candidate_ids_list.append(self.ids[start:start + size])
        if len(candidate_ids_list) == 0:
            return np.array([], dtype=np.uint32)

        all_candidate_ids = np.concatenate(candidate_ids_list)

        # 3) compute similarity to candidate vectors using get_row (on-demand)
        # If many candidates, consider batching the get_row calls at higher layer
        sims = np.empty(all_candidate_ids.shape[0], dtype=np.float32)
        for i_idx, vid in enumerate(all_candidate_ids):
            vec = get_row(int(vid))
            sims[i_idx] = cosine_similarity(query_vector, vec)

        # 4) select top n_arrays
        if n_arrays >= sims.size:
            sel_idx = np.argsort(sims)[::-1]  # descending
        else:
            part = np.argpartition(sims, -n_arrays)[-n_arrays:]
            # order the selected partition
            sel_idx = part[np.argsort(sims[part])[::-1]]

        top_ids = all_candidate_ids[sel_idx].astype(np.uint32)
        return top_ids
