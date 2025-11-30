import numpy as np
import os
import shutil
import heapq
import tqdm
from sklearn.cluster import MiniBatchKMeans, KMeans

ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 64
WINDOW=6000
BATCH_SIZE=4000

class IVFFlat:
    def __init__(self, db_path, index_path, n_centroids,n_centroids_level2, n_probe,n_probes_top):
        self.db_path = db_path
        self.index_path = index_path
        self.n_centroids = n_centroids
        self.n_centroids_level2 = n_centroids_level2
        self.n_probe = n_probe
        self.n_probes_top = n_probes_top
        self.db_size = os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)


    def build(self,data):

        # 1-level clustering
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_centroids,
            init="k-means++",
            batch_size=10_000,
            n_init=5,
            random_state=42
        )
        kmeans.fit(data)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_.astype(np.float32)
        del data, kmeans

        if os.path.exists(self.index_path):
            shutil.rmtree(self.index_path)
        os.makedirs(self.index_path, exist_ok=True)

        cluster_infos = [(cid, np.where(labels == cid)[0].astype(np.uint32))
                        for cid in range(self.n_centroids)]

        kmeans2 = KMeans(
            n_clusters=self.n_centroids_level2,
            init="k-means++",
            n_init=10,
            random_state=42,
        )
        kmeans2.fit(centers)
        centers2 = kmeans2.cluster_centers_.astype(np.float32)
        labels2 = kmeans2.labels_
        cluster_level2_infos = [(cid, np.where(labels2 == cid)[0].astype(np.uint32))
                            for cid in range(self.n_centroids_level2)]

        reordered_centers = []
        reordered_cluster_infos = []
        for _, inds in cluster_level2_infos:
            for ind in inds:
                reordered_centers.append(centers[ind])
                reordered_cluster_infos.append(cluster_infos[ind])
        centers = np.array(reordered_centers, dtype=np.float32)
        cluster_infos = reordered_cluster_infos
        del labels, labels2
        del reordered_centers, reordered_cluster_infos

        header = []
        flat_path = os.path.join(self.index_path, "all_indices.bin")
        with open(flat_path, "wb") as f:
            offset = 0
            for _, inds in cluster_infos:
                length = inds.size
                f.write(inds.tobytes())
                header.append([offset, length])
                offset += length * inds.dtype.itemsize
        header_matrix = np.array(header, dtype=np.uint32)
        header_matrix.tofile(os.path.join(self.index_path, "index_header.bin"))

        # save level2 header (offset, length) for easy slicing later
        level2_header = []
        offset = 0
        for _, inds in cluster_level2_infos:
            length = len(inds)  
            level2_header.append([offset, length])
            offset += length
        np.array(level2_header, dtype=np.uint32).tofile(os.path.join(self.index_path, "level2_header.bin"))

        # normalize centers
        centers /= (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)
        centers.astype(np.float32).tofile(os.path.join(self.index_path, "centroids.dat"))

        centers2 /= (np.linalg.norm(centers2, axis=1, keepdims=True) + 1e-12)
        centers2.astype(np.float32).tofile(os.path.join(self.index_path, "centroids_level2.dat"))

   
    def retrieve(self, query, top_k=5):
        # ----------------------------
        # Normalize query
        # ----------------------------
        query = np.asarray(query, dtype=np.float32).squeeze()
        qn = np.linalg.norm(query)
        normalized_query = query / (qn if qn != 0 else 1)

        centers_top_path = os.path.join(self.index_path, "centroids_level2.dat")
        if not os.path.exists(centers_top_path):
            return []

        centroids_top = np.memmap(
            centers_top_path,
            dtype=np.float32,
            mode='r',
            shape=(self.n_centroids_level2, DIMENSION),
        )
        centroid_top_scores = centroids_top.dot(normalized_query)
        nearest_top_centroids = np.argsort(-centroid_top_scores)[:self.n_probes_top]
        del centroids_top, centroid_top_scores

        level2_header = np.fromfile(
            os.path.join(self.index_path, "level2_header.bin"),
            dtype=np.uint32
        ).reshape(-1, 2)

        ranges = []
        for c in nearest_top_centroids:
            start, length = level2_header[c]
            if length > 0:
                ranges.append((int(start), int(length)))

        if not ranges:
            return []

        first_start = min(s for s, l in ranges)
        last_end = max(s + l for s, l in ranges)
        total_len = last_end - first_start

        centroids_path = os.path.join(self.index_path, "centroids.dat")
        byte_offset = first_start * DIMENSION * 4

        mm = np.memmap(
            centroids_path,
            dtype=np.float32,
            mode="r",
            offset=byte_offset,
            shape=(total_len, DIMENSION)
        )

        level1_scores = []
        level1_ids = []

        for start, length in ranges:
            local_start = start - first_start
            local_end = local_start + length

            block = mm[local_start:local_end]   # zero-copy slice
            sims = block.dot(normalized_query)

            ids = start + np.arange(length, dtype=np.int64)

            level1_scores.append(sims)
            level1_ids.append(ids)

        del mm
        all_scores = np.concatenate(level1_scores)
        all_ids = np.concatenate(level1_ids)
        k = min(self.n_probe, all_scores.size)

        top_idx = np.argpartition(-all_scores, k - 1)[:k]
        order = np.argsort(-all_scores[top_idx])
        nearest_centroids = all_ids[top_idx][order]

        header_arr = np.fromfile(
            os.path.join(self.index_path, "index_header.bin"),
            dtype=np.uint32
        ).reshape(-1, 2)

        all_ids = []
        flat_index_path = os.path.join(self.index_path, "all_indices.bin")
        for c in nearest_centroids:
            offset, length = header_arr[c]
            if length == 0:
                continue
            mm = np.memmap(
                flat_index_path, dtype=np.uint32, mode="r",
                offset=offset, shape=(length,)
            )
            all_ids.extend(mm[:])
            del mm

        all_ids.sort()
        groups = self.group_ids_by_window_fast(all_ids, WINDOW)
        del all_ids

        row_size = DIMENSION * 4
        top_heap = []

        with open(self.db_path, "rb") as f:
            for g in groups:
                start_id = g[0]
                end_id = g[-1]
                offset=np.int64(start_id) * np.int64(row_size)
                f.seek(offset)

                block_len = end_id - start_id + 1
                block = np.frombuffer(f.read(block_len * row_size), dtype=np.float32)
                block = block.reshape(block_len, DIMENSION)

                idxs = g - start_id
                vecs = block[idxs, :]

                scores = vecs.dot(normalized_query)

                for score, idx in zip(scores, g):
                    if len(top_heap) < top_k:
                        heapq.heappush(top_heap, (score, idx))
                        
                    else:
                        heapq.heappushpop(top_heap, (score, idx))

                del vecs, block, scores
        return [idx for score, idx in heapq.nlargest(top_k, top_heap)]

    def group_ids_by_window_fast(self, all_ids, window):
        """
        Group sorted unique IDs into consecutive blocks of max size `window`.
        """
        all_ids = np.asarray(all_ids)
        if len(all_ids) == 0:
            return []

        groups = []
        start = 0
        for i in range(1, len(all_ids)):
            if all_ids[i] - all_ids[start] >= window:
                groups.append(all_ids[start:i])
                start = i
        groups.append(all_ids[start:])
        return groups

    def get_all_rows(self):
        size = os.path.getsize(self.db_path) // (DIMENSION * 4)
        mmap_data = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(size, DIMENSION))
        return np.array(mmap_data)
