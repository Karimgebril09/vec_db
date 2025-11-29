import json
import numpy as np
import os
import shutil
import tqdm
import heapq
from sklearn.cluster import MiniBatchKMeans
from typing import Annotated

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 64

class VecDB:
    def __init__(self, database_file_path="saved_db.dat", index_file_path="index.dat", new_db=True, db_size=None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.dim = db_size
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)

    def generate_database(self, size: int) -> None:
        vectors = np.memmap("new_embeddings.dat", dtype=np.float32, mode='r', shape=(size, DIMENSION))
        self._write_vectors_to_file(vectors)
        self._build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 64)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            raise

    def get_all_rows(self) -> np.ndarray:
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)

    def get_all_ids_rows(self, ids) -> np.ndarray:
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return vectors[ids]

    ############################################################################
    # -------------------------- Retrieval (search) --------------------------- #
    ############################################################################
    def retrieve(self, query, top_k=10, nprobe=40, candidate_multiplier=8):
        """
        IVF-PQ search:
          - find top-nprobe coarse cells (by L2)
          - for each cell, read list of (ids, pq-codes)
          - compute approximate PQ distances using the *query residual* (q - coarse_centroid)
          - keep top candidates, then re-rank by exact dot or L2 (exact)
        """
        q = np.asarray(query, dtype=np.float32).ravel()
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-12:
            q = q / q_norm  # optional normalization if your database vectors are normalized; keep consistent with indexing

        # constants (should match build)
        D = DIMENSION
        # load index metadata
        index_path = self.index_path
        coarse_centroids = np.load(os.path.join(index_path, "coarse_centroids.npy"))  # shape (n_coarse, D)
       
        n_coarse =256
        m = 8
        sub_dim = 8
        k_sub = 256

        # load codebook (m, k_sub, sub_dim)
        codebook = np.load(os.path.join(index_path, "pq_codebook.npy"))
        # offsets for inverted lists (uint64, length n_coarse+1), offsets in number-of-vectors units per list
        offsets = np.fromfile(os.path.join(index_path, "ivfpq_offsets.bin"), dtype=np.uint64)
        ivfpq_path = os.path.join(index_path, "ivfpq.bin")

        # 1) find top-nprobe coarse cells by L2 distance between q and coarse_centroid
        # compute squared L2 distances
        diffs = coarse_centroids - q  # shape (n_coarse, D)
        coarse_dists = np.sum(diffs * diffs, axis=1)
        top_coarse_idx = np.argpartition(coarse_dists, nprobe - 1)[:nprobe]
        top_coarse_idx = top_coarse_idx[np.argsort(coarse_dists[top_coarse_idx])]  # from closest to far
        # We'll need the coarse centroids for each chosen cell
        # Precompute query residual per chosen centroid when needed.

        n_candidates = top_k * candidate_multiplier
        candidate_heap = []  # min-heap of (approx_dist, vid) where smaller = better (L2)

        # compute how many bytes the codebook takes (written at file start)
       

        # For each coarse cell, read its list and compute approximate distances
        with open(ivfpq_path, "rb") as f:
            for c in top_coarse_idx:
                start = int(offsets[c])
                end = int(offsets[c + 1])
                n_vec = end - start
                if n_vec == 0:
                    continue

                # compute residual query: q - centroid_c
                centroid_c = coarse_centroids[c]
                query_residual = q - centroid_c
                sub_q = query_residual.reshape(m, sub_dim)  # shape (m, sub_dim)

                # compute dist_table for this centroid: for each subvector slot, a vector of size k_sub: || sub_q - codebook[i,k] ||^2
                # codebook shape (m, k_sub, sub_dim)
                # dist_table shape (m, k_sub)
                diff = codebook - sub_q[:, np.newaxis, :]  # broadcasting: (m,k_sub,sub_dim) - (m,1,sub_dim)
                dist_table = np.sum(diff * diff, axis=2)  # (m, k_sub)
                del diff

                # position in file to read this list:
                # file layout: [codebook bytes] [for each cell: ids(uint32) then codes(uint8 flattened)] contiguously
                # We stored each cell sequentially in that order during build.
                # We need to know the byte start for this cell. During building we wrote lists in order and tracked offsets in "vector counts".
                # To compute the byte offset we need to walk previous lists lengths; but we stored offsets in vector-count units.
                # We'll compute byte-position as: codebook_nbytes + sum_{i<cell}(len(list_i) * (4 + m))
                # That can be computed by reading offsets array (which stores cumulative counts) => offsets[c] gives number of vectors before cell c.
                byte_pos =  int(offsets[c]) * (4 + m)  # each vector stored as uint32 id (4) + m bytes codes (uint8)
                f.seek(byte_pos)
                block_bytes = f.read(n_vec * (4 + m))

                # parse ids and codes
                ids = np.frombuffer(block_bytes, dtype=np.uint32, count=n_vec, offset=0)
                codes = np.frombuffer(block_bytes, dtype=np.uint8, offset=n_vec * 4).reshape(n_vec, m)
                # now compute approx distances per vector using dist_table
                # dist_table: (m, k_sub)
                # codes.T shape: (m, n_vec)
                # pick dist_table[range(m), codes[i]] for each i
                # Efficient: dist_table[np.arange(m)[:,None], codes.T] -> shape (m, n_vec) -> sum over m -> (n_vec,)
                picked = dist_table[np.arange(m)[:, None], codes.T]  # (m, n_vec)
                approx_dists = picked.sum(axis=0)  # (n_vec,)
                del picked, dist_table, block_bytes

                # push candidates into a min-heap (we want smallest L2 distances)
                for dist, vid in zip(approx_dists, ids):
                    if len(candidate_heap) < n_candidates:
                        heapq.heappush(candidate_heap, (-dist, int(vid)))  # use -dist to keep largest negative = smallest dist if we want max-heap semantics
                    else:
                        # candidate_heap holds negative-dist; smallest dist corresponds to largest -dist
                        if -dist > candidate_heap[0][0]:
                            heapq.heapreplace(candidate_heap, (-dist, int(vid)))

        if not candidate_heap:
            return []

        # convert candidate_heap to list of unique vids (we might deduplicate)
        cand = {vid for _, vid in candidate_heap}
        # now exact re-ranking: load each candidate vector and compute exact similarity or exact L2
        # We'll compute dot(q, vec) if vectors and q are normalized; otherwise compute negative L2 for ranking.
        result_heap = []  # min-heap of (score, vid) where larger score is better

        # Decide exact metric: if your DB vectors are normalized, use dot product; else L2 (smaller is better)
        # We'll use dot product assuming normalization above. Keep consistent with indexing/build normalization.
        for vid in cand:
            vec = self.get_one_row(int(vid)).astype(np.float32)
            # If your vectors were normalized at index time, use dot; otherwise compute -L2 to get descending order
            exact_score = float(np.dot(vec, q))
            if len(result_heap) < top_k:
                heapq.heappush(result_heap, (exact_score, int(vid)))
            else:
                if exact_score > result_heap[0][0]:
                    heapq.heapreplace(result_heap, (exact_score, int(vid)))

        # sort results descending by score
        result_heap.sort(reverse=True)
        final_ids = [int(vid) for score, vid in result_heap]

        return final_ids

    ############################################################################
    # ---------------------------- Index building ---------------------------- #
    ############################################################################
    def _build_index(self, n_coarse=256, m=8, bits=8, batch_size=5000):
        """
        Build IVF-PQ index memory-safely.
        - n_coarse: number of coarse clusters
        - m: number of PQ sub-vectors
        - bits: number of bits per sub-vector (k_sub = 2^bits)
        - batch_size: used for PQ encoding batching
        """
        N, D = self._get_num_records(), DIMENSION
        assert D % m == 0, f"D={D} must be divisible by m={m}"
        sub_dim = D // m
        k_sub = 1 << bits

        data = self.get_all_rows()  # careful: loads all to memory; for huge DB you'd batch-training differently

        # remove old index
        if os.path.exists(self.index_path):
            shutil.rmtree(self.index_path)
        os.makedirs(self.index_path, exist_ok=True)

        # === 1. Coarse quantizer (L2) ===
        print("Training coarse quantizer...")

        coarse_km = MiniBatchKMeans(n_clusters=n_coarse, batch_size=10_000, random_state=DB_SEED_NUMBER)
        coarse_labels = coarse_km.fit_predict(data)
        coarse_cents = coarse_km.cluster_centers_.astype(np.float32)  # shape (n_coarse, D)


        #normalize centroid 
        coarse_cents = coarse_cents / (np.linalg.norm(coarse_cents, axis=1, keepdims=True) + 1e-12)

        np.save(os.path.join(self.index_path, "coarse_centroids.npy"), coarse_cents)

        # === 2. Compute residuals and train PQ codebooks on residuals ===
        print("Training PQ codebooks on residuals...")
        # residual for each vector: x - centroid_of_cell

        residuals = data - coarse_cents[coarse_labels]
        residuals = residuals / (np.linalg.norm(residuals, axis=1, keepdims=True) + 1e-12)
        
        codebook = np.zeros((m, k_sub, sub_dim), dtype=np.float32)

        for i in range(m):
            subvecs = residuals[:, i * sub_dim:(i + 1) * sub_dim]
            km = MiniBatchKMeans(n_clusters=k_sub, batch_size=20_000, random_state=DB_SEED_NUMBER)
            #normalize subvecs
           
            km.fit(subvecs)
            codebook[i] = km.cluster_centers_.astype(np.float32)

        
        np.save(os.path.join(self.index_path, "pq_codebook.npy"), codebook)
       
        # === 3. Encode all vectors using PQ (batching to avoid huge memory) ===
        print("Encoding vectors into PQ codes (memory-safe)...")
        codes = np.zeros((N, m), dtype=np.uint8)  # MUST be uint8 to store codes 0..k_sub-1
        for i in range(m):
            subvecs = residuals[:, i * sub_dim:(i + 1) * sub_dim]  # residual sub-vector
            cents = codebook[i]  # shape (k_sub, sub_dim)
            cents_sq = np.sum(cents * cents, axis=1)  # (k_sub,)

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch = subvecs[start:end]  # (B, sub_dim)
                batch_sq = np.sum(batch * batch, axis=1, keepdims=True)  # (B,1)
                cross = batch @ cents.T  # (B, k_sub)
                dists = batch_sq + cents_sq[None, :] - 2.0 * cross  # (B, k_sub)
                codes[start:end, i] = np.argmin(dists, axis=1).astype(np.uint8)

        # === 4. Build inverted lists (IVF) ===
        print("Building inverted lists...")
        lists = [[] for _ in range(n_coarse)]
        for idx, c in enumerate(coarse_labels):
            lists[c].append(idx)

        # === 5. Write index to disk: layout = [codebook bytes][for each cell: ids(uint32) then codes(uint8*m)] ===
        print("Writing final index...")
        offsets = np.zeros(n_coarse + 1, dtype=np.uint64)
        ivfpq_file = os.path.join(self.index_path, "ivfpq.bin")
        with open(ivfpq_file, "wb") as f:
           

            # write each cell's ids then codes
            total_written = 0
            for c in range(n_coarse):
                ids = np.array(lists[c], dtype=np.uint32)
                n_vec = ids.size
                offsets[c + 1] = offsets[c] + n_vec
                if n_vec == 0:
                    continue
                subset_codes = codes[ids].astype(np.uint8)  # shape (n_vec, m)
                ids.tofile(f)  # uint32 sequence
                subset_codes.tofile(f)  # raw uint8 bytes (n_vec * m)
                total_written += n_vec

        offsets.tofile(os.path.join(self.index_path, "ivfpq_offsets.bin"))
        # Save meta already saved
        print(f"IVF-PQ built! Index file: {ivfpq_file}, size: {os.path.getsize(ivfpq_file) / 1e6:.1f} MB")
#ivf pq better 


# 10M	score	-192.33333333333334	time	0.62	RAM	1.29 MB


# 1M	score	-108.66666666666667	time	0.07	RAM	0.23 MB
# 10M	score	-192.33333333333334	time	0.65	RAM	1.39 MB
# 15M	score	-174.0	time	1.21	RAM	3.21 MB






# 1M	score	-108.66666666666667	time	0.23	RAM	0.23 MB   before normailzation distance

# 1M	score	-108.66666666666667	time	0.07	RAM	0.16 MB    after 

