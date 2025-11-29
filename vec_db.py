import json
import numpy as np
import os
import shutil
import tqdm
import heapq
from sklearn.cluster import MiniBatchKMeans
from typing import Annotated
import varint

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

    def unpack_packed_codes(self, packed_block):
        # packed_block: uint8 array shape (n, packed_cols)
        # output: uint8 array shape (n, M)
        # high nibble = code for 2*i (>>4), low nibble = code for 2*i+1 (&0x0F)
        M = 8  # number of sub-vectors (hard-coded to your m)
        if packed_block.ndim == 1:
            packed_block = packed_block.reshape(1, -1)
        n = packed_block.shape[0]
        codes = np.empty((n, M), dtype=np.uint8)
        # vectorized unpack
        high = (packed_block >> 4).astype(np.uint8)
        low  = (packed_block & 0x0F).astype(np.uint8)
        codes[:, 0::2] = high
        codes[:, 1::2] = low
        return codes

    ############################################################################
    # -------------------------- INDEX BUILDING ------------------------------ #
    ############################################################################
    def _build_index(self, n_coarse=256, m=8, bits=8, batch_size=5000):
        """
        Build IVF-PQ index: one file for IDs (varint), one file for PQ codes (packed).
        Ensures IDs and codes are written in the SAME (sorted) order so retrieval maps them correctly.
        """
        N, D = self._get_num_records(), DIMENSION
        assert D % m == 0, f"D={D} must be divisible by m={m}"
        sub_dim = D // m
        k_sub = 1 << bits
        data = self.get_all_rows()
        print(f"Building IVF-PQ index: N={N}, D={D}, n_coarse={n_coarse}, m={m}, k_sub={k_sub}")

        # clean old index
        if os.path.exists(self.index_path):
            shutil.rmtree(self.index_path)
        os.makedirs(self.index_path, exist_ok=True)

        # 1. Coarse quantizer
        coarse_km = MiniBatchKMeans(n_clusters=n_coarse, batch_size=10_000, random_state=DB_SEED_NUMBER)
        coarse_labels = coarse_km.fit_predict(data)
        coarse_cents = coarse_km.cluster_centers_.astype(np.float32)
        np.save(os.path.join(self.index_path, "coarse_centroids.npy"), coarse_cents)
        print("Coarse quantizer trained.")
        # 2. PQ codebooks
        residuals = data - coarse_cents[coarse_labels]
        codebook = np.zeros((m, k_sub, sub_dim), dtype=np.float32)
        for i in range(m):
            subvecs = residuals[:, i*sub_dim:(i+1)*sub_dim]
            km = MiniBatchKMeans(n_clusters=k_sub, batch_size=20_000, random_state=DB_SEED_NUMBER)
            km.fit(subvecs)
            codebook[i] = km.cluster_centers_.astype(np.float32)
        np.save(os.path.join(self.index_path, "pq_codebook.npy"), codebook)
        print("PQ codebooks trained.")

        # 3. Encode PQ codes (indices into codebook)
        codes = np.zeros((N, m), dtype=np.uint8)
        for i in range(m):
            subvecs = residuals[:, i*sub_dim:(i+1)*sub_dim]
            cents = codebook[i]
            cents_sq = np.sum(cents * cents, axis=1)
            for start in range(0, N, batch_size):
                end = min(start+batch_size, N)
                batch = subvecs[start:end]
                batch_sq = np.sum(batch*batch, axis=1, keepdims=True)
                cross = batch @ cents.T
                dists = batch_sq + cents_sq[None,:] - 2.0*cross
                codes[start:end, i] = np.argmin(dists, axis=1).astype(np.uint8)
        print("PQ codes encoded.")

        # 4. Build inverted lists and write files
        lists = [[] for _ in range(n_coarse)]
        for idx, c in enumerate(coarse_labels):
            lists[c].append(idx)

        offsets = np.zeros(n_coarse+1, dtype=np.uint64)
        byte_positions = []
        current_byte_pos = 0

        packed_cols = m // 2
        ivf_indices_file = os.path.join(self.index_path, "ivf_indices.bin")
        ivf_codes_file   = os.path.join(self.index_path, "ivf_codes.bin")

        print(f"Writing inverted lists to {ivf_indices_file} and {ivf_codes_file}")

        with open(ivf_indices_file, "wb") as f_ids, open(ivf_codes_file, "wb") as f_codes:
            for c in range(n_coarse):
                ids = np.array(lists[c], dtype=np.uint32)
                n_vec = ids.size
                offsets[c+1] = offsets[c] + n_vec

                # Track byte position BEFORE writing this cluster
                byte_positions.append(current_byte_pos)

                if n_vec == 0:
                    continue

                # --- IMPORTANT: sort ids and reorder codes to the same sorted order ---
                ids_sorted = np.sort(ids)
                # Delta encode sorted ids
                deltas = np.empty_like(ids_sorted)
                deltas[0] = ids_sorted[0]
                deltas[1:] = ids_sorted[1:] - ids_sorted[:-1]
                cluster_bytes = bytearray()
                for d in deltas:
                    cluster_bytes.extend(varint.encode(int(d)))
                bytes_written = len(cluster_bytes)
                f_ids.write(cluster_bytes)
                current_byte_pos += bytes_written

                # write PQ codes in the same sorted order
                subset_codes = codes[ids_sorted]  # align order
                packed = np.empty((n_vec, packed_cols), dtype=np.uint8)
                for i_pack in range(packed_cols):
                    high = subset_codes[:,2*i_pack]
                    low  = subset_codes[:,2*i_pack+1]
                    packed[:,i_pack] = (high<<4) | (low & 0x0F)
                f_codes.write(packed.tobytes())

        # Save id byte positions and offsets
        np.save(os.path.join(self.index_path, "id_byte_positions.npy"), np.array(byte_positions, dtype=np.int64))
        offsets.tofile(os.path.join(self.index_path, "ivfpq_offsets.bin"))
        print("Inverted lists written and offsets saved.")

    ############################################################################
    # -------------------------- Retrieval (search) --------------------------- #
    ############################################################################
    def _read_cluster_ids(self, file_path, byte_start, byte_end, num_vectors):
        """
        Read varint-encoded (delta) IDs from byte_start up to byte_end (exclusive) or until num_vectors decoded.
        Returns absolute IDs (uint32).
        """
        if num_vectors == 0:
            return np.array([], dtype=np.uint32)

        with open(file_path, "rb") as f:
            f.seek(byte_start)
            to_read = None
            if byte_end is not None:
                to_read = max(0, byte_end - byte_start)
                data = f.read(to_read)
            else:
                data = f.read()

        decoded = []
        x = 0
        shift = 0
        for b in data:
            x |= (b & 0x7F) << shift
            if b < 128:  # last byte
                decoded.append(x)
                x = 0
                shift = 0
            else:
                shift += 7
            if len(decoded) >= num_vectors:
                break

        if not decoded:
            return np.array([], dtype=np.uint32)

        # Convert deltas -> absolute IDs
        return np.cumsum(np.array(decoded[:num_vectors], dtype=np.uint32))

    def retrieve(self, query, top_k=10, nprobe=255, candidate_multiplier=8):
        """
        IVF-PQ retrieve. Keeps same file layout and reads IDs and codes that were written in the SAME order.
        """
        q = np.asarray(query, dtype=np.float32).ravel()
        D = DIMENSION
        m = 8
        sub_dim = D // m
        packed_cols = m // 2
        index_path = self.index_path

        coarse_centroids = np.load(os.path.join(index_path, "coarse_centroids.npy"))
        codebook = np.load(os.path.join(index_path, "pq_codebook.npy"))
        n_coarse = coarse_centroids.shape[0]
        offsets = np.fromfile(os.path.join(index_path, "ivfpq_offsets.bin"), dtype=np.uint64)
        id_byte_positions = np.load(os.path.join(index_path, "id_byte_positions.npy"))

        ivf_indices_file = os.path.join(index_path, "ivf_indices.bin")
        ivf_codes_file   = os.path.join(index_path, "ivf_codes.bin")
        idx_file_size = os.path.getsize(ivf_indices_file)

        # choose coarse clusters to probe
        diffs = coarse_centroids - q
        coarse_dists = np.sum(diffs*diffs, axis=1)
        top_coarse = np.argpartition(coarse_dists, min(nprobe, n_coarse)-1)[:min(nprobe, n_coarse)]
        top_coarse = top_coarse[np.argsort(coarse_dists[top_coarse])]

        n_candidates = top_k * candidate_multiplier
        candidate_heap = []

        with open(ivf_indices_file, "rb") as f_ids, open(ivf_codes_file, "rb") as f_codes:
            for c in top_coarse:
                start = int(offsets[c])
                end   = int(offsets[c+1])
                n_vec = end - start
                if n_vec == 0:
                    continue

                # compute byte range to read for this cluster's ids
                byte_start = int(id_byte_positions[c])
                if c+1 < len(id_byte_positions):
                    byte_end = int(id_byte_positions[c+1])
                else:
                    byte_end = idx_file_size

                ids = self._read_cluster_ids(f_ids.name, byte_start, byte_end, n_vec)
                if ids.size == 0:
                    continue

                # compute PQ distance table for the query w.r.t. this centroid
                centroid_c = coarse_centroids[c]
                q_res = q - centroid_c
                sub_q = q_res.reshape(m, sub_dim)  # (m, sub_dim)
                diff = codebook - sub_q[:, None, :]  # (m, k_sub, sub_dim)
                dist_table = np.sum(diff*diff, axis=2)  # (m, k_sub)
                del diff

                # read packed PQ codes for this cluster (codes were written in the same order as ids_sorted)
                f_codes.seek(start * packed_cols)  # start is #vectors before this cluster
                packed = np.frombuffer(f_codes.read(n_vec*packed_cols), dtype=np.uint8)
                if packed.size != n_vec*packed_cols:
                    # read error / corruption
                    continue
                packed = packed.reshape(n_vec, packed_cols)
                codes = self.unpack_packed_codes(packed)  # (n_vec, m)

                # compute approximate distances
                # we assume ids are in the same order as codes (we wrote both sorted)
                picked = dist_table[np.arange(m)[:, None], codes.T]  # (m, n_vec)
                approx_dists = picked.sum(axis=0)  # length n_vec
                del picked

                # push into candidate heap (max-heap via negative values as in your original logic)
                for i_local in range(n_vec):
                    vid = int(ids[i_local])
                    dist = float(approx_dists[i_local])
                    if len(candidate_heap) < n_candidates:
                        heapq.heappush(candidate_heap, (-dist, vid))
                    else:
                        if -dist > candidate_heap[0][0]:
                            heapq.heapreplace(candidate_heap, (-dist, vid))

        if not candidate_heap:
            return []

        # exact L2 re-rank among unique candidates
        candidates = {vid for _, vid in candidate_heap}
        result_heap = []
        for vid in candidates:
            vec = self.get_one_row(int(vid)).astype(np.float32)
            score = float(-np.sum((vec - q)**2))  # higher is better
            if len(result_heap) < top_k:
                heapq.heappush(result_heap, (score, int(vid)))
            else:
                if score > result_heap[0][0]:
                    heapq.heapreplace(result_heap, (score, int(vid)))

        # return sorted best ids (descending score)
        result_heap.sort(reverse=True)
        
        result = [vid for score, vid in result_heap]
        print(result)
        return result
