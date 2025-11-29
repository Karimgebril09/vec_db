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
        M = 8  # number of sub-vectors
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



    def encode_and_save_indices(self,path, indices):
        """
        Takes an array/list of uint32 indices, sorts them, delta-encodes them,
        varint-encodes them, and saves to path.
        """

        if len(indices) == 0:
            # Write empty file
            open(path, "wb").close()
            return

        indices = np.sort(indices.astype(np.uint32))

        # Delta encode
        deltas = np.empty_like(indices)
        deltas[0] = indices[0]
        deltas[1:] = indices[1:] - indices[:-1]

        # Varint encode all deltas
        encoded = bytearray()
        for d in deltas:
            encoded.extend(varint.encode(int(d)))

        # Save to disk
        with open(path, "wb") as f:
            f.write(encoded)


    def _read_cluster_ids(self, file_path, byte_start, num_vectors):
      """
      Read specific number of varint-encoded IDs starting from a byte position.
      """
      if num_vectors == 0:
          return np.array([], dtype=np.uint32)

      with open(file_path, "rb") as f:
          f.seek(byte_start)
          data = f.read()  # Read remaining file from this position

      # Decode varints
      decoded = []
      x = 0
      shift = 0
      for b in data:
          x |= (b & 0x7F) << shift
          if b < 128:  # last byte of this varint
              decoded.append(x)
              x = 0
              shift = 0
          else:
              shift += 7

          # Stop once we have enough IDs
          if len(decoded) >= num_vectors:
              break

      if not decoded:
          return np.array([], dtype=np.uint32)

      # Convert deltas → absolute
      return np.cumsum(np.array(decoded[:num_vectors], dtype=np.uint32))

    ############################################################################
    # -------------------------- Retrieval (search) --------------------------- #
    ############################################################################
    ############################################################################
    def _build_index(self, n_coarse=256, m=8, bits=8, batch_size=5000):
      """
      Build IVF-PQ index: one file for IDs (varint), one file for PQ codes.
      """
      N, D = self._get_num_records(), DIMENSION
      assert D % m == 0, f"D={D} must be divisible by m={m}"
      sub_dim = D // m
      k_sub = 1 << bits
      data = self.get_all_rows()

      # clean old index
      if os.path.exists(self.index_path):
          shutil.rmtree(self.index_path)
      os.makedirs(self.index_path, exist_ok=True)

      # 1. Coarse quantizer
      coarse_km = MiniBatchKMeans(n_clusters=n_coarse, batch_size=10_000, random_state=DB_SEED_NUMBER)
      coarse_labels = coarse_km.fit_predict(data)
      coarse_cents = coarse_km.cluster_centers_.astype(np.float32)
      np.save(os.path.join(self.index_path, "coarse_centroids.npy"), coarse_cents)

      # 2. PQ codebooks
      residuals = data - coarse_cents[coarse_labels]
      codebook = np.zeros((m, k_sub, sub_dim), dtype=np.float32)
      for i in range(m):
          subvecs = residuals[:, i*sub_dim:(i+1)*sub_dim]
          km = MiniBatchKMeans(n_clusters=k_sub, batch_size=20_000, random_state=DB_SEED_NUMBER)
          km.fit(subvecs)
          codebook[i] = km.cluster_centers_.astype(np.float32)
      np.save(os.path.join(self.index_path, "pq_codebook.npy"), codebook)

      # 3. Encode PQ codes
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

      # 4. Build inverted lists and track byte positions
      lists = [[] for _ in range(n_coarse)]
      for idx, c in enumerate(coarse_labels):
          lists[c].append(idx)

      offsets = np.zeros(n_coarse+1, dtype=np.uint64)
      byte_positions = []  # Track byte positions for each cluster
      current_byte_pos = 0

      packed_cols = m // 2
      # files for IDs and PQ codes
      ivf_indices_file = os.path.join(self.index_path, "ivf_indices.bin")
      ivf_codes_file   = os.path.join(self.index_path, "ivf_codes.bin")

      with open(ivf_indices_file, "wb") as f_ids, open(ivf_codes_file, "wb") as f_codes:
          for c in range(n_coarse):
              ids = np.array(lists[c], dtype=np.uint32)
              n_vec = ids.size
              offsets[c+1] = offsets[c] + n_vec

              # Track byte position BEFORE writing this cluster
              byte_positions.append(current_byte_pos)

              if n_vec == 0:
                  continue

              # save IDs using varint
              cluster_bytes = bytearray()
              indices = np.sort(ids.astype(np.uint32))
              # Delta encode
              deltas = np.empty_like(indices)
              deltas[0] = indices[0]
              deltas[1:] = indices[1:] - indices[:-1]
              # Varint encode all deltas
              for d in deltas:
                  cluster_bytes.extend(varint.encode(int(d)))

              bytes_written = len(cluster_bytes)
              f_ids.write(cluster_bytes)
              current_byte_pos += bytes_written

              # save PQ codes
              subset_codes = codes[ids]
              packed = np.empty((n_vec, packed_cols), dtype=np.uint8)
              for i in range(packed_cols):
                  high = subset_codes[:,2*i]
                  low  = subset_codes[:,2*i+1]
                  packed[:,i] = (high<<4) | (low & 0x0F)
              f_codes.write(packed.tobytes())

      # Save byte positions for ID reading
      np.save(os.path.join(self.index_path, "id_byte_positions.npy"), np.array(byte_positions, dtype=np.int64))
      offsets.tofile(os.path.join(self.index_path, "ivfpq_offsets.bin"))


    def retrieve(self, query, top_k=10, nprobe=255, candidate_multiplier=8):
      """
      IVF-PQ retrieve with separate files for IDs and PQ codes.
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

      diffs = coarse_centroids - q
      coarse_dists = np.sum(diffs*diffs, axis=1)
      top_coarse = np.argpartition(coarse_dists, nprobe)[:nprobe]
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

              # FIXED: Read only the IDs for this specific cluster
              byte_start = id_byte_positions[c]
              ids = self._read_cluster_ids(f_ids.name, byte_start, n_vec)

              centroid_c = coarse_centroids[c]
              q_res = q - centroid_c
              sub_q = q_res.reshape(m, sub_dim)
              diff = codebook - sub_q[:, None, :]
              dist_table = np.sum(diff*diff, axis=2)
              del diff

              # read packed PQ codes
              f_codes.seek(start*packed_cols)
              packed = np.frombuffer(f_codes.read(n_vec*packed_cols), dtype=np.uint8).reshape(n_vec, packed_cols)
              codes = self.unpack_packed_codes(packed)

              picked = dist_table[np.arange(m)[:, None], codes.T]
              approx_dists = picked.sum(axis=0)
              del picked

              # FIXED: Use the correct number of IDs
              min_len = min(len(approx_dists), len(ids))
              for i in range(min_len):
                  dist, vid = approx_dists[i], ids[i]
                  if len(candidate_heap) < n_candidates:
                      heapq.heappush(candidate_heap, (-dist, int(vid)))
                  else:
                      if -dist > candidate_heap[0][0]:
                          heapq.heapreplace(candidate_heap, (-dist, int(vid)))

      if not candidate_heap:
          return []

      # exact L2 re-rank
      candidates = {vid for _, vid in candidate_heap}
      result_heap = []
      for vid in candidates:
          vec = self.get_one_row(int(vid)).astype(np.float32)
          score = float(-np.sum((vec - q)**2))
          if len(result_heap) < top_k:
              heapq.heappush(result_heap, (score, int(vid)))
          else:
              if score > result_heap[0][0]:
                  heapq.heapreplace(result_heap, (score, int(vid)))

      result_heap.sort(reverse=True)
      result =[vid for score, vid in result_heap]
      print (result)
      return result
