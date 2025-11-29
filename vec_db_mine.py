import json
import numpy as np
import os
import shutil
import tqdm
import heapq
from sklearn.cluster import KMeans, MiniBatchKMeans
from typing import Annotated
import time
import gc

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 64

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index.dat", new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
    
    def generate_database(self, size: int) -> None:
        # rng = np.random.default_rng(DB_SEED_NUMBER)
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
        #TODO: might change to call insert in the index, if you need
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"

    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
    
    def get_all_ids_rows_seek(self, ids):
        """
        Load rows corresponding to a list of IDs from disk efficiently using seek/read.
        Handles sorted or non-contiguous IDs with jumps.

        ids: list or np.array of integers
        returns: np.array of shape (len(ids), DIMENSION)
        """
        row_size = DIMENSION * 4  # float32 = 4 bytes
        row_size = DIMENSION * 4 
        rows = np.empty((len(ids), DIMENSION), dtype=np.float32)

        # Group contiguous indices into blocks
        blocks = []
        start = ids[0]
        prev = ids[0]
        for i in ids[1:]:
            if i != prev + 1:
                blocks.append((start, prev))
                start = i
            prev = i
        blocks.append((start, prev))

        # Read each block using seek
        idx = 0
        with open(self.db_path, "rb") as f:
            for start_block, end_block in blocks:
                f.seek(start_block * row_size)
                block_len = end_block - start_block + 1
                block_data = np.frombuffer(f.read(block_len * row_size), dtype=np.float32)
                block_data = block_data.reshape(block_len, DIMENSION)
                rows[idx:idx+block_len, :] = block_data
                idx += block_len

        return rows
        
    

    def group_ids_by_window_fast(self, all_ids, window):
        all_ids = np.asarray(all_ids)
        n = all_ids.size
        if n == 0:
            return []

        # Compute per-element group max = all_ids[i] + window
        limits = all_ids + window

        # For each i, ends[i] = first index where id > limits[i]
        ends = np.searchsorted(all_ids, limits + 1, side="left")

        # We find group starts: a new group starts where i == previous group's end
        starts = [0]
        for i in range(1, n):
            if ends[i-1] == i:
                starts.append(i)

        starts = np.array(starts, dtype=np.int64)
        group_ends = ends[starts]
        groups = [all_ids[s:e] for s, e in zip(starts, group_ends)]
        return groups

    
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity
    
    # def unpack_packed_codes(self, packed_block):
    #     # packed_block: uint8 array shape (n, packed_cols)
    #     # output: uint8 array shape (n, M)
    #     # high nibble = code for 2*i (>>4), low nibble = code for 2*i+1 (&0x0F)
    #     M = 8  # number of sub-vectors
    #     if packed_block.ndim == 1:
    #         packed_block = packed_block.reshape(1, -1)
    #     n = packed_block.shape[0]
    #     codes = np.empty((n, M), dtype=np.uint8)
    #     # vectorized unpack
    #     high = (packed_block >> 4).astype(np.uint8)
    #     low  = (packed_block & 0x0F).astype(np.uint8)
    #     codes[:, 0::2] = high
    #     codes[:, 1::2] = low
    #     return codes

####################################################################################
####################################################################################
####################################################################################

    

    def retrieve(self, query, top_k=10, nprobe=5):
        M = 8
        d_sub = DIMENSION // M
        K = 256
        N = self._get_num_records()
        self.index_path = f"index_IVFPQ_1M_14_8_256_centroids"
        coarse = np.load(os.path.join(self.index_path, "coarse_centroids.npy"))  # (C,D)
        C, D = coarse.shape

        q = query.astype(np.float32).squeeze()  
        q /= np.linalg.norm(q) + 1e-12

        scores = coarse @ q  
        probe_ids = np.argsort(scores)[-nprobe:][::-1]


        header = np.fromfile(os.path.join(self.index_path, "start.bin"), dtype=np.uint32).reshape(-1,2)
        starts = header[:,0]
        length = header[:,1]
        indices = np.memmap(os.path.join(self.index_path, "indices.bin"), dtype=np.uint32,shape=(N,), mode='r')
        codes = np.memmap(os.path.join(self.index_path, "codebook.bin"), dtype=np.uint8,shape=(N, M), mode='r')

        # FIXED: reshape PQ centroids properly
        pq_centroids_all = np.memmap(os.path.join(self.index_path, "code_book_centroids.dat"),
                                    dtype=np.float32,shape=(C*K,D), mode='r')

        all_ids = []
        all_distances = []

        for cid in probe_ids:
            start = int(starts[cid])
            L = int(length[cid])
            if L == 0:
                continue

            ids_block = indices[start:start+L]
            codes_block = codes[start:start+L]

            DT = np.zeros((K,M), dtype=np.float32)
            for m in range(M):
                q_sub = q[m*d_sub:(m+1)*d_sub]
                all_centroids = np.array(pq_centroids_all[cid*K:(cid+1)*K,m*d_sub:(m+1)*d_sub ])  
                DT[:,m] = -(all_centroids @ q_sub)
                
            dists = np.zeros(L, dtype=np.float32)
            for i in range(L):
                code = codes_block[i]
                sim = 0.0
                for m in range(M):
                    sim += DT[code[m], m]
                dists[i] = sim

            all_ids.append(ids_block)
            all_distances.append(dists)

        if len(all_ids) == 0:
            return []

        all_ids = np.concatenate(all_ids)
        all_distances = np.concatenate(all_distances)

# --- FIXED: Use negative array to find highest similarities ---
        topk_idx = np.argpartition(-all_distances, top_k)[:top_k]
        
        # Sort the top-k results by similarity (highest first)
        final_idx = topk_idx[np.argsort(all_distances[topk_idx])] 

        return list(zip(all_ids[final_idx], all_distances[final_idx]))




    

    def _build_index(self):
        self.index_path = f"index_IVFPQ_1M_14_8_256_centroids"
        self.no_coarse_centroids = 14
        data = self.get_all_rows()

        kmeans = MiniBatchKMeans(
            n_clusters=self.no_coarse_centroids,
            init="k-means++",   
            batch_size=50_000,
            random_state=42
        )
        kmeans.fit(data)
        labels = kmeans.labels_
        coarse_centers = kmeans.cluster_centers_.astype(np.float32)

        norms = np.linalg.norm(coarse_centers, axis=1, keepdims=True)
        coarse_centers = coarse_centers / (norms + 1e-12)

        del kmeans
       
         
        if os.path.exists(self.index_path):
            shutil.rmtree(self.index_path)  # remove old index if any
        os.makedirs(self.index_path, exist_ok=True)

        np.save(os.path.join(self.index_path, "coarse_centroids.npy"), coarse_centers)

        
        cluster_info = {}   # NEW: {cid : (start_offset, sorted_ids)}
        header = []
        all_indices = []
        offset = 0
        for cid in range(self.no_coarse_centroids):
            ids = np.where(labels == cid)[0].astype(np.uint32)
            ids = np.sort(ids)             
            all_indices.append(ids)
            cluster_info[cid] = (offset, ids) 
            header.append([offset, ids.size])
            offset += ids.size

        all_indices_flat = np.concatenate(all_indices)
        all_indices_flat.tofile(os.path.join(self.index_path, "indices.bin"))
        np.array(header, dtype=np.uint32).tofile(os.path.join(self.index_path, "start.bin"))

        # 6. Train PQ for each coarse cluster
        M = 8   # number of sub-vectors
        K = 256  # number of PQ centroids per sub-vector
        data = self.get_all_rows()
        N, D = data.shape
        subvector_len = DIMENSION // M
        pq_centroids_list = []
        codebook = np.zeros((data.shape[0], M), dtype=np.uint8) 

        for cid in range(self.no_coarse_centroids):
            start_index, ids = cluster_info[cid]  
            cluster_data = data[ids]               # <-- (3) direct indexing
            cluster_len = ids.size
            sub_centroids_per_cluster = []

            for m in range(M):
                sub_vectors = cluster_data[:, m*subvector_len:(m+1)*subvector_len]

                pq_kmeans = MiniBatchKMeans(
                    n_clusters=K,
                    batch_size=10_000,
                    random_state=42
                )
                pq_kmeans.fit(sub_vectors)
                sub_centroids = pq_kmeans.cluster_centers_.astype(np.float32)
                sub_centroids /= (np.linalg.norm(sub_centroids, axis=1, keepdims=True) + 1e-12)
                sub_centroids_per_cluster.append(sub_centroids)

                labels_sub = pq_kmeans.predict(sub_vectors).astype(np.uint8)
                codebook[start_index : start_index + cluster_len, m] = labels_sub

                del pq_kmeans, sub_vectors

            pq_centroids_list.append(np.hstack(sub_centroids_per_cluster))

            del cluster_data

        pq_all = np.vstack(pq_centroids_list)
        print(pq_all.shape)
        pq_all.tofile(os.path.join(self.index_path, "code_book_centroids.dat"))
        codebook.tofile(os.path.join(self.index_path, "codebook.bin"))
        print("PQ centroids and codes saved.")

        del data, pq_all, codebook