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

####################################################################################
####################################################################################
####################################################################################


    


    def retrieve(self, query, top_k=5, n_probe=None, chunk_size=50):
        self.no_centroids = 10_000
        self.index_path = f"index_10M_{self.no_centroids}_centroids_f16"
        query = np.asarray(query, dtype=np.float32).squeeze()

        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            query_norm = 1.0
        normalized_query = query / query_norm

        centers_path = os.path.join(self.index_path, "centroids.npy")
        if not os.path.exists(centers_path):
            del query, normalized_query
            return []

        centroids = np.load(centers_path, mmap_mode="r")
        num_centroids, dim = centroids.shape

        # Auto n_probe choice
        if n_probe is None:
            n_probe = 14 if num_centroids <= 15_000_000 else 8

        heap = []
        # Loop over batches
        for start in range(0, num_centroids, 2000):
            end = min(start + 2000, num_centroids)

            batch = centroids[start:end]

            sims = batch.dot(normalized_query)

            # Insert into heap
            for i, s in enumerate(sims):
                cid = start + i
                if len(heap) < n_probe:
                    heapq.heappush(heap, (s, cid))
                else:
                    heapq.heappushpop(heap, (s, cid))

            del sims

        del centroids
          # Extract top n_probe IDs sorted descending similarity
        top = heapq.nlargest(n_probe, heap)
        nearest_centroids = np.array([cid for (_, cid) in top], dtype=np.int32)
        del heap, top


        header_arr = np.fromfile(os.path.join(self.index_path, "index_header.bin"), dtype=np.uint32)
        header_arr = header_arr.reshape(-1, 2)   # shape: (num_centroids, 2)


        all_ids = []
        flat_index_path = os.path.join(self.index_path, "all_indices.bin")
        for c in nearest_centroids:
            offset  = header_arr[c, 0]   # first column
            length  = header_arr[c, 1]   # second column
            if length == 0:
                continue
            ids_mm = np.memmap(flat_index_path, dtype=np.uint32, mode="r",
                            offset=offset, shape=(length,))
            db_ids = ids_mm[:]
            del ids_mm  
            all_ids.extend(db_ids)


        all_ids.sort()

        grouped_ids = self.group_ids_by_window_fast(all_ids, window=1500)
        del all_ids 
        top_heap = []
        row_size = DIMENSION * 4  
        with open(self.db_path, "rb") as f: 
            for group in grouped_ids:
                start_id = group[0]
                end_id = group[-1]

                # Read contiguous block for this group
                f.seek(start_id * row_size)
                block_len = end_id - start_id + 1
                block_data = np.frombuffer(f.read(block_len * row_size), dtype=np.float32)
                block_data = block_data.reshape(block_len, DIMENSION)

                # Select only the rows we need
                relative_indices = group - start_id
                vecs = block_data[relative_indices, :]

                # Compute scores
                scores = vecs.dot(normalized_query)

                # Maintain top-k heap
                for score, id in zip(scores, group):
                    if len(top_heap) < top_k:
                        heapq.heappush(top_heap, (score, id))
                    else:
                        heapq.heappushpop(top_heap, (score, id))

                # Free memory
                del vecs, scores, block_data
        del grouped_ids
        results = [idx for score, idx in heapq.nlargest(top_k, top_heap)]
        del top_heap
        return results


    

    def _build_index(self):
       
        # sqrt(N) rule
        self.no_centroids = 9_000
        self.index_path = f"index_10M_{self.no_centroids}_centroids"
        # data is a reference to the memmap object, not the data in RAM
        data = self.get_all_rows()

        kmeans = MiniBatchKMeans(
            n_clusters=self.no_centroids,
            init="k-means++",   
            batch_size=10_000,
            random_state=42
        )

        kmeans.fit(data)

        # labels and centers are new arrays created in RAM
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_.astype(np.float32)

        # Added deletion of the memmap reference and the kmeans object
        del data
        del kmeans
       
         

        if os.path.exists(self.index_path):
            shutil.rmtree(self.index_path)  # remove old index if any
        os.makedirs(self.index_path, exist_ok=True)
               
        if not os.path.isdir(self.index_path):
            os.makedirs(self.index_path, exist_ok=True)
        # 1. Build up a list of (cluster_id, indices_array)
        cluster_infos = []
        for cid in range(self.no_centroids):
            indices = np.where(labels == cid)[0].astype(np.uint32)
            cluster_infos.append((cid, indices))

      
        header = []

        flat_path = os.path.join(self.index_path, "all_indices.bin")
        with open(flat_path, "wb") as f:
            offset = 0
            for cid, inds in cluster_infos:
                length = inds.size
                f.write(inds.tobytes())
                header.append([offset, length])
                offset += length * inds.dtype.itemsize

        # Convert to a matrix (2 columns: offset, length)
        header_matrix = np.array(header, dtype=np.uint32)

        # Save matrix as binary file
        header_bin_path = os.path.join(self.index_path, "index_header.bin")
        header_matrix.tofile(header_bin_path)

        norms = np.linalg.norm(centers, axis=1, keepdims=True)
        centers = centers / (norms + 1e-12)
        # 4. Save centers as before
        np.save(os.path.join(self.index_path, "centroids.npy"), centers)