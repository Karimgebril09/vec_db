import json
import numpy as np
import os
import shutil
import tqdm
import heapq
from sklearn.cluster import KMeans, MiniBatchKMeans
from typing import Annotated


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
    
    def get_all_ids_rows(self, ids) -> np.ndarray:
        """
        Load only the requested rows from the memmap, without loading all data in RAM.
        Updated: Instead of loading all vectors into memory, we load only the requested batch.
        """
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        
        # Sort IDs to access memmap sequentially (faster disk read)
        sorted_idx = np.argsort(ids)
        sorted_ids = np.array(ids)[sorted_idx]
        
        # Load only selected rows
        result = np.empty((len(ids), DIMENSION), dtype=np.float32)
        result[sorted_idx] = vectors[sorted_ids]
        
        del vectors
        return result
    
    
    
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
        self.no_centroids = 7000
        self.index_path = f"index_10M_{self.no_centroids}_centroids"
        query = np.asarray(query, dtype=np.float32).squeeze()

        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            query_norm = 1.0
        normalized_query = query / query_norm

        centers_path = os.path.join(self.index_path, "centroids.npy")
        if not os.path.exists(centers_path):
            del query, normalized_query
            return []

        # Load centroids with memory-mapping to save RAM
        centroids = np.load(centers_path, mmap_mode="r")
        n_centroids = centroids.shape[0]

        if n_probe is None:
            num_records = self._get_num_records()
            n_probe = 12 if num_records <= 15_000_000 else 10

        # batch_size = 50
        # min_heap = []

        # for start in range(0, n_centroids, batch_size):
        #     end = min(start + batch_size, n_centroids)
        #     batch = centroids[start:end]  # only this batch is loaded in RAM
        #     sims = batch.dot(normalized_query)

        #     for i, score in enumerate(sims):
        #         centroid_index = start + i
        #         if len(min_heap) < n_probe:
        #             heapq.heappush(min_heap, (score, centroid_index))
        #         elif score > min_heap[0][0]:
        #             heapq.heappushpop(min_heap, (score, centroid_index))
            
        #     del batch, sims


        # nearest_centroids = [centroid_index for score, centroid_index in heapq.nlargest(n_probe, min_heap)]
        # del centroids
        sims = centroids.dot(normalized_query)
        nearest_centroids = np.argpartition(sims, -n_probe)[-n_probe:]
        # sort descending
        # nearest_centroids_idx = nearest_centroids_idx[np.argsort(sims[nearest_centroids_idx])[::-1]]
        del sims
        del centroids


        header_arr = np.fromfile(os.path.join(self.index_path, "index_header.bin"), dtype=np.uint32)
        header_arr = header_arr.reshape(-1, 2)   # shape: (num_centroids, 2)


        # Prepare top-k heap
        top_heap = []

        flat_index_path = os.path.join(self.index_path, "all_indices.bin")

        # For each centroid to probe:
        for c in nearest_centroids:
            offset  = header_arr[c, 0]   # first column
            length  = header_arr[c, 1]   # second column
            if length == 0:
                continue

            # Process each chunk by remapping the memmap for that chunk
            for start in range(0, length, chunk_size):
                cur_len = min(chunk_size, length - start)
                # Remap only this chunk
                ids_mm = np.memmap(flat_index_path, dtype=np.uint32, mode="r",
                                offset=offset + start * np.dtype(np.uint32).itemsize,
                                shape=(cur_len,))

                chunk_ids = ids_mm[:]  
                del ids_mm  # free memmap

                
                vecs = self.get_all_ids_rows(chunk_ids)
                scores = vecs.dot(normalized_query)
                for score, id in zip(scores, chunk_ids):
                    if len(top_heap) < top_k:
                        heapq.heappush(top_heap, (score, id))
                    else:
                        heapq.heappushpop(top_heap, (score, id))

                del scores, chunk_ids


        # Extract and return top-k IDs sorted by score
        results = [idx for score, idx in heapq.nlargest(top_k, top_heap)]
        del top_heap
       
        return results


    

    def _build_index(self):
       
        # sqrt(N) rule
        self.no_centroids = 7000
        self.index_path = f"index_10M_{self.no_centroids}_centroids"
        # data is a reference to the memmap object, not the data in RAM
        data = self.get_all_rows()

        kmeans = MiniBatchKMeans(
            n_clusters=self.no_centroids,
            init="k-means++",   # supported and default
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