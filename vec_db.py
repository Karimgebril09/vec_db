

import json
import numpy as np
import os
import shutil
import tqdm
import heapq
from sklearn.cluster import KMeans
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
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
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
    
    
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity



    def retrieve(self, query, top_k=5, n_probe=None, chunk_size=10000):
        query = np.asarray(query, dtype=np.float32).squeeze()

        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            query_norm = 1.0
        normalized_query = query / query_norm

        centers_path = os.path.join(self.index_path, "centers.npy")
        if not os.path.exists(centers_path):
            del query, normalized_query
            return []

        # Load centroids with memory-mapping to save RAM
        centroids = np.load(centers_path, mmap_mode="r")
        n_centroids = centroids.shape[0]

        # Determine number of centroids to probe
        if n_probe is None:
            n_probe = max(1, min(n_centroids, int(np.sqrt(self._get_num_records()))))
            num_records = self._get_num_records()
            if num_records <= 10 * 10**6:
                n_probe = 12
            elif num_records == 15 * 10**6:
                n_probe = 10

        # Compute similarity (cosine) to all centroids
        sims = centroids.dot(normalized_query)
        nearest_centroids = np.argpartition(sims, -n_probe)[-n_probe:]

        # Clean up memory
        del centroids, sims

        # Load header that tells us where in the flat index file each cluster's indices are
        header_path = os.path.join(self.index_path, "index_header.json")
        with open(header_path, "r") as hf:
            header = json.load(hf)

        # Turn header into a dict for faster lookup: cid -> (offset, length)
        header_dict = {item["cid"]: (item["offset"], item["length"]) for item in header}

        # Prepare top-k heap
        top_heap = []

        flat_index_path = os.path.join(self.index_path, "all_indices.bin")

        # For each centroid to probe:
        for c in nearest_centroids:
            offset, length = header_dict[c]
            if length == 0:
                continue

            # Process each chunk by remapping the memmap for that chunk
            for start in range(0, length, chunk_size):
                cur_len = min(chunk_size, length - start)
                # Remap only this chunk
                ids_mm = np.memmap(flat_index_path, dtype=np.uint32, mode="r",
                                offset=offset + start * np.dtype(np.uint32).itemsize,
                                shape=(cur_len,))

                chunk_ids = ids_mm[:]  # copy if you need to
                del ids_mm  # free memmap

                for id in chunk_ids:
                    vec = self.get_one_row(id)
                    norm = np.linalg.norm(vec)
                    score = vec.dot(normalized_query) / (norm if norm != 0 else 1.0)
                    if len(top_heap) < top_k:
                        heapq.heappush(top_heap, (score, id))
                    else:
                        heapq.heappushpop(top_heap, (score, id))

                    del vec, score
                



                del chunk_ids




        # Extract and return top-k IDs sorted by score
        results = [idx for score, idx in heapq.nlargest(top_k, top_heap)]
        del top_heap
        return results
    def _build_index(self):
       
        # sqrt(N) rule
        self.no_centroids = 250
        # data is a reference to the memmap object, not the data in RAM
        data = self.get_all_rows()

        kmeans = KMeans(
            n_clusters=self.no_centroids,
            init='k-means++',   # <-- Use k-means++ initialization
            random_state=42 
        )

        kmeans.fit(data)

        # labels and centers are new arrays created in RAM
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_.astype(np.float32)

        # Added deletion of the memmap reference and the kmeans object
        del data
        del kmeans


               
        if not os.path.isdir(self.index_path):
            os.makedirs(self.index_path, exist_ok=True)
        # 1. Build up a list of (cluster_id, indices_array)
        cluster_infos = []
        for cid in range(self.no_centroids):
            indices = np.where(labels == cid)[0].astype(np.uint32)
            cluster_infos.append((cid, indices))


        # 2. Write all indices into one big flat file
        flat_path = os.path.join(self.index_path, "all_indices.bin")
        with open(flat_path, "wb") as f:
            offset = 0
            header = []  # list of (cid, offset, length)
            for cid, inds in cluster_infos:
                length = inds.size
                f.write(inds.tobytes())  # or .tofile but with offset tracking
                header.append((cid, offset, length))
                offset += length * inds.dtype.itemsize

        # 3. Write header metadata
        header_path = os.path.join(self.index_path, "index_header.json")

        with open(header_path, "w") as hf:
            json.dump([{"cid": cid, "offset": off, "length": length} for cid, off, length in header], hf)


        norms = np.linalg.norm(centers, axis=1, keepdims=True)
        centers = centers / (norms + 1e-12)
        # 4. Save centers as before
        np.save(os.path.join(self.index_path, "centers.npy"), centers)

            




