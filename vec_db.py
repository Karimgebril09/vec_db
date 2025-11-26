import json
import numpy as np
import os
import shutil
import tqdm
import heapq
from sklearn.cluster import KMeans, MiniBatchKMeans
from typing import Annotated
import time

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
    
    def get_all_ids_rows_optimized(self, ids):
        ids = np.array(ids)
        num_records = self._get_num_records()

        sorted_idx = np.argsort(ids)
        sorted_ids = ids[sorted_idx]

        base = sorted_ids[0]
        row_size_bytes = DIMENSION * np.dtype(np.float32).itemsize
        offset = base * row_size_bytes

        # memmap starting from the base
        vectors = np.memmap(
            self.db_path, dtype=np.float32, mode='r',
            offset=offset,
            shape=(num_records - base, DIMENSION)
        )

        local_ids = sorted_ids - base
        
        result = np.empty((len(ids), DIMENSION), dtype=np.float32)
        result[sorted_idx] = vectors[local_ids]

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


    


    def retrieve(self, query, top_k=5, n_probe_level2=5, n_probe_level1=6, chunk_size=50):
        self.no_centroids = 7000
        self.no_level2_centroids = 80
        self.index_path = f"index_10M_{self.no_level2_centroids}_{self.no_centroids}_centroids"

        query = np.asarray(query, dtype=np.float32).squeeze()
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            query_norm = 1.0
        normalized_query = query / query_norm

        centroids_level2_path = os.path.join(self.index_path, "centroids_level2.npy")
        centroids_level1_path = os.path.join(self.index_path, "centroids.npy")
        if not os.path.exists(centroids_level2_path) or not os.path.exists(centroids_level1_path):
            return []

        # Load headers
        level2_header_arr = np.fromfile(
            os.path.join(self.index_path, "level2_header.bin"), dtype=np.uint32
        ).reshape(-1, 2)
        index_header_arr = np.fromfile(
            os.path.join(self.index_path, "index_header.bin"), dtype=np.uint32
        ).reshape(-1, 2)
        flat_index_path = os.path.join(self.index_path, "all_indices.bin")

        # Load level-2 centroids
        centroids_level2 = np.load(centroids_level2_path, mmap_mode="r")
        sims_level2 = centroids_level2.dot(normalized_query)
        # pick top n_probe_level2 level2 centroids
        nearest_level2 = np.argpartition(sims_level2, -n_probe_level2)[-n_probe_level2:]
        del sims_level2, centroids_level2

        centroids_level1 = np.load(centroids_level1_path, mmap_mode="r")
        top_heap = []

        # Iterate over selected top-level clusters
        for lvl2_idx in nearest_level2:
            offset_lvl2, length_lvl2 = level2_header_arr[lvl2_idx]
            if length_lvl2 == 0:
                continue

            # slice first-level centroids for this level2 cluster
            level1_start = offset_lvl2
            level1_end = offset_lvl2 + length_lvl2
            sims_level1 = centroids_level1[level1_start:level1_end].dot(normalized_query)

            # pick top n_probe_level1 first-level centroids
            nearest_first_level = np.argpartition(sims_level1, -n_probe_level1)[-n_probe_level1:]
            del sims_level1

            for idx in nearest_first_level:
                # map to global first-level index
                c = level1_start + idx
                offset, length = index_header_arr[c]
                if length == 0:
                    continue

                # process vectors in chunks
                for start in range(0, length, chunk_size):
                    cur_len = min(chunk_size, length - start)
                    ids_mm = np.memmap(
                        flat_index_path,
                        dtype=np.uint32,
                        mode="r",
                        offset=offset + start * np.dtype(np.uint32).itemsize,
                        shape=(cur_len,)
                    )
                    chunk_ids = ids_mm[:]
                    del ids_mm

                    vecs = self.get_all_ids_rows_optimized(chunk_ids)
                    scores = vecs.dot(normalized_query)

                    for score, id in zip(scores, chunk_ids):
                        if len(top_heap) < top_k:
                            heapq.heappush(top_heap, (score, id))
                        else:
                            heapq.heappushpop(top_heap, (score, id))

                    del scores, chunk_ids, vecs

        # extract top-k sorted
        results = [idx for score, idx in heapq.nlargest(top_k, top_heap)]
        del top_heap
        return results



    

    def _build_index(self):
        self.no_centroids = 7000
        self.no_level2_centroids = 80
        self.index_path = f"index_10M_{self.no_level2_centroids}_{self.no_centroids}_centroids"

        data = self.get_all_rows()

        # 1-level clustering
        kmeans = MiniBatchKMeans(
            n_clusters=self.no_centroids,
            init="k-means++",
            batch_size=10_000,
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
                        for cid in range(self.no_centroids)]

        # 2-level clustering
        kmeans2 = MiniBatchKMeans(
            n_clusters=self.no_level2_centroids,
            init="k-means++",
            batch_size=1000,
            random_state=42,
        )
        kmeans2.fit(centers)
        centers2 = kmeans2.cluster_centers_.astype(np.float32)
        labels2 = kmeans2.labels_
        cluster_level2_infos = [(cid, np.where(labels2 == cid)[0].astype(np.uint32))
                            for cid in range(self.no_level2_centroids)]

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
        np.save(os.path.join(self.index_path, "centroids.npy"), centers)

        centers2 /= (np.linalg.norm(centers2, axis=1, keepdims=True) + 1e-12)
        np.save(os.path.join(self.index_path, "centroids_level2.npy"), centers2)
