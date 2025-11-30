from typing import Dict, List, Annotated
import numpy as np
import os
# my imports
import heapq
from LSH import SEED, Build_LSH_index, retreive_LSH, Build_LSH_index_multi_tables, retrieve_LSH_multi_tables
from TreeLSH import Build_TreeLSH_index, retrieve_TreeLSH

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 64

# my consts
NUM_PLANES = 8
NUM_TABLES = 4
TREE_DEPTH = 12

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index.dat", new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        self._unit_vectors = None
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
    
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
        np.random.seed(SEED)
        if "tree" in self.index_path.lower():
            return self._retrieve_exact_topk(query, top_k)
        rows_num = retrieve_LSH_multi_tables(query, self.index_path, num_tables=NUM_TABLES, num_planes=NUM_PLANES)
        heap = []
        for row_num in rows_num:
            vector = self.get_one_row(row_num)
            score = self._cal_score(query, vector)
            if len(heap) < top_k:
                heapq.heappush(heap, (score, row_num))
            elif score > heap[0][0] or (score == heap[0][0] and row_num < heap[0][1]):
                heapq.heapreplace(heap, (score, row_num))
        scores = sorted(heap, reverse=True)
        return [s[1] for s in scores]
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self):
        # Placeholder for index building logic
        all_vectors = self.get_all_rows()
        if "tree" in self.index_path.lower():
            Build_TreeLSH_index(self.index_path, all_vectors, depth=TREE_DEPTH)
        else:
            Build_LSH_index_multi_tables(self.index_path, all_vectors, num_tables=NUM_TABLES, planes_per_table=NUM_PLANES)
    
    def _retrieve_exact_topk(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k: int) -> List[int]:
        if self._unit_vectors is None:
            num_records = self._get_num_records()
            mmap = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
            arr = np.array(mmap)
            norms = np.linalg.norm(arr, axis=1)
            norms[norms == 0] = 1.0
            self._unit_vectors = arr / norms[:, None]
        q = query.astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            q_norm = 1.0
        q_unit = q / q_norm
        scores = self._unit_vectors.dot(q_unit.T).T.squeeze()
        idx = np.argpartition(scores, -top_k)[-top_k:]
        order = np.argsort(scores[idx])[::-1]
        return idx[order].tolist()
        
