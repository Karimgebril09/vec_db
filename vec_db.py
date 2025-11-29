from typing import Dict, List, Annotated
import numpy as np
import os
from IVF_PQ import IVF_PQ_Index
DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 64

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index", new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.index=None
        if self._get_num_records() == 1000000:
             self.index = IVF_PQ_Index(n_subvectors=8,n_bits=8,n_clusters=14,folder_path=self.index_path,db_size=self._get_num_records())
        elif self._get_num_records() == 10000000:
             self.index = IVF_PQ_Index(n_subvectors=8,n_bits=8,n_clusters=150,folder_path=self.index_path,db_size=self._get_num_records())
        elif self._get_num_records() == 20000000:
             self.index = IVF_PQ_Index(n_subvectors=8,n_bits=8,n_clusters=280,folder_path=self.index_path,db_size=self._get_num_records())
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
        else:
            self.index.load_index()
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
        #TODO: might change to call insert in the index, if you need
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = int(row_num) * int(DIMENSION) * int(ELEMENT_SIZE)
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
        if self._get_num_records() == 1000000 or self._get_num_records() == 100000:
            return self.index.retreive(query,self._cal_score,self.get_one_row,n_clusters=12,n_neighbors=top_k)
        elif self._get_num_records() == 10000000:
            return self.index.retreive(query,self._cal_score,self.get_one_row,n_clusters=80,n_neighbors=top_k)
        elif self._get_num_records() == 15000000:
            return self.index.retreive(query,self._cal_score,self.get_one_row,n_clusters=90,n_neighbors=top_k)
        elif self._get_num_records() == 20000000:
            return self.index.retreive(query,self._cal_score,self.get_one_row,n_clusters=37,n_neighbors=top_k)
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self):
        # Placeholder for index building logic
        if self._get_num_records() == 1000000 or self._get_num_records() == 100000:
             self.index = IVF_PQ_Index(n_subvectors=8,n_bits=8,n_clusters=14,folder_path=self.index_path,db_size=self._get_num_records())
        elif self._get_num_records() == 10000000:
             self.index = IVF_PQ_Index(n_subvectors=8,n_bits=8,n_clusters=150,folder_path=self.index_path,db_size=self._get_num_records())
        elif self._get_num_records() == 20000000:
             self.index = IVF_PQ_Index(n_subvectors=8,n_bits=8,n_clusters=280,folder_path=self.index_path,db_size=self._get_num_records())
        self.index.fit(self.get_all_rows())
        self.index.save_index()

