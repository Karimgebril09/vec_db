from typing import Dict, List, Annotated
import numpy as np
import os
from IVF import IVFFlat
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
    
    
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5):
        """
        Retrieve using IVF-Flat. Automatically create/load index if needed.
        """
        num_records = self._get_num_records()

        # Create/load the IVFFlat index if it doesn't exist
        
        if num_records == 1_000_000:
            self.index = IVFFlat(n_centroids=800, n_probe=10, db_path=self.db_path, index_path=self.index_path)
        elif num_records == 10_000_000:
            self.index = IVFFlat(n_centroids=8000, n_probe=4, db_path=self.db_path, index_path=self.index_path )
        elif num_records == 20_000_000:
            self.index = IVFFlat(n_centroids=16000, n_probe=5, db_path=self.db_path, index_path=self.index_path)
        else:
            raise ValueError(f"No IVFFlat configuration for DB size {num_records}")
      

        # Perform IVFFlat retrieval
        return self.index.retrieve(query, top_k=top_k)

    

    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self):
        """
        Build or rebuild the IVF-Flat index depending on DB size.
        """
        num_records = self._get_num_records()

        # Select IVF-Flat configuration based on DB size
        if num_records == 1_000_000:
            self.index = IVFFlat(n_centroids=800, n_probe=10, db_path=self.db_path, index_path=self.index_path)
        elif num_records == 10_000_000:
            self.index = IVFFlat(n_centroids=8000, n_probe=8, db_path=self.db_path, index_path=self.index_path)
        elif num_records == 20_000_000:
            self.index = IVFFlat(n_centroids=16000, n_probe=5, db_path=self.db_path, index_path=self.index_path)
        else:
            raise ValueError(f"No IVFFlat configuration for DB size {num_records}")

        # Train the index and save
        self.index.build( self.get_all_rows())
       
