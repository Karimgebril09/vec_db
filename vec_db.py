from typing import Dict, List, Annotated
import numpy as np
import os


DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 64

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index.dat", new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path


        self.n_clusters = 100  # default number of clusters
        self.n_probe = 5       # default number of clusters to probe


        self.ivf_index = None  # IVFIndex instance
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
    
    def retrieve(self, query: np.ndarray, top_k=5):
        # Automatically load the index if not loaded
        if self.ivf_index is None:
            print("IVF index not loaded. Loading from file...")
            self.ivf_index = IVFIndex(n_clusters=0)  # placeholder, will be set by load
            self.ivf_index.file_name = self.index_path
            self.ivf_index.load()  # this will set cluster_centers and inverted_index

        if not self.ivf_index.fitted:
            raise ValueError("IVF index not built or loaded correctly.")

        # Use the IVFIndex retrieve function with get_row for memory efficiency
        top_indices = self.ivf_index.retrieve(
            query_vector=query,
            n_clusters=self.n_probe,
            n_arrays=top_k,
            cosine_similarity=self._cal_score,
            get_row=self.get_one_row  
        )
        return top_indices


    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self):

        # Load all vectors once to fit KMeans
        vectors = self.get_all_rows()
        self.ivf_index = IVFIndex(n_clusters=self.n_clusters)
        self.ivf_index.fit(vectors)
        
        # Save the index to disk
        self.ivf_index.save(self.index_path)
        print("IVF index built and saved.")




# import numpy as np

# def main():
#     DB_SIZE = 1_000_000  # 1 million vectors

#     # Step 1: Initialize VecDB and generate random data
#     print("Initializing database with 1M random vectors...")
#     db = VecDB(database_file_path="saved_db.dat",
#                 index_file_path="ivf_index.pkl",
#                 new_db=False ,
#                 db_size=DB_SIZE)
    
#     # Step 2: Build IVF index
#     print("Building IVF index...")
#     db._build_index()  # will fit KMeans and save the index

# if __name__ == "__main__":
#     main()
