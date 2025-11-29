import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
import pickle
import logging
from sklearn.metrics.pairwise import euclidean_distances
import os
import gzip
import varint
from compression_utils import decode_single_list, encode_and_save_indices_with_offsets
logging.basicConfig(level=logging.INFO)
import gc
class IVF_PQ_Index:
    def __init__(self, n_subvectors,n_bits, db_size,n_clusters,random_state=42,dimension=64,folder_path='index'):
        self.db_size = db_size
        # PQ parameters
        self.n_bits = n_bits    
        self.m=n_subvectors
        self.k = 2**n_bits
        self.sub_vec_dim = dimension // n_subvectors
        # IVF parameters
        self.dimension = dimension
        self.n_clusters=np.uint16(n_clusters)
        self.cluster_centers = None

        # additional parameters
        self.random_state=42
        self.inverted_index = {}
        self.subvector_estimators_centers=np.empty((self.n_clusters,self.m,self.k,self.sub_vec_dim ))
        if dimension % n_subvectors != 0:
            raise ValueError("dimension needs to be a multiple of n_subvectors")

        self.folder_path = folder_path
        self.is_trained = False

        
    def _predict_kmeans(self,query, centers):
        distances = np.linalg.norm(query[:, np.newaxis] - centers, axis=2)
        predictions = np.argmin(distances, axis=1)
        return predictions


    def _encode(self, vectors,estimators):
        #! result array to store the codewords
        result = np.empty((vectors.shape[0], self.m), dtype=np.uint32)
        for i in range(self.m):
            #! predict the assigned cluster for each group of subvectors
            estimator =estimators[i]
            #! to slice arrays
            data_slicer=self.sub_vec_dim 
            query = vectors[:, i * data_slicer : (i + 1) * data_slicer]
            result[:, i] = self._predict_kmeans(query, estimator)

        return result
    def _add(self, vectors,estimators):
        codewords = self._encode(vectors,estimators)
        codewords = codewords.astype(np.uint8)
        return codewords

    def _subvector_distance(self, queries,codewords,cluster_index):
        if not self.is_trained:
            raise ValueError("Index is not trained yet")
        #! table to store the distances between the query subvectors and the codewords subvector
        distances_table = np.zeros(( queries.shape[0],self.m,self.k), dtype=np.float32)
        #! calculate the distance between the queries sub vectors and the clusters centers for each subvector
        for i in range(self.m):
            #! to slice arrays
            data_slicer=self.sub_vec_dim 
            query = queries[:, i * data_slicer : (i + 1) * data_slicer]
            mmap=np.memmap(os.path.join(self.folder_path,'sub_cluster_centers.dat'), dtype=np.float32, mode='r', shape=(self.n_clusters,self.m,self.k,self.sub_vec_dim ))
            # index=(cluster_index*self.k*self.dimension*np.dtype(np.float32).itemsize)
            subvector_estimators_centers=mmap[cluster_index]
            mmap.flush()
            centers = subvector_estimators_centers[i]
            distances_table[:, i, :] = euclidean_distances(query, centers, squared=True)
        #! calculate the distance between the query vectors and the codewords
        distances = np.zeros((queries.shape[0], len(codewords)), dtype=np.float32)
        for i in range(self.m):
            distances += distances_table[:, i, codewords[:, i]]
        return distances

    def _searchPQ(self, query_vectors, codewords,cluster_index,n_neighbors=1):
        if not self.is_trained:
            raise ValueError("Index is not trained yet")
        #! calculate the distance between the query vector and the codewords
        distances = self._subvector_distance(query_vectors, codewords,cluster_index)
        #! get the nearest vectors indices
        nearest_vector_indices = np.argsort(distances, axis=1)[:, :n_neighbors]
        return nearest_vector_indices
    def retreive(self,query_vector,cosine_similarity,get_row,n_clusters=3,n_neighbors=10):
        gc.collect()
        #! calculate the similarities between the query vector and the cluster centers
        mmap=np.memmap(os.path.join(self.folder_path,'cluster_centers.dat'), dtype=np.float32, mode='r', shape=(self.n_clusters, self.dimension))
        self.cluster_centers=mmap[:].reshape(self.n_clusters,self.dimension)
        mmap.flush()
        similarities = np.array([cosine_similarity(query_vector, center) for center in self.cluster_centers]).squeeze()
        del self.cluster_centers
        #! get the n nearest clusters
        nearest_clusters = np.argpartition(similarities, -n_clusters)[-n_clusters:]
        #! get nearest n vectors
        similarities = np.empty((0,))
        vectors = np.empty((0, ))
        self.start_offsets=np.memmap(os.path.join(self.folder_path,'start_offsets_codewords.dat'), dtype=np.uint32, mode='r', shape=(self.n_clusters,))
        self.start_offsets=self.start_offsets[:]
        for cluster in nearest_clusters:
            start=self.start_offsets[cluster]
            end=self.start_offsets[cluster+1] if cluster+1<self.n_clusters else self.db_size
            indices_start_offsets=np.memmap(os.path.join(self.folder_path,'start_offsets_indices.bin'), dtype=np.uint32, mode='r',offset=cluster*4, shape=(self.n_clusters,))
            # indices_memmap=np.memmap(os.path.join(self.folder_path,'all_indices.dat'), dtype=np.uint32, mode='r', offset=start*4, shape=(end-start,))
            # indices=indices_memmap[:]
            indices=decode_single_list(os.path.join(self.folder_path,'all_indices.bin'), indices_start_offsets[0])
            code_path = os.path.join(self.folder_path, "codewords.dat")
            codebook_memmap = np.memmap(code_path, dtype=np.uint8, mode='r', offset=start * (self.m), shape=(end - start, self.m ))      
            
            codewords=codebook_memmap[:]
            del codebook_memmap
            n_neighbors_per_sub_cluster = (
                n_neighbors*3 if self.db_size==1000000
                else n_neighbors*25 if self.db_size==10000000 
                else n_neighbors*30 if self.db_size==15000000
                else int(n_neighbors*20)
            )
            nearest_vector_indices=self._searchPQ(query_vectors=query_vector.reshape(1,-1),codewords=codewords,n_neighbors=n_neighbors_per_sub_cluster,cluster_index=cluster).flatten()
            
            new_similarities = np.array([cosine_similarity(query_vector, get_row(i)) for i in indices[nearest_vector_indices]]).squeeze()
            vectors = np.append(vectors,indices[nearest_vector_indices])
            similarities = np.append(similarities, new_similarities) 
        nearest_arrays = np.argpartition(similarities, -n_neighbors)[-n_neighbors:]
        return vectors[nearest_arrays]
    
    def load_index(self):
        folder_path=self.folder_path
        self.n_clusters=os.path.getsize(os.path.join(folder_path,'cluster_centers.dat')) // (self.dimension * np.dtype(np.float32).itemsize)
        self.n_bits=int(np.log2(self.k))
        self.k=2**self.n_bits
        self.is_trained = True

    

    def fit(self, vectors):
        if self.is_trained:
            raise ValueError("IVF is already trained.")
        #! fit the data to the kmeans
        kmeans=MiniBatchKMeans(n_clusters=self.n_clusters,random_state=self.random_state,verbose=1,batch_size=10000)
        labels=kmeans.fit_predict(vectors)
        self.cluster_centers=kmeans.cluster_centers_
        #! separate labels for subclusters training
        label_indices = {label: np.where(labels == label)[0] for label in np.unique(labels)}
        subvector_estimators = [
            [
                MiniBatchKMeans(
                    n_clusters=self.k,
                    init='random',
                    max_iter=50,
                    random_state=42,
                    batch_size=1000
                ) 
                for _ in range(self.m)
            ]
            for _ in range(self.n_clusters)
        ]
        logging.info(f"Created subvector estimators {subvector_estimators[0]!r}")
        #! train the subclusters and assign the codewords and create the inverted index
        clusters_offset=0
        self.start_offsets=np.zeros((self.n_clusters,),dtype=np.uint32)
        for i in range(self.n_clusters):
            print(f"Training cluster {i}")
            indices = label_indices[i]
            sub_vecstors = vectors[indices]
            self.start_offsets[i]=clusters_offset
            for j in range(self.m):
                #! to slice arrays
                sub_vec_size=self.sub_vec_dim  
                estimator = subvector_estimators[i][j]
                subvectors = sub_vecstors[:, j * sub_vec_size : (j + 1) * sub_vec_size]
                estimator.fit(subvectors)
            clusters_offset += len(indices)
                
            self.subvector_estimators_centers[i]=np.array([estimator.cluster_centers_ for estimator in subvector_estimators[i]])
            codewords = self._add(vectors[indices],self.subvector_estimators_centers[i])
            zipped_lists = list(zip(indices, codewords))
            indices, codewords = zip(*zipped_lists)
            self.inverted_index[i] = (codewords, indices)
        self.is_trained = True




    def save_index(self):
        folder_path=self.folder_path
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        mmap=np.memmap(os.path.join(folder_path,'cluster_centers.dat'), dtype=np.float32, mode='w+', shape=(self.n_clusters, self.dimension))
        mmap[:]=self.cluster_centers
        mmap.flush()
        mmap=np.memmap(os.path.join(folder_path,'sub_cluster_centers.dat'), dtype=np.float32, mode='w+', shape=(self.n_clusters,self.m,self.k,self.sub_vec_dim ))
        mmap[:]=self.subvector_estimators_centers
        mmap.flush()      
        mmap=np.memmap(os.path.join(folder_path,'start_offsets_codewords.dat'), dtype=np.uint32, mode='w+', shape=(self.n_clusters,))
        mmap[:]=self.start_offsets
        mmap.flush()
        all_codes_words=[]
        all_indices=[]
        self.list_indices=[indices for _,indices in self.inverted_index.values()]
        offsets=encode_and_save_indices_with_offsets(os.path.join(folder_path,'all_indices.bin'), self.list_indices)
        mmap=np.memmap(os.path.join(folder_path,'start_offsets_indices.bin'), dtype=np.uint32, mode='w+', shape=(self.n_clusters,))
        mmap[:]=np.array(offsets, dtype=np.uint32)
        mmap.flush()
        for label,codewords in self.inverted_index.items():
            all_codes_words.extend(codewords[0])
        all_codes_words = np.array(all_codes_words, dtype=np.uint8)
        mmap=np.memmap(os.path.join(folder_path,'codewords.dat'), dtype=np.uint8, mode='w+', shape=(self.db_size, self.m))
        mmap[:]=all_codes_words
        mmap.flush()

        