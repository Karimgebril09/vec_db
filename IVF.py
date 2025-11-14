import numpy as np
from sklearn.cluster import KMeans
import pickle

class IVFIndex:
    def __init__(self, n_clusters=100, file_name=None, random_state=42):
        """
        n_clusters: number of IVF clusters
        file_name: optional file to save/load the index
        """
        self.n_clusters = n_clusters
        self.cluster_centers = None
        self.inverted_index = {}  # dict: cluster_id -> indices
        self.random_state = random_state
        self.file_name = file_name
        self.fitted = False

    def fit(self, vectors):
        """Fit KMeans and build the inverted index"""
        print("Fitting centroids...")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        labels = kmeans.fit_predict(vectors)
        self.cluster_centers = kmeans.cluster_centers_

        # Build inverted index
        self.inverted_index = {}
        for idx, label in enumerate(labels):
            if label not in self.inverted_index:
                self.inverted_index[label] = []
            self.inverted_index[label].append(idx)

        # Convert lists to numpy arrays for efficiency
        for key in self.inverted_index:
            self.inverted_index[key] = np.array(self.inverted_index[key])

        self.fitted = True
        print("Index built successfully.")

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({
                'cluster_centers': self.cluster_centers,
                'inverted_index': self.inverted_index
            }, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.cluster_centers = data['cluster_centers']
            self.inverted_index = data['inverted_index']
            self.n_clusters = self.cluster_centers.shape[0]
            self.fitted = True

                

    def retrieve(self, query_vector, n_clusters, n_arrays, cosine_similarity, get_row):
        """
        Retrieve top n_arrays vectors for a query_vector
        n_clusters: number of nearest clusters to probe
        cosine_similarity: function(q, x) -> similarity
        get_row: function(i) -> vector at index i
        """
        if not self.fitted:
            raise ValueError("Index not fitted.")

        # Compute similarity to cluster centers
        similarities = np.array([cosine_similarity(query_vector, center)
                                 for center in self.cluster_centers]).squeeze()
        nearest_clusters = np.argpartition(similarities, -n_clusters)[-n_clusters:]

        # Gather candidate indices
        vectors_indices = [self.inverted_index[c] for c in nearest_clusters]
        all_vectors_indices = np.concatenate(vectors_indices)

        # Compute similarity to candidate vectors
        sims = np.array([cosine_similarity(query_vector, get_row(i))
                         for i in all_vectors_indices]).squeeze()

        # Select top n_arrays
        nearest_arrays = np.argpartition(sims, -n_arrays)[-n_arrays:]
        return all_vectors_indices[nearest_arrays]
