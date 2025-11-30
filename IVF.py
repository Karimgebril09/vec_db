import numpy as np
import os
import shutil
import heapq
import tqdm
from sklearn.cluster import MiniBatchKMeans, KMeans

ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 64
WINDOW=6000
BATCH_SIZE=4000

class IVFFlat:
    def __init__(self, db_path, index_path, n_centroids, n_probe):
        self.db_path = db_path
        self.index_path = index_path
        self.n_centroids = n_centroids
        self.n_probe = n_probe
        self.db_size = os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def build(self,data):
       
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_centroids,
            init="k-means++",
            batch_size=10_000,   
            n_init=5,         
            random_state=50,
        ) 
        kmeans.fit(data)

        labels = kmeans.labels_
        centers = kmeans.cluster_centers_.astype(np.float32)

        del data
        del kmeans

        if os.path.exists(self.index_path):
            shutil.rmtree(self.index_path)
        os.makedirs(self.index_path, exist_ok=True)

        cluster_infos = []
        for cid in range(self.n_centroids):
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

        header_matrix = np.array(header, dtype=np.uint32)
        header_matrix.tofile(os.path.join(self.index_path, "index_header.bin"))

        norms = np.linalg.norm(centers, axis=1, keepdims=True)
        centers = centers / (norms + 1e-12)
        centers.astype(np.float32).tofile(os.path.join(self.index_path, "centroids.dat"))
        print("IVF-Flat index built successfully.")

   
    def retrieve(self, query, top_k=5):
        query = np.asarray(query, dtype=np.float32).squeeze()
        qn = np.linalg.norm(query)
        normalized_query = query / (qn if qn != 0 else 1)

        centers_path = os.path.join(self.index_path, "centroids.dat")
        if not os.path.exists(centers_path):
            return []

        item_size = DIMENSION * 4
        batch_size = 4000
        centroid_scores = np.zeros(self.n_centroids, dtype=np.float32)

        for start in range(0, self.n_centroids, batch_size):
            end = min(start + batch_size, self.n_centroids)
            n_items = end - start
            offset_bytes = start * item_size

            batch = np.memmap(
                centers_path, dtype=np.float32, mode='r',
                shape=(n_items, DIMENSION), offset=offset_bytes
            )
            sims = batch.dot(normalized_query)
            centroid_scores[start:end] = sims

            del sims, batch

        if len(centroid_scores)<1000:
            nearest_centroids = np.argsort(-centroid_scores)[:self.n_probe]
        else:
            nearest_centroids = np.argpartition(-centroid_scores, self.n_probe - 1)[:self.n_probe]
        del centroid_scores

        header_arr = np.fromfile(
            os.path.join(self.index_path, "index_header.bin"),
            dtype=np.uint32
        ).reshape(-1, 2)

        all_ids = []
        flat_index_path = os.path.join(self.index_path, "all_indices.bin")
        for c in nearest_centroids:
            offset, length = header_arr[c]
            if length == 0:
                continue
            mm = np.memmap(
                flat_index_path, dtype=np.uint32, mode="r",
                offset=offset, shape=(length,)
            )
            all_ids.extend(mm[:])
            del mm

        all_ids.sort()
        groups = self.group_ids_by_window_fast(all_ids, WINDOW)
        del all_ids

        row_size = DIMENSION * 4
        top_heap = []

        with open(self.db_path, "rb") as f:
            for g in groups:
                start_id = g[0]
                end_id = g[-1]
                offset=np.int64(start_id) * np.int64(row_size)
                f.seek(offset)

                block_len = end_id - start_id + 1
                block = np.frombuffer(f.read(block_len * row_size), dtype=np.float32)
                block = block.reshape(block_len, DIMENSION)

                idxs = g - start_id
                vecs = block[idxs, :]

                scores = vecs.dot(normalized_query)

                for score, idx in zip(scores, g):
                    if len(top_heap) < top_k:
                        heapq.heappush(top_heap, (score, idx))
                        
                    else:
                        heapq.heappushpop(top_heap, (score, idx))

                del vecs, block, scores
        return [idx for score, idx in heapq.nlargest(top_k, top_heap)]

    def group_ids_by_window_fast(self, all_ids, window):
        """
        Group sorted unique IDs into consecutive blocks of max size `window`.
        """
        all_ids = np.asarray(all_ids)
        if len(all_ids) == 0:
            return []

        groups = []
        start = 0
        for i in range(1, len(all_ids)):
            if all_ids[i] - all_ids[start] >= window:
                groups.append(all_ids[start:i])
                start = i
        groups.append(all_ids[start:])
        return groups

    def get_all_rows(self):
        size = os.path.getsize(self.db_path) // (DIMENSION * 4)
        mmap_data = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(size, DIMENSION))
        return np.array(mmap_data)
