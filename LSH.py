import numpy as np
import os
DIMENSION=64
SEED=42

def compute_hash(vector: np.ndarray, plane_norms: np.ndarray) -> str:
    dot_products = np.dot(vector, plane_norms.T)
    hash_bits = (dot_products > 0).astype(int)[0]
    hash_str = ''.join(hash_bits.astype(str))
    return hash_str

def Build_LSH_index(index_path:str,dataset_vectors: np.ndarray, num_planes: int):
    np.random.seed(SEED)
    plane_norms = np.random.rand(num_planes, DIMENSION) - 0.5
    hash_buckets = {}   
    Projections=np.dot(dataset_vectors, plane_norms.T)
    hash_bits = (Projections > 0).astype(int)
    for i in range(len(dataset_vectors)):
        hash_str = ''.join(hash_bits[i].astype(str))
        if hash_str not in hash_buckets:
            hash_buckets[hash_str] = []
        hash_buckets[hash_str].append(i)

    if not os.path.exists(index_path):
        os.makedirs(index_path)

    for key in hash_buckets:
        optimized_int=np.array(hash_buckets[key], dtype=np.uint32)
        np.save(os.path.join(index_path, f"{key}.npy"), optimized_int)

    #make memmap for plane norms
    plane_norms_path = os.path.join(index_path, "plane_norms.dat")
    mmap_plane_norms = np.memmap(plane_norms_path, dtype=np.float32, mode='w+', shape=plane_norms.shape)
    mmap_plane_norms[:] = plane_norms[:]
    mmap_plane_norms.flush()

    print("LSH index built successfully.")


def retreive_LSH(Plane_norms: np.ndarray, query_vector: np.ndarray, index_path: str):
    
    hash_str = compute_hash(query_vector, Plane_norms)

    indices = []
    try:
        bucket_file = os.path.join(index_path, f"{hash_str}.npy")
        print(bucket_file)
        indices=np.load(bucket_file)
    except:
        print("Bucket not found")
    return indices
