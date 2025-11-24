import numpy as np
import os
DIMENSION=64
SEED=3


def HammingDistance(s1: str, s2: str) -> int:
    """Compute the Hamming distance between two 6-bit strings."""
    return (int(s1, 2) ^ int(s2, 2)).bit_count()


def orthogonal_planes(num_planes, dim):
    """
    Generate num_planes orthogonal vectors in dim-dimensional space.
    QR decomposition ensures orthogonality (improves LSH distribution).
    """
    A = np.random.randn(dim, dim)
    Q, _ = np.linalg.qr(A)
    return Q.T[:num_planes, :].astype(np.float32)  # shape: (num_planes, dim



def Build_LSH_index(index_path:str,dataset_vectors: np.ndarray, num_planes: int):
    np.random.seed(SEED)
    plane_norms = orthogonal_planes(num_planes, DIMENSION) 
    hash_buckets = {}   
    Projections=np.dot(dataset_vectors, plane_norms.T)
    hash_bits = (Projections > 0).astype(int)
    
    for i in range(len(dataset_vectors)):
        hash_int = hash_bits[i].dot(1 << np.arange(num_planes)[::-1])
        if hash_int not in hash_buckets:
            hash_buckets[hash_int] = []
        hash_buckets[hash_int].append(i)

    if not os.path.exists(index_path):
        os.makedirs(index_path)

    for key in hash_buckets:
        optimized_int = np.array(hash_buckets[key], dtype=np.uint32)
        bucket_file = os.path.join(index_path, f"{key}.dat")
        mmap_bucket = np.memmap(bucket_file, dtype=np.uint32, mode='w+', shape=optimized_int.shape)
        mmap_bucket[:] = optimized_int[:]
        mmap_bucket.flush()

    np.save(os.path.join(index_path, "hash_keys.npy"), np.array(list(hash_buckets.keys())))

    #make memmap for plane norms
    plane_norms_path = os.path.join(index_path, "plane_norms.dat")
    mmap_plane_norms = np.memmap(plane_norms_path, dtype=np.float32, mode='w+', shape=plane_norms.shape)
    mmap_plane_norms[:] = plane_norms[:]
    mmap_plane_norms.flush()

    print("LSH index built successfully.")


def retreive_LSH(Plane_norms: np.ndarray, query_vector: np.ndarray, index_path: str):
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)
    dot_products = np.dot(query_vector, Plane_norms.T)
    hash_bits = (dot_products > 0).astype(int)[0]
    hash_str = ''.join(hash_bits.astype(str))

    indices = []
    bucket_file = os.path.join(index_path, f"{hash_str}.dat")
    if os.path.exists(bucket_file):
        indices = np.memmap(bucket_file, dtype=np.uint32, mode='r')
        return indices
    files=np.load(os.path.join(index_path, "hash_keys.npy"))
    min_distance=float('inf')
    closest_hash=None
    for file in files:
        if file == "plane_norms.dat":
            continue
        current_hash=file.split(".npy")[0]
        distance=HammingDistance(hash_str, current_hash)
        if distance<min_distance:
            min_distance=distance
            closest_hash=current_hash
    if closest_hash is not None:
        indices=np.memmap(os.path.join(index_path, f"{closest_hash}.dat"), dtype=np.uint32, mode='r')
    return indices
