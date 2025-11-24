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
        hash_str = ''.join(hash_bits[i].astype(str))
        
        if hash_str not in hash_buckets:
            hash_buckets[hash_str] = []
        hash_buckets[hash_str].append(i)

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
    dot_products = np.dot(query_vector, Plane_norms.T)
    hash_bits = (dot_products > 0).astype(int)
    if hash_bits.ndim > 1:
        hash_bits = hash_bits[0] 
    hash_str = "".join(map(str, hash_bits))

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

#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################

def Build_LSH_index_multi_tables(index_path: str, dataset_vectors: np.ndarray, num_tables: int, planes_per_table: int):
    np.random.seed(SEED)
    
    if not os.path.exists(index_path):
        os.makedirs(index_path)

    for t in range(num_tables):
        table_path = os.path.join(index_path, f"table_{t}")
        os.makedirs(table_path, exist_ok=True)

        plane_norms = orthogonal_planes(planes_per_table, DIMENSION)
        projections = np.dot(dataset_vectors, plane_norms.T)
        hash_bits = (projections > 0).astype(int)

        hash_buckets = {}
        for i in range(len(dataset_vectors)):
            hash_str = "".join(map(str, hash_bits[i]))
            if hash_str not in hash_buckets:
                hash_buckets[hash_str] = []
            hash_buckets[hash_str].append(i)

        # Save each bucket as memmap
        for key in hash_buckets:
            indices = np.array(hash_buckets[key], dtype=np.uint32)
            bucket_file = os.path.join(table_path, f"{key}.dat")
            mmap_bucket = np.memmap(bucket_file, dtype=np.uint32, mode='w+', shape=indices.shape)
            mmap_bucket[:] = indices[:]
            mmap_bucket.flush()

        # Save plane norms
        plane_file = os.path.join(table_path, "plane_norms.dat")
        mmap_planes = np.memmap(plane_file, dtype=np.float32, mode='w+', shape=plane_norms.shape)
        mmap_planes[:] = plane_norms[:]
        mmap_planes.flush()

        # Save hash keys
        np.save(os.path.join(table_path, "hash_keys.npy"), np.array(list(hash_buckets.keys())))

    print(f"LSH index built successfully with {num_tables} tables, {planes_per_table} planes per table.")


def retrieve_LSH_multi_tables(query_vector: np.ndarray, index_path: str, num_tables: int,num_planes: int):
    """
    Retrieve candidate indices from all LSH tables for the query vector.
    """
    all_indices = set()  # Use set to avoid duplicates

    for t in range(num_tables):
        table_path = os.path.join(index_path, f"table_{t}")

        # Load plane norms for this table
        plane_file = os.path.join(table_path, "plane_norms.dat")
        plane_norms = np.memmap(plane_file, dtype=np.float32, mode='r',
                                shape=(num_planes, DIMENSION))
           

        # Compute hash bits
        dot_products = np.dot(query_vector, plane_norms.T)
        hash_bits = (dot_products > 0).astype(int)[0]
        hash_str = "".join(map(str, hash_bits))

        # Try to load exact bucket
        bucket_file = os.path.join(table_path, f"{hash_str}.dat")
        if os.path.exists(bucket_file):
            indices = np.memmap(bucket_file, dtype=np.uint32, mode='r')
            all_indices.update(indices.tolist())
        else:
            # Fallback: find nearest bucket by Hamming distance
            files = np.load(os.path.join(table_path, "hash_keys.npy"))
            min_distance = float('inf')
            closest_hash = None
            for f in files:
                distance = sum(c1 != c2 for c1, c2 in zip(hash_str, f))
                if distance < min_distance:
                    min_distance = distance
                    closest_hash = f
            if closest_hash is not None:
                indices = np.memmap(os.path.join(table_path, f"{closest_hash}.dat"), dtype=np.uint32, mode='r')
                all_indices.update(indices.tolist())

    return np.array(list(all_indices), dtype=np.uint32)


