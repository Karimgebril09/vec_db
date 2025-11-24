import numpy as np
import os
from scipy.linalg import orth
DIMENSION=64
SEED=3

def compute_hash(vector: np.ndarray, plane_norms: np.ndarray) -> str:
    dot_products = np.dot(vector, plane_norms.T)
    hash_bits = (dot_products > 0).astype(int)[0]
    hash_str = ''.join(hash_bits.astype(str))
    return hash_str

def HammingDistance(s1: str, s2: str) -> int:
    """Compute the Hamming distance between two 6-bit strings."""
    return (int(s1, 2) ^ int(s2, 2)).bit_count()

def check_orthogonal(planes: np.ndarray, tol: float = 1e-6) -> bool:
    """
    Check if all rows in `planes` are mutually orthogonal.

    planes: np.ndarray of shape (num_planes, dim)
    tol: tolerance for floating point comparison

    Returns True if all planes are orthogonal, False otherwise.
    """
    # Compute the dot product matrix between rows
    dot_matrix = np.dot(planes, planes.T)  # shape (num_planes, num_planes)

    # Identity matrix for comparison
    identity = np.eye(planes.shape[0], dtype=planes.dtype)

    # Compare: off-diagonal elements should be 0, diagonal should be 1
    return np.allclose(dot_matrix, identity, atol=tol)


def orthogonal_planes(num_planes, dim):
    # 2. Generate a random M x M matrix
    A = np.random.randn(dim, dim)

    # 3. Perform QR decomposition to get an orthonormal matrix Q
    # The columns of Q are the basis vectors for R^M.
    Q, R = np.linalg.qr(A)

    # 4. Extract the first N rows of the transpose of Q (Q.T)
    # The rows of Q.T are the original orthonormal columns of Q.
    # Taking the first N rows gives us N mutually orthogonal basis vectors.
    Orthogonal_Basis_Set = Q.T[:num_planes, :]

    assert check_orthogonal(Orthogonal_Basis_Set), "Planes are not orthogonal"
    return Orthogonal_Basis_Set


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
        optimized_int=np.array(hash_buckets[key], dtype=np.uint32)
        np.save(os.path.join(index_path, f"{key}.npy"), optimized_int)

    #make memmap for plane norms
    plane_norms_path = os.path.join(index_path, "plane_norms.dat")
    mmap_plane_norms = np.memmap(plane_norms_path, dtype=np.float32, mode='w+', shape=plane_norms.shape)
    mmap_plane_norms[:] = plane_norms[:]
    mmap_plane_norms.flush()

    print("LSH index built successfully.")


def retreive_LSH(Plane_norms: np.ndarray, query_vector: np.ndarray, index_path: str):
    dot_products = np.dot(query_vector, Plane_norms.T)
    hash_bits = (dot_products > 0).astype(int)[0]
    hash_str = ''.join(hash_bits.astype(str))

    indices = []
    try:
        bucket_file = os.path.join(index_path, f"{hash_str}.npy")
        indices=np.load(bucket_file)
    except:
        pass
        # print("Bucket not found")
    return indices
