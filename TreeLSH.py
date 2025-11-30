import numpy as np
import os

DIMENSION = 64
SEED = 3


def HammingDistance(s1: str, s2: str) -> int:
    return (int(s1, 2) ^ int(s2, 2)).bit_count()


def Build_TreeLSH_index(index_path: str, dataset_vectors: np.ndarray, depth: int):
    np.random.seed(SEED)
    planes = {}
    for level in range(depth):
        nodes = 1 << level
        for n in range(nodes):
            path = format(n, f"0{level}b") if level > 0 else ""
            v = np.random.randn(DIMENSION).astype(np.float32)
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            planes[path] = v

    if not os.path.exists(index_path):
        os.makedirs(index_path)
    planes_dir = os.path.join(index_path, "planes")
    buckets_dir = os.path.join(index_path, "buckets")
    os.makedirs(planes_dir, exist_ok=True)
    os.makedirs(buckets_dir, exist_ok=True)

    leaf_buckets = {}
    for i in range(len(dataset_vectors)):
        path = ""
        x = dataset_vectors[i]
        for level in range(depth):
            p = planes[path]
            bit = 1 if np.dot(x, p) > 0 else 0
            path = path + str(bit)
        if path not in leaf_buckets:
            leaf_buckets[path] = []
        leaf_buckets[path].append(i)

    for path, vec in planes.items():
        f = os.path.join(planes_dir, f"{path if path != '' else 'root'}.dat")
        mmap_plane = np.memmap(f, dtype=np.float32, mode="w+", shape=(DIMENSION,))
        mmap_plane[:] = vec[:]
        mmap_plane.flush()

    for leaf, indices in leaf_buckets.items():
        arr = np.array(indices, dtype=np.uint32)
        f = os.path.join(buckets_dir, f"{leaf}.dat")
        mmap_bucket = np.memmap(f, dtype=np.uint32, mode="w+", shape=arr.shape)
        mmap_bucket[:] = arr[:]
        mmap_bucket.flush()

    np.save(os.path.join(index_path, "leaf_keys.npy"), np.array(list(leaf_buckets.keys())))
    np.save(os.path.join(index_path, "meta.npy"), np.array([depth], dtype=np.uint32))


def retrieve_TreeLSH(query_vector: np.ndarray, index_path: str, depth: int, max_hamming: int = 2, candidate_target: int = 1000):
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)
    planes_dir = os.path.join(index_path, "planes")
    buckets_dir = os.path.join(index_path, "buckets")
    x = query_vector[0]

    def route_from_prefix(prefix_bits: str) -> str:
        path = prefix_bits
        for level in range(len(prefix_bits), depth):
            node_file = os.path.join(planes_dir, f"{path if path != '' else 'root'}.dat")
            plane = np.memmap(node_file, dtype=np.float32, mode="r", shape=(DIMENSION,))
            bit = 1 if np.dot(x, plane) > 0 else 0
            del plane
            path = path + str(bit)
        return path

    margins = []
    bits = []
    path = ""
    for level in range(depth):
        node_file = os.path.join(planes_dir, f"{path if path != '' else 'root'}.dat")
        plane = np.memmap(node_file, dtype=np.float32, mode="r", shape=(DIMENSION,))
        dot = float(np.dot(x, plane))
        bit = 1 if dot > 0 else 0
        margins.append(abs(dot))
        bits.append(bit)
        del plane
        path = path + str(bit)

    indices_set = set()

    def add_bucket(p: str):
        f = os.path.join(buckets_dir, f"{p}.dat")
        if os.path.exists(f):
            mm = np.memmap(f, dtype=np.uint32, mode="r")
            indices_set.update(mm.tolist())

    add_bucket(path)

    order = sorted(range(depth), key=lambda i: margins[i])
    for i in order:
        if len(indices_set) >= candidate_target:
            break
        flipped_bit = '1' if bits[i] == 0 else '0'
        prefix = ''.join(str(b) for b in bits[:i]) + flipped_bit
        leaf = route_from_prefix(prefix)
        add_bucket(leaf)

    if len(indices_set) < candidate_target:
        pair_order = []
        for a in range(min(depth, 12)):
            for b in range(a+1, min(depth, 12)):
                pair_order.append((margins[a] + margins[b], a, b))
        pair_order.sort()
        for _, a, b in pair_order[:32]:
            if len(indices_set) >= candidate_target:
                break
            pa = '1' if bits[a] == 0 else '0'
            pb = '1' if bits[b] == 0 else '0'
            prefix = ''.join(str(b) for b in bits[:a]) + pa
            prefix = prefix + ''.join(str(b) for b in bits[a+1:b]) + pb
            leaf = route_from_prefix(prefix)
            add_bucket(leaf)

    if len(indices_set) == 0:
        keys = np.load(os.path.join(index_path, "leaf_keys.npy"), allow_pickle=True)
        best = None
        best_d = None
        for k in keys:
            d = HammingDistance(path, k)
            if d <= max_hamming and (best_d is None or d < best_d):
                best_d = d
                best = k
                if d == 0:
                    break
        if best is not None:
            add_bucket(best)

    return np.array(list(indices_set), dtype=np.uint32)

