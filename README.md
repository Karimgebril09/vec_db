<div >

# 🚀 Vector DB with IVF-Flat Search

## 🏆 Competition Results

| Placement | Achievement | Score |
|-----------|-------------|-------|
| 🥈 **2nd Place** | Fast & Accurate Vector Retrieval | ⭐⭐⭐⭐⭐ |

</div>

## Overview

This project implements an efficient **IVF-Flat** (Inverted File with Flat Posting Lists) index for approximate nearest neighbor search over millions of 64-dimensional vectors. It uses memory-mapped storage and careful algorithmic choices at each stage to achieve competitive latency and recall.

---

## 📋 Architecture Overview

| Stage | Purpose | Key Optimization |
|-------|---------|------------------|
| 🔨 **BUILD** | Partition vectors using clustering | K-Means++ initialization |
| 💾 **STORE** | Compress index on disk | Pre-normalized centroids |
| 🔍 **RETRIEVE** | Find top-k similar vectors | Batch centroid scoring |

---

## 🔨 1. BUILD: Clustering with K-Means++

The index build phase creates a quantized partition of the vector space using k-means clustering.

###  Strategy

| Aspect | Approach | Benefit |
|--------|----------|---------|
| **Initialization** | K-Means++ (smart centroid seeding) |  Avoids poor local minima |
| **Scaling** | MiniBatchKMeans (10K batches) |  Handles millions of vectors |
| **Quality** | `n_init=5` multiple runs |  Consistent cluster quality |
| **Speed** | Batch processing |  Bounded memory usage |

---

## 💾 2. STORE STRUCTURE: Compressed Index on Disk

Once centroids are trained, all vectors are partitioned into cluster-based inverted lists. The inverted index is stored as compact binary files.

###  Structure

| Component | Format | Purpose | Size |
|-----------|--------|---------|------|
| **Centroids** (`centroids.dat`) | L2-normalized float32 | Centroid ID | ~3 MB |
| **Inverted Lists** (`all_indices.bin`) | uint32 array | Vector ID groups | ~40 MB |
| **Index Header** (`index_header.bin`) | [offset, length] pairs | Quick lookup | ~1 KB |

###  Size Reduction Strategy

**Pre-normalize centroids at build time**  
- Formula: `centers = centers / (norms + 1e-12)`
- Saves computation at query time.
- Normalized centroids enable direct dot product = cosine similarity.

 **Store only IDs in inverted lists, not full vectors**  
- Each inverted list stores uint32 IDs (4 bytes each), not vectors.
- Full vectors stay in the database file, fetched on-demand.



## 🔍 3. RETRIEVE: Fast Centroid-Normalized Search

The retrieval pipeline uses the pre-normalized centroids to eliminate redundant computation.

### 📍 Retrieve Algorithm

| Step | Action | Optimization |
|------|--------|---------------|
|  1| Normalize query: `q_norm = q / \|\|q\|\|` | One-time cost |
|  2| Score query vs. centroids in 4K batches | Batch memmap access |
| 3 | Select top `n_probe` centroids | argsort/argpartition |
| 4 | Fetch vector IDs from inverted lists | Grouped by window |
| 5 | Stream vectors, compute similarities | Keep top-k heap |
| 6 | Return final top-k IDs | Minimal memory |
### ⚡ Key Optimization 1: Pre-Normalized Centroids

**What**: Centroids are L2-normalized **at build time**, not at query time.

| When | Task | Cost |
|------|------|------|
|  **Build** | `norms = \|\|centers\|\|` → normalize once | **One-time** |
| **Query** | `dot(q_norm, c_norm)` = cosine similarity | **Direct!** |

### ⚡ Key Optimization 2: Windowed Sequential I/O

**What**: `group_ids_by_window_fast()` groups candidate vector IDs into blocks of max size `window` (6000).

**Why it matters**:
-  **Disk reads are sequential**, not random → 10–100x faster I/O.
- **Windowing (6000 vectors = ~1.5MB)** fits in L3 cache.
- Transforms worst-case (millions of random seeks) into few sequential scans.


```python
# Input: [100, 102, 105, 6100, 6102, 12000]
# Output: [[100, 102, 105], [6100, 6102], [12000]]
#         (groups where max gap ≥ 6000)
```
- Groups IDs where gap between min and max ≥ window size.
- Each group reads one contiguous block from disk.
- Avoids cache thrashing and OS page faults.

**Impact**:
- Reduces I/O latency 
---

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Create and build an index** (e.g., 1M vectors):
   ```python
   from vec_db import VecDB
   
   db = VecDB(db_size=1_000_000, database_file_path="saved_db.dat", 
              index_file_path="index_1M", new_db=True)
   # Index builds automatically
   ```


## Index & Retrieval Parameters

| DB Size | Centroids | n_probe |
|---------|-----------|---------|
| 1M      | 800       | 10      |
| 10M     | 8000      | 4       |
| 20M     | 16000     | 4       |


## 📊 Evaluation

**First Run**
| DB Size | Score | Query Time | RAM Usage |
|---------|-------|-----------|-----------|
| 1M      | 0.0   | 0.09s     | 0.02 MB   |
| 10M     | 0.0   | 2.38s     | 0.02 MB   |
| 20M     | 0.0   | 1.26s     | 1.02 MB   |

**Second Run**
| DB Size | Score | Build Time | RAM Usage |
|---------|-------|-----------|-----------|
| 1M      | 0.0   | 0.08s     | 0.01 MB   |
| 10M     | 0.0   | 0.15s     | 0.24 MB   |
| 20M     | 0.0   | 0.26s     | 0.95 MB   |


---

## 📁 Files

| File | Purpose |
|------|---------|
| [vec_db.py](vec_db.py) | 🔧 `VecDB` wrapper & orchestration |
| [IVF.py](IVF.py) | 🏗️ IVF-Flat build & retrieval |
| [evaluation.py](evaluation.py) | 📊 Performance benchmarks |
| [compression.py](compression.py) | 📦 Index packaging utilities |
| [requirements.txt](requirements.txt) | 📋 Dependencies |

