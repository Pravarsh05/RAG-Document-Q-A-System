# PostgreSQL & pgvector HNSW Indexing Guide

## 1. Scope and Prerequisites
This guide defines index configuration, maintenance, and query execution standards for the production PostgreSQL database utilizing the `pgvector` extension.

## 2. Vector Index Architectures: HNSW vs IVFFlat
- **IVFFlat (Inverted File Flat)**:
  - Divides vectors into Voronoi lists using k-means clustering.
  - Requires pre-training on representative vector data.
  - Recall drops sharply when the distribution drifts without rebuilding the index.
- **HNSW (Hierarchical Navigable Small World)**:
  - Constructs a multi-layer graph of nearest neighbors.
  - Delivers superior Recall@K (>98%) with significantly faster query search times under concurrency.
  - Requires no clustering pre-training and allows incremental real-time insertions.

## 3. Tuned HNSW Parameters
For 384-dimensional dense vectors (`BAAI/bge-small-en-v1.5`):
- `m = 16`: The maximum number of bi-directional connection links per node in the graph. Higher values increase build time and memory usage but improve recall.
- `ef_construction = 64`: Size of the dynamic candidate list during graph construction. Ensures dense neighborhood connectivity.
- `ef_search = 40`: Size of the dynamic candidate list evaluated at query search time. Provides sub-15ms search latency with 97%+ recall.

## 4. Distance Operator Standards
- **Cosine Distance (`<=>`)**: Default operator for normalized unit vectors. Optimal for embeddings whose semantic magnitude is normalized to 1.0.
- **L2 Euclidean Distance (`<->`)**: Used for unnormalized geometric coordinate vectors.
- **Inner Product (`<#>`)**: Used when embeddings are pre-normalized and maximum dot-product search is desired.

## 5. Maintenance and Vacuuming
- Autovacuum thresholds are set with `autovacuum_vacuum_scale_factor = 0.05` and `autovacuum_analyze_scale_factor = 0.02`.
- Vector tables maintain maintenance work memory `maintenance_work_mem = '2GB'` during parallel HNSW graph creation.
