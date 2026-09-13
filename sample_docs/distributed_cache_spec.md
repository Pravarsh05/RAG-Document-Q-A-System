# Distributed Cache Specification & Cache Invalidation Protocol

## 1. System Overview
The distributed caching layer sits between the FastAPI retrieval application and downstream vector/relational databases. It provides sub-millisecond retrieval of frequent semantic queries, query embeddings, and user sessions.

## 2. Cluster Architecture & Network Topology
- **Primary Cache Engine**: Redis Enterprise / KeyDB cluster operating in high-availability Sentinel configuration.
- **Port Allocations**: Standard client communication occurs over port `6379`. Cluster bus gossip communication runs over port `16379`.
- **Replication Factor**: 3-node replication (1 primary, 2 read replicas) with automatic failover quorum within 3500ms.
- **Memory Eviction Policy**: `volatile-lru` with active memory defragmentation enabled when fragmentation ratio exceeds 1.5.

## 3. Cache Key Derivation Scheme
Cache keys are constructed deterministically to eliminate cache poisoning and cross-tenant leakage:
`rag:q:{index_version}:{pipeline}:{top_k}:{candidate_k}:{doc_filter}:{model_name}:{embedding_model}:{query_sha256}`

- **Index Version**: An atomic integer counter `rag:index_version` incremented on every document upload or deletion.
- **TTL Configuration**: Default query cache entries expire after 3600 seconds (1 hour). High-frequency metadata keys expire after 300 seconds.

## 4. Invalidation Mechanics
When a new document is ingested or an existing document is removed:
1. The ingestion pipeline executes `INCR rag:index_version`, immediately invalidating all previous query cache hits in $O(1)$ time without expensive wildcard scan keys.
2. In-memory development caches flush local dictionaries and advance their local version clock.
3. Cold queries execute against the underlying search pipeline and repopulate the cache under the new version prefix.

## 5. Rate Limiting Coordination
Client rate limiting is enforced via a distributed sliding-window counter using Redis sorted sets (ZSET):
- Window size: 60 seconds.
- Default limit: 120 requests per minute per IP / API key.
- Throttled requests receive an HTTP 429 status code with a `Retry-After` header.
