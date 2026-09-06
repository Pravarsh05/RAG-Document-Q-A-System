# RAG System Architecture & Engineering Specifications

## 1. Overview
The Document Q&A Retrieval-Augmented Generation (RAG) system is engineered to deliver high-precision answers with strict citations over enterprise documents.

## 2. Ingestion & Chunking Strategies
Empirical evaluation shows that chunking strategy fundamentally dictates downstream retrieval recall. The pipeline implements:
- **Fixed-Size Chunking**: Slices text into uniform character windows with fixed overlap (e.g. 500 characters, 100 overlap).
- **Sentence-Based Chunking**: Respects natural grammatical sentence boundaries to preserve coherent ideas without cutting words in half.
- **Hierarchical Semantic Chunking**: Respects markdown header hierarchies, paragraph blocks, and semantic topic shifts.

## 3. Vector Store & Embeddings
The dense vector index resides in **pgvector** on Postgres with **HNSW indexing** for scalable approximate nearest neighbor queries. Dense vectors are generated using open-source `BAAI/bge-small-en-v1.5` embeddings (384 dimensions) or OpenAI embeddings.

## 4. Hybrid Retrieval & Reciprocal Rank Fusion
Pure vector search frequently misses exact keyword matches, technical codes, and names. The system integrates **BM25 keyword search** alongside dense vector search. Candidate lists from both dense and sparse retrievers are merged using **Reciprocal Rank Fusion (RRF)** with standard formula:
$$RRF(d) = \sum \frac{1}{k + rank(d)}$$

## 5. Cross-Encoder Re-ranking
Top candidates from the hybrid stage pass through a cross-encoder model: **cross-encoder/ms-marco-MiniLM-L-6-v2**. The cross-encoder jointly computes self-attention across the query and candidate chunk, reordering candidates to maximize precision@k.

## 6. Cited LLM Generation
Generation uses the Claude API (or OpenAI models) with strict grounding instructions. Every factual statement must cite its supporting source using `[Chunk ID]` brackets. Uncited or hallucinated claims are penalized.
