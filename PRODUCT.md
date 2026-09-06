# Product


## Platform

web

## Stack

React (Vite, TypeScript, Tailwind CSS, TanStack React Query, Recharts) + Python FastAPI backend

## Users

Developers, AI engineers, researchers, and knowledge workers who need to index multi-format documents (PDF, Markdown, HTML, TXT) and query them with grounded, cited responses.

## Product Purpose

A production-grade Document Q&A system that combines multi-strategy document chunking (sentence, semantic, fixed-size), hybrid retrieval (dense vector embeddings + sparse BM25), Cross-Encoder re-ranking, and citation-grounded answer generation with live evaluation metrics.

## Positioning

Unlike basic vector-only search systems, this platform implements hybrid search (Reciprocal Rank Fusion) and cross-encoder re-ranking to deliver higher precision@5, eliminate hallucinations with explicit chunk citations, and provide real-time retrieval traces for full explainability.

## Operating Context

- **Standalone Mode**: FastAPI serves the built single-page React frontend directly on `http://localhost:8000`.
- **Development Mode**: Vite dev server on `:5173` with FastAPI proxy to `:8000`.
- **Zero-Docker / Offline Ready**: Runs locally on SQLite + in-memory cache + local sentence-transformers/mock generation, with seamless production support for PostgreSQL (pgvector), Redis, and OpenAI / Claude LLMs.

## Capabilities and Constraints

- **Document Ingestion**: Drag-and-drop file upload with sentence-aware, semantic, and fixed chunking strategies.
- **Query & Retrieval**: Real-time multi-mode retrieval (`vector_only`, `bm25_only`, `hybrid`, `hybrid_rerank`).
- **Interactive Trace & Citations**: Visual rank-ordered retrieval trace badges (keyword, semantic, both) and clickable inline citation chips with preview popovers.
- **Evaluation Dashboard**: Comparative benchmark charts measuring precision@5, recall@5, faithfulness, relevance, and latency across pipelines.
- **Settings & Config**: Runtime toggling of query cache, LLM provider (Anthropic, OpenAI, local), and embedding configurations.

## Brand Commitments

- **Tone & Aesthetic**: Industrial, technical, high-density, terminal-inspired console with clean monospace accents and high-contrast dark theme (`ink` background, `mist` text, `lexical` amber, `vector` teal, and `hybrid` violet).
- **Name**: `rag-document-qa` (RAG Document Q&A Console).

## Product Principles

1. **Grounded & Verifiable**: Every generated statement links directly to its source chunk with exact citations.
2. **Transparent Retrieval**: The retrieval trace makes the hybrid fusion and ranking visible at a glance.
3. **Zero-Friction Local Experience**: Works out of the box with zero external API keys required, while scaling smoothly to cloud providers.
4. **Information Density with Clarity**: Clean, structured tabular and visual layouts tailored for technical analysis.
