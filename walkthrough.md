# Walkthrough - RAG Frontend Integration & Setup Guide

This document provides an overview of the frontend integration with the RAG Document Q&A System, detailing the architectural enhancements, API contracts, verification results, and operational instructions.

---

## 1. Architectural Summary

The system integrates a React + TypeScript + Vite + Tailwind CSS console with the FastAPI backend, supporting both single-server deployment and dedicated frontend hot-reloading development mode.

```
┌────────────────────────────────────────────────────────┐
│               Frontend Web Console                     │
│  - Query Console (Retrieval Traces & Citation Chips)   │
│  - Document Library (Drag-and-Drop & Multi-Strategy)   │
│  - Evaluation Dashboard (Precision / Recall / Faith.)  │
│  - System Settings (Cache, Embedding & LLM Toggles)    │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP / JSON
┌───────────────────────────▼────────────────────────────┐
│                    FastAPI Backend                     │
│  - Ingestion & Multi-Strategy Chunking                 │
│  - Hybrid Search (Dense Vector + Sparse BM25 + RRF)    │
│  - Cross-Encoder Re-Ranking (ms-marco-MiniLM)          │
│  - Citation-Grounded LLM Generation                    │
│  - Evaluation & Config Endpoints                       │
│  - Embedded Static SPA Asset Mount (/assets, /*)       │
└────────────────────────────────────────────────────────┘
```

---

## 2. Implemented Features & Endpoints

### Backend API Enhancements (`main.py`)

| Endpoint | Method | Description |
|---|---|---|
| `/ingest` | `POST` | Upload and chunk documents (PDF, Markdown, HTML, TXT) with `sentence`, `semantic`, or `fixed` strategy. |
| `/query` | `POST` | Execute question answering with `pipeline` / `retrieval_mode` (`vector`, `hybrid`, `hybrid_rerank`), returning citations and visual retrieval traces. |
| `/documents` | `GET` | List all indexed documents with chunk count, strategy, file type, and status. |
| `/documents/{id}` | `DELETE` | Delete a document and cascade remove all associated chunk vectors from the index. |
| `/eval/results` | `GET` | Return benchmark evaluation metrics for all pipelines (precision@5, recall@5, faithfulness, relevance, latency). |
| `/eval/run` | `POST` | Trigger a fresh evaluation harness run across ground-truth Q&A test sets. |
| `/config` | `GET`, `PATCH` | Read and update runtime system settings (query caching, embedding model, LLM provider). |
| `/health` | `GET` | Health check for API, database connectivity, cache status, and models. |
| `/*` | `GET` | Single-Page Application (SPA) static asset serving from `rag-frontend/dist`. |

### Retrieval Signal Tagging (`retrieval/`)
- Vector search candidates are tagged with `signal: "vector"`.
- BM25 keyword candidates are tagged with `signal: "bm25"`.
- Reciprocal Rank Fusion (RRF) identifies intersection chunks and tags them with `signal: "hybrid"`.
- Chunks and citations expose `rank`, `signal`, and `score` to render color-coded trace strips (Amber = keyword, Teal = semantic, Violet = combined).

---

## 3. Frontend Pages & Components (`rag-frontend/`)

1. **Query Page (`Query.tsx`)**:
   - Retrieval mode switch (Vector-only, Hybrid, Hybrid + Re-rank).
   - Document-specific filter dropdown.
   - Grounded citations with inline preview popovers (`CitationChip.tsx`).
   - Visual retrieval trace strip (`RetrievalTrace.tsx`).
   - Latency, model, and cache status indicators.
   - Starter prompts for quick exploration.

2. **Library Page (`Library.tsx`)**:
   - Drag-and-drop file upload with chunking strategy selector.
   - Live document table with file type badges, chunk counts, ingestion timestamp, and delete action.

3. **Evaluation Page (`Eval.tsx`)**:
   - Interactive bar chart comparing Precision@5, Recall@5, and Faithfulness across pipelines.
   - Latency and relevance metrics table.
   - On-demand "Run Live Evaluation" trigger.

4. **Settings Page (`Settings.tsx`)**:
   - Live backend health status monitoring.
   - Embedding model inspection (`BAAI/bge-small-en` vs `OpenAI`).
   - LLM generation provider switch (`Anthropic`, `OpenAI`, `Mock`).
   - Interactive Query Cache toggle (TTL configuration).

---

## 4. Verification & Testing

### Frontend Build
```bash
cd rag-frontend
npm run build
```
- **Result**: Production bundle generated in `rag-frontend/dist` with 0 TypeScript/lint errors.

### Backend Automated Test Suite
```bash
python -m pytest -v
```
- **Result**: **20/20 test cases passing** (100% pass rate) across:
  - Ingestion and chunking strategies (`sentence`, `semantic`, `fixed`)
  - Loaders (PDF, Markdown, HTML, TXT)
  - Embeddings & Vector Search
  - BM25 Keyword Search & Cross-Encoder Re-ranking
  - LLM Generation & Citation Extraction
  - API Routes (`/health`, `/ingest`, `/query`, `/documents`, `/eval/results`, `/config`)

---

## 5. How to Run

### Mode A: Full Stack Single Server (Production / Standalone)
FastAPI automatically serves the built React frontend at the root URL:
```bash
python -m uvicorn main:app --reload
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

### Mode B: Frontend Development (Hot Module Replacement)
```bash
# Terminal 1 - Backend API:
python -m uvicorn main:app --reload --port 8000

# Terminal 2 - Frontend Dev Server:
cd rag-frontend
npm run dev
```
Open **[http://localhost:5173](http://localhost:5173)** in your browser.
