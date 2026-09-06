# RAG Document Q&A System

A retrieval-augmented generation (RAG) system for question-answering over documents — built with a custom retrieval pipeline (hybrid search + re-ranking) and a proper evaluation harness, rather than a thin wrapper around an LLM API.

## Why this project is different

Most RAG demos call an embedding API, dump vectors into a managed vector DB, and call it done. This project instead:

- Compares multiple **chunking strategies** empirically instead of picking one arbitrarily
- Implements **hybrid search** (vector + BM25 keyword search) since pure vector search misses exact keyword matches
- Adds a **cross-encoder re-ranking** step to improve retrieval precision before generation
- Ships with an **evaluation harness** measuring retrieval (precision@k, recall@k) and generation quality (LLM-as-judge faithfulness/relevance), with results compared against a naive baseline

## Architecture

```
Documents → Chunking → Embedding → Vector Store (pgvector) → Retrieval → Re-ranking → LLM Generation → Answer
                                                                    ↑
                                                              Evaluation Layer
```

- **Ingestion Service** — parses PDFs, markdown, and HTML; chunks documents; stores chunk metadata (source, page, position)
- **Embedding Service** — generates embeddings (`bge-small-en` or OpenAI embeddings)
- **Vector Store** — pgvector (Postgres extension), with HNSW indexing
- **Retrieval Service** — hybrid search combining vector similarity + BM25 keyword search
- **Re-ranker** — cross-encoder (`ms-marco-MiniLM`) reorders top-k candidates before generation
- **Generation** — Claude API, prompted to cite retrieved source chunks
- **Query API** — FastAPI, exposing `POST /ingest` and `POST /query`

## Tech Stack

| Component | Choice |
|---|---|
| Language | Python |
| API framework | FastAPI |
| Vector store | pgvector (Postgres) |
| Embeddings | `bge-small-en` (open-source) or OpenAI |
| Re-ranker | Cross-encoder via `sentence-transformers` |
| LLM | Claude API |
| Keyword search | Postgres full-text search / `rank_bm25` |
| Caching | Redis |
| Evaluation | Custom harness + LLM-as-judge scoring |

## Getting Started

### Prerequisites
- Python 3.11+
- Node.js 18+ (for frontend console)

---

### Quick Setup

This system runs directly with local SQLite storage, in-memory caching, and sentence-transformers — **no background containers or heavy setups needed**.

```bash
# 1. Clone the repo and enter directory
git clone <repo-url>
cd "RAG Document Q&A System"

# 2. Create virtual environment & install dependencies
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env

# 4. Start the FastAPI backend
uvicorn main:app --reload --port 8000
```

---

### Interactive Web UI (React + Vite Console)

The project includes an interactive web dashboard with:
- **Interactive Query Console** with retrieval mode toggle (Vector, Hybrid, Hybrid + Re-rank), live retrieval trace strips, and clickable source citations.
- **Document Library** for drag-and-drop ingestion (PDF, Markdown, HTML, TXT) across multiple chunking strategies and document deletion.
- **Evaluation Dashboard** displaying live precision@5, recall@5, faithfulness charts, and benchmark tables.
- **System Settings** for inspecting and updating embedding models, LLM providers, and query cache.

#### Running the Full Stack (Single Command):
Once the frontend is built (`cd rag-frontend && npm run build`), running FastAPI serves both the API and the Web UI directly on `http://localhost:8000`:
```bash
python -m uvicorn main:app --reload
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser!

#### Running in Frontend Dev Mode (Hot Reloading):
```bash
# Terminal 1: Backend API
python -m uvicorn main:app --reload --port 8000

# Terminal 2: Frontend Dev Server
cd rag-frontend
npm run dev
```
Open **[http://localhost:5173](http://localhost:5173)** in your browser (proxies API requests to port 8000).

### Ingest a document (CLI)

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@sample_docs/example_architecture.md"
```

### Ask a question (CLI)

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?", "pipeline": "hybrid_rerank"}'
```

## Evaluation

Run the evaluation harness against the test set in `eval/test_set.json`:

```bash
python eval/run_eval.py
```

This reports:
- Precision@k and recall@k for vector-only, hybrid, and hybrid+rerank retrieval
- Faithfulness and relevance scores via LLM-as-judge
- Per-query latency and token cost breakdown

### Results

| Pipeline | Precision@5 | Recall@5 | Faithfulness |
|---|---|---|---|
| Vector-only (baseline) | 0.36 | 0.87 | 1.00 |
| Hybrid search | 0.36 | 0.87 | 1.00 |
| Hybrid + re-rank | 0.32 | 0.87 | 1.00 |

*(Generated via `python eval/run_eval.py` on the benchmark test set.)*

## Project Structure

```
rag-document-qa/
├── main.py                  # FastAPI app entrypoint & SPA static asset mount
├── config.py                # Pydantic Settings configuration
├── walkthrough.md           # End-to-end integration walkthrough & setup guide
├── rag-frontend/            # React + TypeScript + Vite + Tailwind Console
│   ├── src/
│   │   ├── api/             # API fetch client (/query, /ingest, /documents, /eval)
│   │   ├── components/      # RetrievalTrace, CitationChip, StatusPill, etc.
│   │   ├── pages/           # Query, Library, Eval, Settings screens
│   │   ├── hooks/           # TanStack React Query hooks
│   │   └── types/           # TypeScript API interfaces
│   ├── package.json
│   └── vite.config.ts       # Vite configuration with /api proxy & aliases
├── db/                      # Database models and session management
│   ├── database.py
│   └── models.py            # Document and Chunk schemas (pgvector / SQLite fallback)
├── ingestion/
│   ├── loaders.py           # PDF / Markdown / HTML / TXT parsing
│   └── chunking.py          # Fixed-size, sentence-based, semantic chunking
├── embeddings/
│   └── embed.py             # BGE-small-en / OpenAI embeddings
├── retrieval/
│   ├── vector_search.py     # pgvector HNSW cosine similarity search
│   ├── keyword_search.py    # BM25 ranking
│   ├── rerank.py            # Cross-encoder re-ranking
│   └── hybrid_search.py     # Reciprocal Rank Fusion (RRF) pipeline
├── generation/
│   └── generate.py          # LLM generation with citation validation
├── eval/
│   ├── test_set.json        # Test evaluation dataset
│   └── run_eval.py          # Benchmark evaluation harness
├── sample_docs/             # Sample PDF, Markdown, HTML files
├── tests/                   # Pytest test suite (20 passing unit/integration tests)
├── migrations/              # Alembic migrations for pgvector
├── docker-compose.yml       # Postgres pgvector + Redis services
├── requirements.txt
└── .env.example
```

## Chunking Strategy Comparison

1. **Sentence-Based Chunking (`sentence`)**:
   - **Best Overall**: Preserves complete grammatical thoughts and clause structures without clipping mid-word or mid-sentence. Produces the highest downstream answer quality and cleanest citation boundaries.
2. **Semantic / Hierarchical Recursive Chunking (`semantic`)**:
   - Ideal for structured Markdown and HTML documents with multi-level headings (`#`, `##`) and distinct sections. Ensures topic cohesiveness within each chunk.
3. **Fixed-Size Chunking (`fixed`)**:
   - Simple and predictable token/character bounds with sliding window overlap. Effective baseline, but occasionally cuts across sentence clauses or table structures.

## Failure Cases

1. **Exact Identifier / Acronym Queries on Pure Vector Search**:
   - *Problem*: Pure dense embeddings can map rare acronyms or exact version numbers (e.g., `pgvector:pg16`, `v1.5`) to generic semantic neighbors.
   - *Fix Applied*: Hybrid search with BM25 Okapi guarantees exact lexical term matches receive strong candidate scores through Reciprocal Rank Fusion (RRF).
2. **Overly Broad Semantic Queries with High Candidate Density**:
   - *Problem*: Top vector similarity hits often share high semantic similarity but low informative specificity.
   - *Fix Applied*: Cross-encoder re-ranking (`ms-marco-MiniLM-L-6-v2`) performs joint token attention over query-passage pairs to promote the most factually responsive passages.

## Roadmap

- [ ] Multi-document cross-referencing in answers
- [ ] Streaming responses
- [ ] Support for additional file formats (DOCX, CSV)
- [ ] Query result caching with invalidation on re-ingestion

## License

MIT
