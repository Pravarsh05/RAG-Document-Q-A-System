# RAG Console — Frontend

Frontend for the RAG Document Q&A System (see project root README for the
backend). React + TypeScript + Vite + Tailwind.

## Setup

```bash
npm install
cp .env.example .env
npm run dev
```

Runs on http://localhost:5173 and proxies `/api/*` to the FastAPI backend
on http://localhost:8000 (see `vite.config.ts`).

## Structure

```
src/
├── api/client.ts          # fetch wrapper for /ingest, /query, /documents, /eval
├── types/api.ts           # types mirroring the backend's Pydantic schemas
├── hooks/useApi.ts        # React Query hooks (polling, mutations)
├── components/
│   ├── Sidebar.tsx
│   ├── RetrievalModeToggle.tsx   # vector / hybrid / hybrid+rerank switch
│   ├── RetrievalTrace.tsx        # signature element: per-chunk signal trace
│   ├── CitationChip.tsx          # inline [1][2] markers with source preview
│   ├── StatusPill.tsx
│   └── EmptyState.tsx
└── pages/
    ├── Query.tsx           # chat interface
    ├── Library.tsx         # document upload + ingestion status
    ├── Eval.tsx             # precision/recall/faithfulness charts
    └── Settings.tsx        # chunking/embedding/cache config
```

## Integration Status (Completed)

All backend endpoints and contracts are integrated and aligned:

1. **`GET /documents` & `DELETE /documents/{id}`** — Live document index listing with file type detection, chunk counting, and document deletion.
2. **`retrieval_mode` / `pipeline` param on `POST /query`** — Supports `vector`, `hybrid`, and `hybrid_rerank` retrieval mode overrides per-request.
3. **`GET /eval/results` & `POST /eval/run`** — Live benchmark charts and interactive on-demand evaluation runs.
4. **Citation & Retrieval Trace Shape** — `SearchResult` and `Citation` include `rank`, `signal` (`vector`, `bm25`, `hybrid`), `score`, `text`, and metadata for the visual retrieval trace strip.
5. **System Settings (`GET /config`, `PATCH /config`, `GET /health`)** — Live connection to backend configuration, model selection, and query caching.

## Design notes

Two accent colors are used deliberately, not decoratively: amber traces
exact/BM25 keyword hits, teal traces vector/semantic hits, and a violet
blend marks chunks both signals agreed on. This shows up in the
`RetrievalTrace` strip under every answer and in citation chip colors,
making the hybrid-search + Reciprocal Rank Fusion story from the backend
README visible in the UI rather than buried in a metrics table.
