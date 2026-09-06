import type {
  DocumentSummary,
  EvalRow,
  QueryResponse,
  RetrievalMode,
  ChunkingStrategy,
  AppConfig,
  SystemHealth,
} from "@/types/api";

// Base path is proxied to the FastAPI backend in dev (see vite.config.ts).
// In prod, point this at the deployed API origin via an env var if needed.
const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "/api";

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    ...init,
    headers: {
      ...(init?.body instanceof FormData
        ? {}
        : { "Content-Type": "application/json" }),
      ...init?.headers,
    },
  });

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new ApiError(res.status, body || res.statusText);
  }

  return res.json() as Promise<T>;
}

export const api = {
  /** POST /ingest -- uploads a document for chunking + embedding. */
  ingestDocument(file: File, chunkingStrategy: ChunkingStrategy) {
    const form = new FormData();
    form.append("file", file);
    form.append("chunking_strategy", chunkingStrategy);
    return request<{ document_id: string; filename: string; chunk_count: number; status: string }>("/ingest", {
      method: "POST",
      body: form,
    });
  },

  /** GET /documents -- list all ingested documents. */
  listDocuments() {
    return request<DocumentSummary[]>("/documents");
  },

  /** DELETE /documents/{id} -- delete a document and its chunks. */
  deleteDocument(documentId: string) {
    return request<{ message: string; document_id: string }>(`/documents/${documentId}`, {
      method: "DELETE",
    });
  },

  /** GET /documents/{id}/chunks -- list all chunks for a document. */
  getDocumentChunks(documentId: string) {
    return request<import("@/types/api").DocumentChunk[]>(`/documents/${documentId}/chunks`);
  },

  /** POST /query -- query RAG pipeline with selected retrieval mode. */
  askQuestion(question: string, retrievalMode: RetrievalMode, filterDocumentId?: string, signal?: AbortSignal) {
    return request<QueryResponse>("/query", {
      method: "POST",
      signal,
      body: JSON.stringify({
        question,
        pipeline: retrievalMode,
        retrieval_mode: retrievalMode,
        filter_document_id: filterDocumentId || null,
      }),
    });
  },

  /** GET /eval/results -- fetch evaluation benchmark results. */
  getEvalResults() {
    return request<EvalRow[]>("/eval/results");
  },

  /** POST /eval/run -- run live evaluation harness and return results. */
  runEval() {
    return request<EvalRow[]>("/eval/run", {
      method: "POST",
    });
  },

  /** GET /config -- fetch system configuration. */
  getConfig() {
    return request<AppConfig>("/config");
  },

  /** PATCH /config -- update runtime settings. */
  updateConfig(config: Partial<AppConfig>) {
    return request<AppConfig>("/config", {
      method: "PATCH",
      body: JSON.stringify(config),
    });
  },

  /** GET /health -- check system and service health. */
  getHealth() {
    return request<SystemHealth>("/health");
  },
};

export { ApiError };
