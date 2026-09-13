// Types mirroring FastAPI Pydantic response schemas

export type IngestionStatus = "processing" | "ready" | "failed";
export type ChunkingStrategy = "sentence" | "semantic" | "fixed";
export type RetrievalMode = "vector" | "bm25" | "hybrid" | "hybrid_rerank";
export type RetrievalSignal = "vector" | "bm25" | "hybrid" | "hybrid_rerank";

export interface DocumentSummary {
  id: string;
  filename: string;
  fileType: "pdf" | "markdown" | "html" | string;
  chunkCount: number;
  chunkingStrategy: ChunkingStrategy | string;
  ingestedAt: string;
  status: IngestionStatus;
}

export interface SourceChunk {
  id: string;
  chunk_id?: string;
  documentId: string;
  documentName: string;
  page?: number;
  position: number;
  text: string;
  rank: number;
  signal: RetrievalSignal;
  score: number;
}

export interface DocumentChunk {
  id: string;
  document_id: string;
  filename?: string;
  content: string;
  chunk_index: number;
  page_number?: number;
  start_char?: number;
  end_char?: number;
  chunking_strategy: string;
  chunk_metadata?: Record<string, unknown>;
}

export interface QueryResponse {
  question: string;
  answer: string;
  citations: SourceChunk[];
  retrievedChunks?: SourceChunk[];
  retrieved_chunks?: SourceChunk[];
  retrievalMode: RetrievalMode;
  retrieval_mode?: RetrievalMode;
  latencyMs: number;
  latency_ms?: Record<string, number>;
  modelName?: string;
  model_name?: string;
  cached?: boolean;
  grounding_status?: "grounded" | "insufficient_evidence" | "citation_mismatch" | "refusal";
  groundingStatus?: string;
  rewritten_query?: string;
  claim_verifications?: Array<{
    claim: string;
    cited_chunk_id?: string;
    is_supported: boolean;
    confidence: number;
    reason: string;
  }>;
  multi_doc_provenance?: Record<string, number>;
}

export interface EvalRow {
  pipeline: string;
  precisionAt5: number;
  recallAt5: number;
  faithfulness: number;
  relevance?: number;
  avgLatencyMs?: number;
  recallAt1?: number;
  recallAt3?: number;
  mrr?: number;
  ndcgAt5?: number;
  citationCorrectness?: number;
  unsupportedClaimRate?: number;
  refusalAccuracy?: number;
}


export interface AppConfig {
  embedding_provider: string;
  embedding_model: string;
  embedding_dimension: number;
  llm_provider: string;
  llm_model: string;
  cache_enabled: boolean;
  cache_ttl_seconds: number;
  default_chunking_strategy: string;
  default_chunk_size: number;
  default_chunk_overlap: number;
  vector_store: string;
}

export interface SystemHealth {
  status: string;
  database: string;
  cache: string;
  embedding_provider: string;
  embedding_model: string;
  llm_provider: string;
  llm_model: string;
}
