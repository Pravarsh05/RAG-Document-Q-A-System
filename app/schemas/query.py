from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from retrieval.hybrid_search import RetrievalPipeline


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, json_schema_extra={"example": "What is the main architecture of this system?"})
    pipeline: Optional[str] = Field(None, description="Retrieval pipeline: vector_only, bm25_only, hybrid, hybrid_rerank, optimized_hybrid_rerank")
    retrieval_mode: Optional[str] = Field(None, description="Frontend alias for pipeline: vector, bm25, hybrid, hybrid_rerank")
    mode: Optional[str] = Field(None, description="Alternative alias for retrieval mode")
    top_k: int = Field(5, ge=1, le=50, description="Number of final context chunks to feed LLM")
    candidate_k: int = Field(20, ge=1, le=100, description="Candidates retrieved before re-ranking")
    filter_document_id: Optional[str] = Field(None, description="Optional document ID filter")
    enable_query_rewriting: Optional[bool] = Field(None, description="Enable query rewriting / expansion")
    enable_citation_verification: Optional[bool] = Field(None, description="Enable claim-level citation verification")

    def get_pipeline(self) -> RetrievalPipeline:
        raw = (self.pipeline or self.retrieval_mode or self.mode or "hybrid_rerank").lower().strip()
        mapping: Dict[str, RetrievalPipeline] = {
            "vector": "vector_only",
            "vector_only": "vector_only",
            "vector-only": "vector_only",
            "bm25": "bm25_only",
            "bm25_only": "bm25_only",
            "bm25-only": "bm25_only",
            "hybrid": "hybrid",
            "hybrid_rerank": "hybrid_rerank",
            "hybrid-rerank": "hybrid_rerank",
            "hybrid+rerank": "hybrid_rerank",
            "optimized": "hybrid_rerank",
            "optimized_hybrid_rerank": "hybrid_rerank",
        }
        return mapping.get(raw, "hybrid_rerank")


class SourceChunkResponse(BaseModel):
    id: str
    chunk_id: str
    documentId: str
    document_id: str
    documentName: str
    filename: str
    page: Optional[int] = None
    page_number: Optional[int] = None
    position: int = 0
    chunk_index: int = 0
    text: str
    content: str
    rank: int = 1
    signal: str = "vector"
    score: float = 0.0
    metadata: Dict[str, Any] = {}


class CitationResponse(BaseModel):
    id: str
    chunk_id: str
    documentId: str = ""
    document_id: str = ""
    documentName: str = ""
    filename: Optional[str] = None
    page: Optional[int] = None
    page_number: Optional[int] = None
    position: int = 0
    chunk_index: int = 0
    text: str = ""
    snippet: Optional[str] = None
    rank: int = 1
    signal: str = "vector"
    score: float = 0.0
    is_verified: bool = True
    support_score: float = 1.0


class ClaimVerificationItem(BaseModel):
    claim: str
    cited_chunk_id: Optional[str] = None
    is_supported: bool
    confidence: float
    reason: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    pipeline: str
    retrievalMode: str
    retrieval_mode: str
    citations: List[CitationResponse]
    retrieved_chunks: List[SourceChunkResponse]
    retrievedChunks: List[SourceChunkResponse]
    latency_ms: Dict[str, float]
    latencyMs: float
    model_name: str
    modelName: str
    cached: bool = False
    grounding_status: Literal["grounded", "insufficient_evidence", "citation_mismatch", "refusal"] = "grounded"
    groundingStatus: str = "grounded"
    rewritten_query: Optional[str] = None
    claim_verifications: List[ClaimVerificationItem] = []
    multi_doc_provenance: Dict[str, int] = {}
