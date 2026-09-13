from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class EvalRow(BaseModel):
    pipeline: str
    precisionAt5: float
    recallAt5: float
    faithfulness: float
    relevance: float
    avgLatencyMs: float
    # Extended metrics
    recallAt1: Optional[float] = None
    recallAt3: Optional[float] = None
    mrr: Optional[float] = None
    ndcgAt5: Optional[float] = None
    citationCorrectness: Optional[float] = None
    unsupportedClaimRate: Optional[float] = None
    refusalAccuracy: Optional[float] = None


class BenchmarkRunResponse(BaseModel):
    run_id: str
    timestamp: str
    dataset_name: str
    dataset_size: int
    has_run: bool
    results: List[EvalRow]
    summary: Dict[str, Any] = {}


class ConfigResponse(BaseModel):
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    llm_provider: str
    llm_model: str
    cache_enabled: bool
    cache_ttl_seconds: int
    default_chunking_strategy: str
    default_chunk_size: int
    default_chunk_overlap: int
    vector_store: str
    query_rewriting_enabled: bool
    citation_verification_enabled: bool
    relevance_threshold: float


class ConfigUpdateRequest(BaseModel):
    cache_enabled: Optional[bool] = None
    embedding_provider: Optional[str] = None
    llm_provider: Optional[str] = None
    query_rewriting_enabled: Optional[bool] = None
    citation_verification_enabled: Optional[bool] = None
    relevance_threshold: Optional[float] = None
