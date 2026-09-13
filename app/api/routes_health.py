import logging
from fastapi import APIRouter, Depends, status, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.core.config import settings
from db.database import get_db
from app.services.cache_service import cache_service
from embeddings.embed import get_embedding_service
from app.schemas.evaluation import ConfigResponse, ConfigUpdateRequest

logger = logging.getLogger(__name__)
router = APIRouter(tags=["System"])


@router.get("/health")
def health_check(db: Session = Depends(get_db)):
    db_status = "healthy"
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"unhealthy: {e}"

    cache_status = "in-memory (zero-docker)"
    if cache_service.redis_client and settings.CACHE_ENABLED:
        try:
            cache_service.redis_client.ping()
            cache_status = "healthy (redis)"
        except Exception as e:
            cache_status = f"unhealthy: {e}"
    elif not settings.CACHE_ENABLED:
        cache_status = "disabled"

    return {
        "status": "online",
        "database": db_status,
        "cache": cache_status,
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "embedding_model": settings.EMBEDDING_MODEL,
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": settings.LLM_MODEL,
    }


@router.get("/health/ready")
def readiness_check(db: Session = Depends(get_db)):
    """Kubernetes / Docker readiness probe."""
    checks = {}
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = "ready"
    except Exception as e:
        logger.error(f"Readiness check failed on DB: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Database not ready: {e}")

    try:
        emb = get_embedding_service()
        checks["embedding_service"] = f"ready ({emb.__class__.__name__})"
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Embedding service not ready: {e}")

    return {"status": "ready", "checks": checks}


@router.get("/health/live")
def liveness_check():
    """Kubernetes / Docker liveness probe."""
    return {"status": "alive"}


@router.get("/config", response_model=ConfigResponse)
def get_config():
    return ConfigResponse(
        embedding_provider=settings.EMBEDDING_PROVIDER,
        embedding_model=settings.EMBEDDING_MODEL,
        embedding_dimension=settings.EMBEDDING_DIMENSION,
        llm_provider=settings.LLM_PROVIDER,
        llm_model=settings.LLM_MODEL,
        cache_enabled=settings.CACHE_ENABLED,
        cache_ttl_seconds=settings.CACHE_TTL_SECONDS,
        default_chunking_strategy=settings.DEFAULT_CHUNKING_STRATEGY,
        default_chunk_size=settings.DEFAULT_CHUNK_SIZE,
        default_chunk_overlap=settings.DEFAULT_CHUNK_OVERLAP,
        vector_store="pgvector (PostgreSQL) / SQLite In-Memory fallback",
        query_rewriting_enabled=settings.QUERY_REWRITING_ENABLED,
        citation_verification_enabled=settings.CITATION_VERIFICATION_ENABLED,
        relevance_threshold=settings.RELEVANCE_THRESHOLD,
    )


@router.patch("/config", response_model=ConfigResponse)
def update_config(cfg: ConfigUpdateRequest):
    if cfg.cache_enabled is not None:
        settings.CACHE_ENABLED = cfg.cache_enabled
    if cfg.embedding_provider:
        settings.EMBEDDING_PROVIDER = cfg.embedding_provider
    if cfg.llm_provider:
        settings.LLM_PROVIDER = cfg.llm_provider
    if cfg.query_rewriting_enabled is not None:
        settings.QUERY_REWRITING_ENABLED = cfg.query_rewriting_enabled
    if cfg.citation_verification_enabled is not None:
        settings.CITATION_VERIFICATION_ENABLED = cfg.citation_verification_enabled
    if cfg.relevance_threshold is not None:
        settings.RELEVANCE_THRESHOLD = cfg.relevance_threshold
    return get_config()
