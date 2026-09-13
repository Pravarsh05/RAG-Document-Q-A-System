import os
from typing import Optional, Literal, List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Security & CORS
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000"
    API_KEY: Optional[str] = None  # If set, requires X-API-Key or Bearer header
    RATE_LIMIT_RPM: int = 120
    MAX_UPLOAD_SIZE_MB: int = 10

    # Database (Default: local SQLite database file, zero-docker setup)
    DATABASE_URL: str = "sqlite:///./rag_app.db"
    DB_ECHO: bool = False

    # Redis Cache
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_ENABLED: bool = False
    CACHE_TTL_SECONDS: int = 3600

    # LLM
    ANTHROPIC_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    LLM_PROVIDER: Literal["anthropic", "openai", "mock"] = "mock"
    LLM_MODEL: str = "claude-3-5-sonnet-20241022"

    # Embeddings
    EMBEDDING_PROVIDER: Literal["sentence_transformers", "openai", "mock"] = "sentence_transformers"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_DEVICE: str = "cpu"

    # Re-ranker
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_DEVICE: str = "cpu"

    # Ingestion Defaults
    DEFAULT_CHUNK_SIZE: int = 500
    DEFAULT_CHUNK_OVERLAP: int = 100
    DEFAULT_CHUNKING_STRATEGY: str = "sentence"

    # Retrieval & Grounding Quality Controls
    RELEVANCE_THRESHOLD: float = 0.20  # Minimum relevance score to attempt answer synthesis
    QUERY_REWRITING_ENABLED: bool = True
    CITATION_VERIFICATION_ENABLED: bool = True
    RRF_K: int = 60
    DEFAULT_CANDIDATE_K: int = 20
    DEFAULT_TOP_K: int = 5

    @property
    def allowed_origins(self) -> List[str]:
        if not self.CORS_ORIGINS or self.CORS_ORIGINS.strip() == "*":
            if self.APP_ENV == "production":
                return ["http://localhost:5173", "http://localhost:3000"]
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]


settings = Settings()
