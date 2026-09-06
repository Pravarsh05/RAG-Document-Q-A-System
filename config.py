import os
from typing import Optional, Literal
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

    # Database (Default: local SQLite database file, zero-docker setup)
    DATABASE_URL: str = "sqlite:///./rag_app.db"
    DB_ECHO: bool = False

    # Redis
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
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    RERANKER_DEVICE: str = "cpu"

    # Ingestion Defaults
    DEFAULT_CHUNK_SIZE: int = 500
    DEFAULT_CHUNK_OVERLAP: int = 100
    DEFAULT_CHUNKING_STRATEGY: str = "sentence"


settings = Settings()
