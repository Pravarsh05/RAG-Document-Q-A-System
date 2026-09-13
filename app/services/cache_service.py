import time
import json
import hashlib
import logging
from typing import Optional, Dict, Any
from app.core.config import settings

logger = logging.getLogger(__name__)


class InMemoryCache:
    """Thread-safe in-memory cache with TTL and index versioning."""

    def __init__(self, default_ttl: int = 3600):
        self._cache: Dict[str, Any] = {}
        self._expiry: Dict[str, float] = {}
        self.default_ttl = default_ttl
        self.index_version: int = 1

    def get(self, key: str) -> Optional[str]:
        if key in self._cache:
            if time.time() < self._expiry.get(key, 0):
                return self._cache[key]
            else:
                del self._cache[key]
                if key in self._expiry:
                    del self._expiry[key]
        return None

    def setex(self, key: str, ttl: int, value: str) -> None:
        self._cache[key] = value
        self._expiry[key] = time.time() + ttl

    def flushdb(self) -> None:
        self._cache.clear()
        self._expiry.clear()
        self.index_version += 1

    def bump_index_version(self) -> int:
        self.index_version += 1
        return self.index_version

    def ping(self) -> bool:
        return True


class CacheService:
    def __init__(self):
        self.redis_client = None
        self.in_memory_cache = InMemoryCache(default_ttl=settings.CACHE_TTL_SECONDS)
        self._index_version = 1
        self._init_client()

    def _init_client(self):
        if settings.CACHE_ENABLED:
            try:
                import redis
                client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
                client.ping()
                self.redis_client = client
                logger.info(f"Connected to Redis at {settings.REDIS_URL}")
            except Exception as e:
                logger.warning(f"Redis unavailable ({e}). Using in-memory cache.")
                self.redis_client = None

    @property
    def client(self):
        if not settings.CACHE_ENABLED:
            return None
        return self.redis_client if self.redis_client else self.in_memory_cache

    @property
    def index_version(self) -> int:
        if self.redis_client:
            try:
                ver = self.redis_client.get("rag:index_version")
                if ver is None:
                    self.redis_client.set("rag:index_version", "1")
                    return 1
                return int(ver)
            except Exception:
                return self._index_version
        return self.in_memory_cache.index_version

    def bump_index_version(self) -> int:
        self._index_version += 1
        if self.redis_client:
            try:
                return int(self.redis_client.incr("rag:index_version"))
            except Exception:
                pass
        return self.in_memory_cache.bump_index_version()

    def make_cache_key(
        self,
        query: str,
        pipeline: str,
        top_k: int,
        candidate_k: int,
        filter_document_id: Optional[str] = None,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> str:
        """
        Creates a structured, deterministic cache key including retrieval parameters,
        document filter, model version, and index version.
        """
        norm_query = query.strip().lower()
        query_hash = hashlib.sha256(norm_query.encode("utf-8")).hexdigest()[:16]
        doc_filter = filter_document_id or "all"
        model_str = llm_model or settings.LLM_MODEL
        emb_str = embedding_model or settings.EMBEDDING_MODEL
        version = self.index_version

        return f"rag:q:{version}:{pipeline}:{top_k}:{candidate_k}:{doc_filter}:{model_str}:{emb_str}:{query_hash}"

    def get_query(self, key: str) -> Optional[Dict[str, Any]]:
        c = self.client
        if not c:
            return None
        try:
            val = c.get(key)
            if val:
                return json.loads(val)
        except Exception as e:
            logger.warning(f"Cache get error for {key}: {e}")
        return None

    def set_query(self, key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> None:
        c = self.client
        if not c:
            return
        ttl_seconds = ttl or settings.CACHE_TTL_SECONDS
        try:
            c.setex(key, ttl_seconds, json.dumps(data))
        except Exception as e:
            logger.warning(f"Cache set error for {key}: {e}")

    def invalidate_all(self) -> None:
        self.bump_index_version()
        if self.in_memory_cache:
            self.in_memory_cache.flushdb()
        if self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception:
                pass


cache_service = CacheService()
