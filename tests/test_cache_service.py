import time
import pytest
from app.services.cache_service import CacheService, InMemoryCache


def test_in_memory_cache_set_and_get():
    cache = InMemoryCache(default_ttl=60)
    cache.setex("key1", 60, "value1")
    assert cache.get("key1") == "value1"
    assert cache.get("non_existent") is None


def test_in_memory_cache_ttl_expiration():
    cache = InMemoryCache(default_ttl=1)
    cache.setex("temp_key", 1, "temp_val")
    assert cache.get("temp_key") == "temp_val"
    time.sleep(1.1)
    assert cache.get("temp_key") is None


def test_in_memory_cache_flushdb():
    cache = InMemoryCache(default_ttl=60)
    cache.setex("k1", 60, "v1")
    cache.setex("k2", 60, "v2")
    v_before = cache.index_version
    cache.flushdb()
    assert cache.get("k1") is None
    assert cache.get("k2") is None
    assert cache.index_version > v_before


def test_cache_key_deterministic():
    svc = CacheService()
    k1 = svc.make_cache_key("What is pgvector?", "hybrid_rerank", top_k=5, candidate_k=20)
    k2 = svc.make_cache_key("What is pgvector?", "hybrid_rerank", top_k=5, candidate_k=20)
    assert k1 == k2


def test_cache_key_case_and_whitespace_invariant():
    svc = CacheService()
    k1 = svc.make_cache_key("What is pgvector?", "hybrid_rerank", top_k=5, candidate_k=20)
    k2 = svc.make_cache_key("  what is pgvector?  ", "hybrid_rerank", top_k=5, candidate_k=20)
    assert k1 == k2


def test_cache_key_differs_by_pipeline():
    svc = CacheService()
    k_vec = svc.make_cache_key("test query", "vector_only", top_k=5, candidate_k=20)
    k_hyb = svc.make_cache_key("test query", "hybrid", top_k=5, candidate_k=20)
    assert k_vec != k_hyb


def test_cache_key_differs_by_top_k():
    svc = CacheService()
    k_3 = svc.make_cache_key("test query", "hybrid_rerank", top_k=3, candidate_k=20)
    k_5 = svc.make_cache_key("test query", "hybrid_rerank", top_k=5, candidate_k=20)
    assert k_3 != k_5


def test_cache_key_differs_by_doc_filter():
    svc = CacheService()
    k_all = svc.make_cache_key("test query", "hybrid_rerank", top_k=5, candidate_k=20, filter_document_id=None)
    k_doc = svc.make_cache_key("test query", "hybrid_rerank", top_k=5, candidate_k=20, filter_document_id="doc-123")
    assert k_all != k_doc


def test_cache_invalidation_changes_keys():
    svc = CacheService()
    k_before = svc.make_cache_key("query", "hybrid", top_k=5, candidate_k=20)
    svc.invalidate_all()
    k_after = svc.make_cache_key("query", "hybrid", top_k=5, candidate_k=20)
    assert k_before != k_after
