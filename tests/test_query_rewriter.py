import pytest
from app.services.query_rewriter import QueryRewriter


def test_query_rewriter_removes_conversational_prefixes():
    rewriter = QueryRewriter(enabled=True)
    queries = [
        ("Can you please explain how does pgvector work?", "pgvector work"),
        ("Could you tell me about Reciprocal Rank Fusion?", "Reciprocal Rank Fusion"),
        ("I was wondering what is cross-encoder reranking?", "cross-encoder reranking"),
        ("What are the primary chunking strategies?", "the primary chunking strategies"),
        ("Tell me about Redis cache invalidation.", "Redis cache invalidation"),
    ]
    for raw, expected_core in queries:
        core = rewriter.clean_conversational_noise(raw)
        assert expected_core.lower() in core.lower()


def test_query_rewriter_extracts_technical_expansions():
    rewriter = QueryRewriter(enabled=True)
    expansions = rewriter.extract_expansions("How does pgvector HNSW indexing perform?")
    assert len(expansions) > 0
    assert any(term in ["postgres", "vector", "similarity", "approximate", "neighbors"] for term in expansions)


def test_query_rewriter_expands_query():
    rewriter = QueryRewriter(enabled=True)
    rewritten, expansions = rewriter.rewrite("What is pgvector?")
    assert len(expansions) > 0
    assert "pgvector" in rewritten
    assert any(exp in rewritten for exp in expansions)


def test_query_rewriter_handles_plain_query_without_expansions():
    rewriter = QueryRewriter(enabled=True)
    rewritten, expansions = rewriter.rewrite("Simple question with no technical terms")
    assert rewritten == "Simple question with no technical terms"
    assert expansions == []


def test_query_rewriter_disabled_mode():
    rewriter = QueryRewriter(enabled=False)
    raw = "Can you please tell me about pgvector?"
    rewritten, expansions = rewriter.rewrite(raw)
    assert rewritten == raw
    assert expansions == []


def test_query_rewriter_handles_empty_or_whitespace():
    rewriter = QueryRewriter(enabled=True)
    rewritten, expansions = rewriter.rewrite("")
    assert rewritten == ""
    assert expansions == []

    rewritten_ws, expansions_ws = rewriter.rewrite("   ")
    assert rewritten_ws == "   " or rewritten_ws == ""


def test_query_rewriter_handles_bm25_expansions():
    rewriter = QueryRewriter(enabled=True)
    rewritten, expansions = rewriter.rewrite("How does BM25 keyword matching work?")
    assert any(term in ["sparse", "lexical", "keyword", "frequency", "idf"] for term in expansions)


def test_query_rewriter_handles_rrf_expansions():
    rewriter = QueryRewriter(enabled=True)
    rewritten, expansions = rewriter.rewrite("Explain RRF fusion algorithm.")
    assert any(term in ["reciprocal", "rank", "fusion", "rankings", "score"] for term in expansions)


def test_query_rewriter_deduplicates_already_present_terms():
    rewriter = QueryRewriter(enabled=True)
    # If the user already included 'postgres' and 'vector', rewriter should not duplicate them
    expansions = rewriter.extract_expansions("pgvector postgres vector")
    assert "postgres" not in expansions
    assert "vector" not in expansions


def test_query_rewriter_preserves_special_characters():
    rewriter = QueryRewriter(enabled=True)
    rewritten, _ = rewriter.rewrite("cross-encoder/ms-marco-MiniLM-L-6-v2")
    assert "cross-encoder" in rewritten or "ms-marco" in rewritten
