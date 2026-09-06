import pytest
from generation.generate import (
    GenerationService,
    MockLLMClient,
    format_context,
    SearchResult,
)


def test_format_context():
    chunks = [
        SearchResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            filename="arch.pdf",
            page_number=2,
            content="This is chunk 1 content.",
            score=0.9,
        ),
        SearchResult(
            chunk_id="chunk-2",
            document_id="doc-1",
            filename="arch.pdf",
            page_number=3,
            content="This is chunk 2 content.",
            score=0.85,
        ),
    ]
    ctx = format_context(chunks)
    assert "[Chunk chunk-1]" in ctx
    assert "Source: arch.pdf, Page 2" in ctx
    assert "[Chunk chunk-2]" in ctx


def test_generation_with_citations():
    chunks = [
        SearchResult(
            chunk_id="chunk-abc",
            document_id="doc-1",
            filename="guide.md",
            page_number=1,
            content="pgvector with HNSW index provides approximate nearest neighbor search.",
            score=0.95,
        )
    ]
    service = GenerationService(llm_client=MockLLMClient())
    res = service.generate_answer("How does vector indexing work?", chunks)

    assert res.question == "How does vector indexing work?"
    assert len(res.answer) > 0
    assert len(res.citations) >= 1
    assert res.citations[0].chunk_id == "chunk-abc"
    assert res.generation_latency_ms >= 0
