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
    assert "[Chunk 1]" in ctx
    assert "ID: chunk-1" in ctx
    assert "Source: arch.pdf, Page 2" in ctx
    assert "[Chunk 2]" in ctx
    assert "ID: chunk-2" in ctx


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


def test_format_context_empty():
    ctx = format_context([])
    assert ctx == "No relevant context found."


def test_generation_with_empty_chunks():
    service = GenerationService(llm_client=MockLLMClient())
    res = service.generate_answer("How does quantum teleportation work?", [])
    assert "cannot answer" in res.answer.lower() or "no relevant context" in res.answer.lower()
    assert len(res.citations) == 0


def test_no_dummy_first_chunk_fallback():
    # If the LLM produces an answer without citations, citations list must be empty
    service = GenerationService()
    chunks = [
        SearchResult(
            chunk_id="chunk-first",
            document_id="doc-1",
            filename="doc.md",
            content="Important facts.",
            chunk_index=0,
            score=0.9,
        )
    ]
    # Call internal _extract_citations with an uncited answer
    citations = service._extract_citations("This is an answer that has no bracketed numbers whatsoever.", chunks)
    assert citations == []  # No fake assignment to chunk-first!


def test_generation_result_to_dict():
    res = SearchResult(chunk_id="c1", document_id="d1", filename="f.md", content="text", chunk_index=0, score=0.9)
    service = GenerationService(llm_client=MockLLMClient())
    gen = service.generate_answer("query", [res])
    d = gen.to_dict()
    assert d["question"] == "query"
    assert "answer" in d
    assert "generation_latency_ms" in d
    assert "citations" in d

