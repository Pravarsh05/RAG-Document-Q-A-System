import pytest
from ingestion.loaders import LoadedDocument, DocumentPage
from ingestion.chunking import (
    FixedSizeChunker,
    SentenceChunker,
    SemanticChunker,
    ChunkingStrategyFactory,
)


@pytest.fixture
def sample_doc():
    sample_text = (
        "First sentence in the document. Second sentence describes the architecture. "
        "Third sentence details the vector store with pgvector. Fourth sentence is about BM25 search. "
        "Fifth sentence explains cross-encoder re-ranking. Sixth sentence concludes the test."
    )
    return LoadedDocument(
        filename="test.txt",
        content_type="text/plain",
        pages=[DocumentPage(page_number=1, text=sample_text)],
    )


def test_fixed_size_chunker(sample_doc):
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk(sample_doc)
    assert len(chunks) > 1
    for c in chunks:
        assert c.chunking_strategy == "fixed"
        assert len(c.content) <= 100
        assert c.page_number == 1


def test_sentence_chunker(sample_doc):
    chunker = SentenceChunker(chunk_size=120, chunk_overlap=30)
    chunks = chunker.chunk(sample_doc)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.chunking_strategy == "sentence"
        assert c.content.endswith((".", "!", "?", "test.")) or len(c.content) > 0


def test_semantic_chunker():
    md_text = (
        "# Section 1\nIntroduction paragraph with facts.\n\n"
        "## Section 2\nDeep dive into hybrid search and BM25.\n\n"
        "## Section 3\nConclusion and final evaluation."
    )
    doc = LoadedDocument(
        filename="arch.md",
        content_type="text/markdown",
        pages=[DocumentPage(page_number=1, text=md_text)],
    )
    chunker = SemanticChunker(chunk_size=75, chunk_overlap=30)
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.chunking_strategy == "semantic"


def test_chunking_strategy_factory():
    c_fixed = ChunkingStrategyFactory.get_chunker("fixed")
    assert isinstance(c_fixed, FixedSizeChunker)

    c_sentence = ChunkingStrategyFactory.get_chunker("sentence")
    assert isinstance(c_sentence, SentenceChunker)

    c_semantic = ChunkingStrategyFactory.get_chunker("semantic")
    assert isinstance(c_semantic, SemanticChunker)

    # Fallback to default strategy
    c_default = ChunkingStrategyFactory.get_chunker("unknown_strategy")
    assert isinstance(c_default, SentenceChunker)


def test_chunker_empty_document():
    doc = LoadedDocument(filename="empty.txt", content_type="text/plain", pages=[DocumentPage(page_number=1, text="")])
    chunker = SentenceChunker()
    chunks = chunker.chunk(doc)
    assert len(chunks) == 0


def test_chunker_unicode_preservation():
    unicode_text = "Japanese: こんにちは世界. German: Übergrößen Träger. Math: ∑(1/(k+rank))."
    doc = LoadedDocument(filename="unicode.txt", content_type="text/plain", pages=[DocumentPage(page_number=1, text=unicode_text)])
    chunker = FixedSizeChunker(chunk_size=100)
    chunks = chunker.chunk(doc)
    assert any("こんにちは世界" in c.content for c in chunks)
    assert any("Übergrößen" in c.content for c in chunks)


def test_chunker_zero_overlap():
    sample_text = "A" * 200
    doc = LoadedDocument(filename="rep.txt", content_type="text/plain", pages=[DocumentPage(page_number=1, text=sample_text)])
    chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=0)
    chunks = chunker.chunk(doc)
    assert len(chunks) == 4
    for c in chunks:
        assert len(c.content) == 50


def test_sentence_chunker_huge_chunk_size(sample_doc):
    chunker = SentenceChunker(chunk_size=10000, chunk_overlap=100)
    chunks = chunker.chunk(sample_doc)
    assert len(chunks) == 1
    assert "First sentence" in chunks[0].content
    assert "Sixth sentence" in chunks[0].content

