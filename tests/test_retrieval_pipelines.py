import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.database import Base
from db.models import Document, Chunk
from retrieval.vector_search import SearchResult
from retrieval.hybrid_search import HybridSearchService
from retrieval.rerank import MockReranker


@pytest.fixture
def memory_db():
    engine = create_engine("sqlite:///:memory:", echo=False, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    doc = Document(filename="test.md", content_type="text/markdown", file_hash="hash1")
    session.add(doc)
    session.flush()

    chunks = [
        Chunk(
            document_id=doc.id,
            content="PostgreSQL with pgvector provides high-performance HNSW indexes.",
            chunk_index=0,
            chunking_strategy="sentence",
            embedding=[0.1] * 384,
        ),
        Chunk(
            document_id=doc.id,
            content="Redis distributed caching invalidates on new ingestion.",
            chunk_index=1,
            chunking_strategy="sentence",
            embedding=[-0.1] * 384,
        ),
        Chunk(
            document_id=doc.id,
            content="Cross-encoder re-ranking calculates joint self-attention scores.",
            chunk_index=2,
            chunking_strategy="sentence",
            embedding=[0.05] * 384,
        ),
        Chunk(
            document_id=doc.id,
            content="Sentence chunking splits text into grammatical sentence units.",
            chunk_index=3,
            chunking_strategy="sentence",
            embedding=[-0.05] * 384,
        ),
    ]
    session.add_all(chunks)
    session.commit()
    return session


def test_rrf_fusion_logic():
    retriever = HybridSearchService(rrf_k=60)
    list1 = [
        SearchResult(chunk_id="c1", document_id="d1", filename="f1", content="t1", chunk_index=0, score=0.9, metadata={"signal": "vector"}),
        SearchResult(chunk_id="c2", document_id="d1", filename="f1", content="t2", chunk_index=1, score=0.8, metadata={"signal": "vector"}),
    ]
    list2 = [
        SearchResult(chunk_id="c2", document_id="d1", filename="f1", content="t2", chunk_index=1, score=0.95, metadata={"signal": "bm25"}),
        SearchResult(chunk_id="c3", document_id="d1", filename="f1", content="t3", chunk_index=2, score=0.7, metadata={"signal": "bm25"}),
    ]
    fused = retriever.reciprocal_rank_fusion([list1, list2], top_k=5)
    # c2 is in both lists: ranked first and signal resolved to 'hybrid'
    assert len(fused) == 3
    assert fused[0].chunk_id == "c2"
    assert fused[0].metadata["signal"] == "hybrid"



def test_rrf_empty_lists():
    retriever = HybridSearchService()
    fused = retriever.reciprocal_rank_fusion([], top_k=5)
    assert fused == []


def test_mock_reranker():
    reranker = MockReranker()
    chunks = [
        SearchResult(chunk_id="c1", document_id="d1", filename="f1", content="unrelated topic", chunk_index=0, score=0.5),
        SearchResult(chunk_id="c2", document_id="d1", filename="f1", content="deep learning neural network", chunk_index=1, score=0.5),
    ]
    reranked = reranker.rerank("neural network", chunks, top_k=2)
    assert len(reranked) == 2
    assert reranked[0].chunk_id == "c2"


def test_vector_only_search(memory_db):
    retriever = HybridSearchService()
    results = retriever.search(db=memory_db, query="pgvector HNSW", pipeline="vector_only", top_k=2)
    assert len(results) >= 1
    assert "pgvector" in results[0].content or "Redis" in results[0].content


def test_bm25_only_search(memory_db):
    retriever = HybridSearchService()
    results = retriever.search(db=memory_db, query="pgvector", pipeline="bm25_only", top_k=2)
    assert len(results) >= 1
    assert "pgvector" in results[0].content.lower()


def test_hybrid_search(memory_db):
    retriever = HybridSearchService()
    results = retriever.search(db=memory_db, query="pgvector caching", pipeline="hybrid", top_k=2)
    assert len(results) >= 1


def test_hybrid_rerank_search(memory_db):
    retriever = HybridSearchService()
    results = retriever.search(db=memory_db, query="pgvector HNSW", pipeline="hybrid_rerank", top_k=2)
    assert len(results) >= 1


def test_filter_document_id(memory_db):
    retriever = HybridSearchService()
    doc = memory_db.query(Document).first()
    results = retriever.search(
        db=memory_db,
        query="pgvector",
        pipeline="hybrid",
        top_k=2,
        filter_document_id=doc.id,
    )
    assert len(results) >= 1
    assert all(r.document_id == doc.id for r in results)

    # Filter by non-existent doc ID returns empty
    results_empty = retriever.search(
        db=memory_db,
        query="pgvector",
        pipeline="hybrid",
        top_k=2,
        filter_document_id="non-existent-uuid",
    )
    assert len(results_empty) == 0


def test_top_k_parameter_honored(memory_db):
    retriever = HybridSearchService()
    results = retriever.search(db=memory_db, query="test", pipeline="vector_only", top_k=1)
    assert len(results) <= 1
