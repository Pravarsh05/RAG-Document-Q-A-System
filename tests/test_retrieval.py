import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.database import Base
from db.models import Document, Chunk
from embeddings.embed import MockEmbeddingService
from retrieval.vector_search import VectorSearchService, SearchResult
from retrieval.keyword_search import BM25KeywordSearchService
from retrieval.rerank import MockReranker
from retrieval.hybrid_search import HybridSearchService


@pytest.fixture
def memory_db():
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    doc = Document(filename="guide.md", content_type="text/markdown", file_hash="hash123")
    session.add(doc)
    session.flush()

    emb_svc = MockEmbeddingService(dimension=64)
    chunks_content = [
        "PostgreSQL pgvector extension enables fast vector similarity search using HNSW indexing.",
        "BM25 keyword search is optimal for exact keyword and identifier matching.",
        "Cross-encoders score query-document pairs to re-rank candidate passages.",
        "Alembic handles database migrations for documents and chunks schemas.",
    ]

    for idx, text in enumerate(chunks_content):
        chunk = Chunk(
            document_id=doc.id,
            content=text,
            chunk_index=idx,
            page_number=1,
            embedding=emb_svc.embed_query(text),
        )
        session.add(chunk)

    session.commit()
    yield session, emb_svc
    session.close()


def test_vector_search(memory_db):
    session, emb_svc = memory_db
    v_search = VectorSearchService(embedding_service=emb_svc)
    results = v_search.search(session, query="vector similarity search", top_k=2)
    assert len(results) == 2
    assert all(isinstance(r, SearchResult) for r in results)


def test_bm25_search(memory_db):
    session, _ = memory_db
    bm25 = BM25KeywordSearchService()
    results = bm25.search(session, query="pgvector HNSW", top_k=2)
    assert len(results) >= 1
    assert "pgvector" in results[0].content.lower()


def test_reranker():
    reranker = MockReranker()
    mock_results = [
        SearchResult(chunk_id="1", document_id="d1", filename="f1", content="Unrelated topic", score=0.8),
        SearchResult(chunk_id="2", document_id="d1", filename="f1", content="Exact query keyword match", score=0.5),
    ]
    reranked = reranker.rerank(query="keyword match", results=mock_results, top_k=2)
    assert len(reranked) == 2
    assert reranked[0].chunk_id == "2"


def test_hybrid_search_pipeline(memory_db):
    session, emb_svc = memory_db
    hybrid = HybridSearchService(embedding_service=emb_svc, reranker=MockReranker())

    for pipe in ["vector_only", "bm25_only", "hybrid", "hybrid_rerank"]:
        res = hybrid.search(session, query="HNSW vector indexing", pipeline=pipe, top_k=2)
        assert isinstance(res, list)
        if pipe != "bm25_only":
            assert len(res) >= 1
