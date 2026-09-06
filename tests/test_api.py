import io
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sqlalchemy.pool import StaticPool
from db.database import Base, get_db
from main import app

# Setup test in-memory SQLite database with StaticPool
test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
Base.metadata.create_all(bind=test_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "database" in data


def test_ingest_and_query_flow():
    # 1. Ingest markdown document
    md_content = b"# Vector Search Guide\n\npgvector provides HNSW indexes for high performance."
    file_payload = ("test_guide.md", io.BytesIO(md_content), "text/markdown")

    ingest_resp = client.post(
        "/ingest",
        files={"file": file_payload},
        data={"chunking_strategy": "sentence", "chunk_size": 200, "chunk_overlap": 50},
    )
    assert ingest_resp.status_code == 200
    ingest_data = ingest_resp.json()
    assert ingest_data["filename"] == "test_guide.md"
    assert ingest_data["chunk_count"] >= 1
    doc_id = ingest_data["document_id"]

    # 2. List documents
    docs_resp = client.get("/documents")
    assert docs_resp.status_code == 200
    docs = docs_resp.json()
    assert len(docs) >= 1
    assert any(d["id"] == doc_id for d in docs)

    # 3. Query document
    query_resp = client.post(
        "/query",
        json={
            "question": "What does pgvector provide?",
            "pipeline": "hybrid_rerank",
            "top_k": 3,
        },
    )
    assert query_resp.status_code == 200
    q_data = query_resp.json()
    assert "answer" in q_data
    assert "citations" in q_data
    assert "retrieved_chunks" in q_data
    assert len(q_data["retrieved_chunks"]) >= 1

    # 4. Clean up / Delete document
    del_resp = client.delete(f"/documents/{doc_id}")
    assert del_resp.status_code == 200


def test_eval_and_config_endpoints():
    # Test GET /eval/results
    eval_resp = client.get("/eval/results")
    assert eval_resp.status_code == 200
    eval_data = eval_resp.json()
    assert len(eval_data) >= 3
    assert any(row["pipeline"] == "hybrid+rerank" for row in eval_data)

    # Test GET /config
    cfg_resp = client.get("/config")
    assert cfg_resp.status_code == 200
    cfg_data = cfg_resp.json()
    assert "embedding_provider" in cfg_data
    assert "cache_enabled" in cfg_data

    # Test PATCH /config
    patch_resp = client.patch("/config", json={"cache_enabled": True})
    assert patch_resp.status_code == 200
    assert patch_resp.json()["cache_enabled"] is True

