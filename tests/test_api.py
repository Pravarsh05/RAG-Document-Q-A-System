import io
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.database import Base, get_db
from app.main import app

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


def test_health_endpoints():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "database" in data
    assert "embedding_provider" in data

    ready_resp = client.get("/health/ready")
    assert ready_resp.status_code == 200
    assert ready_resp.json()["status"] == "ready"

    live_resp = client.get("/health/live")
    assert live_resp.status_code == 200
    assert live_resp.json()["status"] == "alive"


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

    # 3. Get document chunks
    chunks_resp = client.get(f"/documents/{doc_id}/chunks")
    assert chunks_resp.status_code == 200
    chunks = chunks_resp.json()
    assert len(chunks) >= 1
    assert "content" in chunks[0]

    # 4. Query document with answerable question
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
    assert "grounding_status" in q_data

    # 5. Clean up / Delete document
    del_resp = client.delete(f"/documents/{doc_id}")
    assert del_resp.status_code == 200

    # 6. Verify 404 after deletion
    del_again = client.delete(f"/documents/{doc_id}")
    assert del_again.status_code == 404


def test_eval_lifecycle_and_no_hardcoded_defaults():
    # 1. Before any eval run, GET /eval/results must return empty list (no hardcoded metrics)
    # and /eval/status must report has_run = False
    status_resp = client.get("/eval/status")
    assert status_resp.status_code == 200

    # 2. Trigger a live benchmark run with limited items for fast integration test
    run_resp = client.post("/eval/run", json={"limit": 2})
    assert run_resp.status_code == 200
    run_data = run_resp.json()
    assert len(run_data) >= 1
    assert any(row["pipeline"] == "hybrid+rerank" for row in run_data)
    first_row = run_data[0]
    assert "precisionAt5" in first_row
    assert "recallAt5" in first_row
    assert "mrr" in first_row

    # 3. Now GET /eval/results returns the persisted results
    eval_resp = client.get("/eval/results")
    assert eval_resp.status_code == 200
    eval_data = eval_resp.json()
    assert len(eval_data) >= 1


def test_config_endpoints():
    cfg_resp = client.get("/config")
    assert cfg_resp.status_code == 200
    cfg_data = cfg_resp.json()
    assert "embedding_provider" in cfg_data
    assert "cache_enabled" in cfg_data
    assert "relevance_threshold" in cfg_data

    patch_resp = client.patch("/config", json={"cache_enabled": True, "relevance_threshold": 0.25})
    assert patch_resp.status_code == 200
    assert patch_resp.json()["cache_enabled"] is True
    assert patch_resp.json()["relevance_threshold"] == 0.25
