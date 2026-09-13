import io
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from app.services.ingestion_service import IngestionService
from app.core.rate_limit import SlidingWindowRateLimiter
from app.core.security import get_current_user_id
from app.core.config import settings
from app.main import app

client = TestClient(app)


def test_ingestion_service_rejects_unsupported_extension():
    svc = IngestionService()
    with pytest.raises(HTTPException) as exc:
        svc._validate_content_format(b"echo hello", "exe", "script.exe")
    # Also test read_and_validate_upload with invalid ext
    class MockFile:
        filename = "virus.exe"
        content_type = "application/x-msdownload"
        async def read(self, n):
            return b""
    import asyncio
    with pytest.raises(HTTPException) as exc:
        asyncio.run(svc.read_and_validate_upload(MockFile()))
    assert exc.value.status_code == 400


def test_pdf_magic_bytes_validation():
    svc = IngestionService()
    # Invalid PDF lacking %PDF- header
    with pytest.raises(HTTPException) as exc:
        svc._validate_content_format(b"This is just plain text masquerading as a PDF", "pdf", "fake.pdf")
    assert exc.value.status_code == 400
    assert "magic header" in exc.value.detail

    # Valid PDF magic header
    svc._validate_content_format(b"%PDF-1.4\n%real pdf content...", "pdf", "real.pdf")


def test_binary_null_byte_rejection_in_text():
    svc = IngestionService()
    with pytest.raises(HTTPException) as exc:
        svc._validate_content_format(b"Text with malicious null byte: \x00 here", "txt", "attack.txt")
    assert exc.value.status_code == 400
    assert "null byte" in exc.value.detail


def test_sliding_window_rate_limiter_allows_under_limit():
    limiter = SlidingWindowRateLimiter(requests_per_minute=10)
    for _ in range(10):
        limiter.check("client_test_1")


def test_sliding_window_rate_limiter_blocks_over_limit():
    limiter = SlidingWindowRateLimiter(requests_per_minute=5)
    for _ in range(5):
        limiter.check("client_test_2")
    with pytest.raises(HTTPException) as exc:
        limiter.check("client_test_2")
    assert exc.value.status_code == 429
    assert "Rate limit exceeded" in exc.value.detail
    assert "Retry-After" in exc.value.headers


def test_api_key_auth_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "rag_sec_test_secret_key_123")
    
    # Missing API key -> 401
    with pytest.raises(HTTPException) as exc1:
        get_current_user_id(None, None, None)
    assert exc1.value.status_code == 401

    # Wrong API key -> 401
    with pytest.raises(HTTPException) as exc2:
        get_current_user_id(None, "Bearer wrong_token", None)
    assert exc2.value.status_code == 401

    # Valid X-API-Key -> success
    uid = get_current_user_id(None, None, "rag_sec_test_secret_key_123")
    assert uid == "default_user"

    # Valid Bearer -> success
    uid_bearer = get_current_user_id("custom_tenant", "Bearer rag_sec_test_secret_key_123", None)
    assert uid_bearer == "custom_tenant"


def test_request_id_middleware_attaches_headers():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "X-Request-ID" in resp.headers
    assert "X-Response-Time-Ms" in resp.headers
    assert len(resp.headers["X-Request-ID"]) > 5


def test_custom_request_id_propagated():
    custom_id = "test-correlation-uuid-9876"
    resp = client.get("/health", headers={"X-Request-ID": custom_id})
    assert resp.status_code == 200
    assert resp.headers["X-Request-ID"] == custom_id


def test_upload_empty_file_rejected_via_api():
    file_payload = ("empty.txt", io.BytesIO(b""), "text/plain")
    resp = client.post("/ingest", files={"file": file_payload})
    assert resp.status_code == 400


def test_upload_invalid_extension_rejected_via_api():
    file_payload = ("test.sh", io.BytesIO(b"#!/bin/bash\nrm -rf /"), "application/x-sh")
    resp = client.post("/ingest", files={"file": file_payload})
    assert resp.status_code == 400
    assert "Unsupported file extension" in resp.json()["detail"]


def test_upload_corrupt_pdf_rejected_via_api():
    file_payload = ("malicious.pdf", io.BytesIO(b"NOT A REAL PDF HEADER"), "application/pdf")
    resp = client.post("/ingest", files={"file": file_payload})
    assert resp.status_code == 400
    assert "magic header" in resp.json()["detail"]
