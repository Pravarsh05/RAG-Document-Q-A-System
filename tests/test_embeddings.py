import pytest
import numpy as np
from embeddings.embed import MockEmbeddingService, get_embedding_service


def test_mock_embedding_dimensions():
    svc = MockEmbeddingService(dimension=128)
    assert svc.dimension == 128
    
    vec = svc.embed_query("test query")
    assert len(vec) == 128
    
    # Verify unit vector norm
    norm = np.linalg.norm(vec)
    assert np.isclose(norm, 1.0, atol=1e-3)


def test_mock_embedding_batch():
    svc = MockEmbeddingService(dimension=64)
    docs = ["doc 1 text", "doc 2 text", "doc 3 text"]
    embeddings = svc.embed_documents(docs)
    assert len(embeddings) == 3
    for emb in embeddings:
        assert len(emb) == 64


def test_mock_deterministic_embedding():
    svc = MockEmbeddingService(dimension=64)
    v1 = svc.embed_query("identical text")
    v2 = svc.embed_query("identical text")
    assert v1 == v2
