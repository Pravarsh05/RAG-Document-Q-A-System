import pytest
from app.services.citation_verifier import CitationVerifier
from retrieval.vector_search import SearchResult


@pytest.fixture
def verifier():
    return CitationVerifier()


@pytest.fixture
def sample_chunks():
    return [
        SearchResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            filename="arch.md",
            content="pgvector on PostgreSQL provides HNSW indexing for scalable vector similarity search.",
            chunk_index=0,
            score=0.92,
        ),
        SearchResult(
            chunk_id="chunk-2",
            document_id="doc-1",
            filename="arch.md",
            content="Reciprocal Rank Fusion merges dense vector search with sparse BM25 keyword rankings.",
            chunk_index=1,
            score=0.88,
        ),
    ]


def test_extract_claims_with_numeric_citations(verifier):
    answer = (
        "- pgvector provides HNSW indexing for vector search [1].\n"
        "- Reciprocal Rank Fusion combines BM25 and vector lists [2].\n"
    )
    claims = verifier.extract_claims_with_citations(answer)
    assert len(claims) == 2
    assert claims[0]["numeric_cites"] == [1]
    assert claims[1]["numeric_cites"] == [2]


def test_extract_claims_with_named_citations(verifier):
    answer = "- System uses HNSW [Chunk 1].\n- Combines BM25 [Chunk chunk-2]."
    claims = verifier.extract_claims_with_citations(answer)
    assert len(claims) == 2
    assert "1" in claims[0]["named_cites"]
    assert "chunk-2" in claims[1]["named_cites"]


def test_verify_supported_citation(verifier, sample_chunks):
    answer = "- pgvector provides HNSW indexing for scalable vector similarity search [1]."
    res = verifier.verify(answer, sample_chunks)
    assert res["grounding_status"] == "grounded"
    assert res["citation_correctness"] >= 0.8
    assert res["unsupported_claim_rate"] == 0.0
    assert len(res["claim_verifications"]) == 1
    assert res["claim_verifications"][0]["is_supported"] is True


def test_verify_unsupported_citation_mismatch(verifier, sample_chunks):
    # Claim cites chunk 1, but discusses quantum computing Shor's algorithm
    answer = "- Quantum computers decrypt RSA encryption in polynomial time [1]."
    res = verifier.verify(answer, sample_chunks)
    assert res["grounding_status"] == "citation_mismatch"
    assert res["unsupported_claim_rate"] > 0.0
    assert res["claim_verifications"][0]["is_supported"] is False


def test_verify_claim_without_citations(verifier, sample_chunks):
    answer = "The system uses quantum entanglement."
    res = verifier.verify(answer, sample_chunks)
    assert res["unsupported_claim_rate"] == 1.0
    assert res["grounding_status"] == "insufficient_evidence"


def test_refusal_detection(verifier, sample_chunks):
    answer = "I cannot answer this question based on the provided documents as no relevant context was found."
    assert verifier.is_refusal_answer(answer) is True

    res = verifier.verify(answer, sample_chunks)
    assert res["grounding_status"] == "refusal"
    assert res["unsupported_claim_rate"] == 0.0
    assert res["citation_correctness"] == 1.0


def test_empty_answer_handling(verifier, sample_chunks):
    res = verifier.verify("", sample_chunks)
    assert res["grounding_status"] == "insufficient_evidence"
    assert res["unsupported_claim_rate"] == 1.0


def test_empty_chunks_handling(verifier):
    answer = "- pgvector provides HNSW [1]."
    res = verifier.verify(answer, [])
    assert res["grounding_status"] == "insufficient_evidence"


def test_out_of_range_citation_number(verifier, sample_chunks):
    # Cites chunk [99] which doesn't exist
    answer = "- Feature requires high memory [99]."
    res = verifier.verify(answer, sample_chunks)
    assert res["unsupported_claim_rate"] == 1.0
    assert res["claim_verifications"][0]["is_supported"] is False


def test_multiple_citations_per_claim(verifier, sample_chunks):
    answer = "- The architecture combines pgvector HNSW with Reciprocal Rank Fusion [1] [2]."
    res = verifier.verify(answer, sample_chunks)
    assert res["grounding_status"] == "grounded"
    assert res["claim_verifications"][0]["is_supported"] is True
