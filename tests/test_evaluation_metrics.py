import pytest
from eval.metrics import compute_retrieval_metrics, compute_generation_metrics, is_refusal
from retrieval.vector_search import SearchResult


def test_is_refusal_detection():
    assert is_refusal("I cannot answer this question based on the provided documents.") is True
    assert is_refusal("Information not found in provided documents.") is True
    assert is_refusal("The system uses pgvector for HNSW indexing.") is False


def test_retrieval_metrics_perfect_ranking():
    chunks = [
        SearchResult(chunk_id="c1", document_id="d1", filename="arch.md", content="pgvector HNSW index", chunk_index=0, score=0.9),
        SearchResult(chunk_id="c2", document_id="d1", filename="arch.md", content="BM25 search", chunk_index=1, score=0.8),
    ]
    res = compute_retrieval_metrics(
        retrieved_chunks=chunks,
        expected_keywords=["pgvector", "HNSW"],
        target_documents=["arch.md"],
        is_answerable=True,
    )
    assert res["recall_at_1"] == 1.0
    assert res["recall_at_3"] == 1.0
    assert res["recall_at_5"] == 1.0
    assert res["precision_at_5"] > 0.0
    assert res["mrr"] == 1.0
    assert res["ndcg_at_5"] == 1.0


def test_retrieval_metrics_second_rank():
    chunks = [
        SearchResult(chunk_id="c1", document_id="d1", filename="other.md", content="unrelated topic", chunk_index=0, score=0.9),
        SearchResult(chunk_id="c2", document_id="d1", filename="arch.md", content="pgvector HNSW index", chunk_index=1, score=0.8),
    ]
    res = compute_retrieval_metrics(
        retrieved_chunks=chunks,
        expected_keywords=["pgvector", "HNSW"],
        target_documents=["arch.md"],
        is_answerable=True,
    )
    assert res["recall_at_1"] == 0.0
    assert res["recall_at_3"] == 1.0
    assert res["mrr"] == 0.5  # 1/2


def test_retrieval_metrics_no_match():
    chunks = [
        SearchResult(chunk_id="c1", document_id="d1", filename="other.md", content="completely unrelated", chunk_index=0, score=0.9),
    ]
    res = compute_retrieval_metrics(
        retrieved_chunks=chunks,
        expected_keywords=["quantum", "qubit"],
        target_documents=["quantum.md"],
        is_answerable=True,
    )
    assert res["recall_at_1"] == 0.0
    assert res["recall_at_5"] == 0.0
    assert res["mrr"] == 0.0
    assert res["ndcg_at_5"] == 0.0


def test_retrieval_metrics_unanswerable_question():
    res = compute_retrieval_metrics(
        retrieved_chunks=[],
        expected_keywords=[],
        target_documents=[],
        is_answerable=False,
    )
    assert res["recall_at_5"] == 1.0
    assert res["mrr"] == 1.0


def test_generation_metrics_grounded_answer():
    chunks = [
        SearchResult(chunk_id="c1", document_id="d1", filename="arch.md", content="pgvector provides HNSW indexes for PostgreSQL.", chunk_index=0, score=0.9),
    ]
    answer = "Based on the documents, pgvector provides HNSW indexes for PostgreSQL [1]."
    gt = "pgvector provides HNSW indexes on PostgreSQL."
    res = compute_generation_metrics(
        generated_answer=answer,
        ground_truth_answer=gt,
        retrieved_chunks=chunks,
        is_answerable=True,
    )
    assert res["faithfulness"] >= 0.85
    assert res["relevance"] >= 0.5
    assert res["citation_correctness"] >= 0.8
    assert res["unsupported_claim_rate"] <= 0.2
    assert res["refusal_accuracy"] == 1.0  # correct answerable response


def test_generation_metrics_unanswerable_refusal():
    chunks = []
    answer = "I cannot answer this question based on the provided documents as no relevant context was found."
    gt = "I cannot answer this question based on the provided documents."
    res = compute_generation_metrics(
        generated_answer=answer,
        ground_truth_answer=gt,
        retrieved_chunks=chunks,
        is_answerable=False,
    )
    assert res["refusal_accuracy"] == 1.0
    assert res["faithfulness"] == 1.0
    assert res["citation_correctness"] == 1.0
    assert res["unsupported_claim_rate"] == 0.0


def test_generation_metrics_unanswerable_hallucination():
    chunks = []
    # Model attempts to answer an unanswerable question
    answer = "The Kubernetes join token is kubeadm-join-12345."
    gt = "I cannot answer this question."
    res = compute_generation_metrics(
        generated_answer=answer,
        ground_truth_answer=gt,
        retrieved_chunks=chunks,
        is_answerable=False,
    )
    assert res["refusal_accuracy"] == 0.0
    assert res["faithfulness"] == 0.0
    assert res["citation_correctness"] == 0.0


def test_generation_metrics_uncited_answer():
    chunks = [SearchResult(chunk_id="c1", document_id="d1", filename="arch.md", content="content here", chunk_index=0, score=0.9)]
    answer = "Here is an answer that forgot all citations entirely."
    gt = "content here"
    res = compute_generation_metrics(
        generated_answer=answer,
        ground_truth_answer=gt,
        retrieved_chunks=chunks,
        is_answerable=True,
    )
    assert res["citation_correctness"] == 0.0


def test_retrieval_metrics_ndcg_discounting():
    # Rank 1 hit vs Rank 2 hit
    chunks_1 = [
        SearchResult(chunk_id="c1", document_id="d1", filename="doc.md", content="target content", chunk_index=0, score=0.9),
    ]
    chunks_2 = [
        SearchResult(chunk_id="c0", document_id="d0", filename="other.md", content="irrelevant", chunk_index=0, score=0.95),
        SearchResult(chunk_id="c1", document_id="d1", filename="doc.md", content="target content", chunk_index=1, score=0.9),
    ]
    res_1 = compute_retrieval_metrics(chunks_1, ["target"], ["doc.md"], True)
    res_2 = compute_retrieval_metrics(chunks_2, ["target"], ["doc.md"], True)
    assert res_1["ndcg_at_5"] > res_2["ndcg_at_5"]


def test_empty_retrieved_chunks_for_answerable():
    res = compute_retrieval_metrics([], ["keyword"], ["doc.md"], True)
    assert res["recall_at_1"] == 0.0
    assert res["recall_at_5"] == 0.0
    assert res["mrr"] == 0.0
    assert res["ndcg_at_5"] == 0.0
    assert res["precision_at_5"] == 0.0


def test_partial_keyword_overlap_threshold():
    # 1 out of 4 keywords found (25%) should count as match
    chunks = [
        SearchResult(chunk_id="c1", document_id="d1", filename="u.md", content="alpha bravo other words", chunk_index=0, score=0.9)
    ]
    res = compute_retrieval_metrics(chunks, ["alpha", "charlie", "delta", "echo"], [], True)
    assert res["recall_at_1"] == 1.0
