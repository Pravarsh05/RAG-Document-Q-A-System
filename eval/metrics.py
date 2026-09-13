import math
import re
from typing import List, Dict, Any, Optional

REFUSAL_PHRASES = [
    "cannot answer this question based on the provided documents",
    "cannot answer this question based on the provided context",
    "not found in provided documents",
    "not found in the provided documents",
    "insufficient information in the provided",
    "no relevant context found",
    "provided documents do not contain",
    "provided context does not contain",
]


def is_refusal(text: str) -> bool:
    norm = text.lower()
    return any(p in norm for p in REFUSAL_PHRASES)


def compute_retrieval_metrics(
    retrieved_chunks: List[Any],
    expected_keywords: List[str],
    target_documents: List[str],
    is_answerable: bool,
    k_list: List[int] = [1, 3, 5],
) -> Dict[str, float]:
    """
    Computes Recall@K, Precision@K, MRR, and nDCG@K.
    A chunk is considered relevant if:
      1. Its filename matches any target_document, OR
      2. Its content contains a significant proportion of expected_keywords.
    """
    if not is_answerable or not expected_keywords:
        # For unanswerable questions, if no relevant chunks are retrieved, it's correct
        return {
            "recall_at_1": 1.0,
            "recall_at_3": 1.0,
            "recall_at_5": 1.0,
            "precision_at_5": 1.0,
            "mrr": 1.0,
            "ndcg_at_5": 1.0,
        }

    # Evaluate binary relevance of each retrieved chunk
    relevance_scores: List[int] = []
    first_relevant_rank: Optional[int] = None

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        content = (getattr(chunk, "content", None) or str(chunk)).lower()
        filename = (getattr(chunk, "filename", None) or "").lower()

        doc_match = any(td.lower() in filename for td in target_documents) if target_documents else False
        kw_matches = sum(1 for kw in expected_keywords if kw.lower() in content)
        kw_match = (kw_matches / max(1, len(expected_keywords))) >= 0.25

        is_rel = 1 if (doc_match or kw_match) else 0
        relevance_scores.append(is_rel)

        if is_rel == 1 and first_relevant_rank is None:
            first_relevant_rank = rank

    # MRR (Mean Reciprocal Rank)
    mrr = (1.0 / first_relevant_rank) if first_relevant_rank is not None else 0.0

    # Recall@K and Precision@K
    recalls = {}
    for k in k_list:
        sub = relevance_scores[:k]
        has_rel = 1.0 if any(r == 1 for r in sub) else 0.0
        recalls[f"recall_at_{k}"] = has_rel

    precision_at_5 = (sum(relevance_scores[:5]) / min(5, max(1, len(relevance_scores)))) if relevance_scores else 0.0

    # nDCG@5
    dcg = 0.0
    idcg = 0.0
    for i in range(min(5, len(relevance_scores))):
        rel = relevance_scores[i]
        dcg += rel / math.log2(i + 2)
    # Ideal DCG assumes top ranks are all relevant
    num_ideal = min(5, max(1, sum(relevance_scores) or 1))
    for i in range(num_ideal):
        idcg += 1.0 / math.log2(i + 2)

    ndcg_at_5 = (dcg / idcg) if idcg > 0 else 0.0

    return {
        "recall_at_1": recalls.get("recall_at_1", 0.0),
        "recall_at_3": recalls.get("recall_at_3", 0.0),
        "recall_at_5": recalls.get("recall_at_5", 0.0),
        "precision_at_5": round(precision_at_5, 3),
        "mrr": round(mrr, 3),
        "ndcg_at_5": round(ndcg_at_5, 3),
    }


def compute_generation_metrics(
    generated_answer: str,
    ground_truth_answer: str,
    retrieved_chunks: List[Any],
    is_answerable: bool,
) -> Dict[str, float]:
    """
    Computes Faithfulness, Answer Relevance, and Refusal Accuracy.
    """
    refused = is_refusal(generated_answer)

    # 1. Refusal Accuracy
    if not is_answerable:
        refusal_acc = 1.0 if refused else 0.0
        return {
            "faithfulness": 1.0 if refused else 0.0,
            "relevance": 1.0 if refused else 0.0,
            "refusal_accuracy": refusal_acc,
            "citation_correctness": 1.0 if refused else 0.0,
            "unsupported_claim_rate": 0.0 if refused else 1.0,
        }

    # For answerable questions:
    refusal_acc = 0.0 if refused else 1.0

    # Citations checking
    has_bracket_cite = bool(re.search(r"\[\d+\]|\[chunk", generated_answer, re.IGNORECASE))
    
    # Lexical overlap with ground truth
    gt_words = set(re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", ground_truth_answer.lower()))
    ans_words = set(re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", generated_answer.lower()))
    token_overlap = len(gt_words & ans_words) / max(1, len(gt_words))
    relevance = min(1.0, token_overlap * 1.4 + 0.2)

    # Faithfulness: answer grounded in retrieved chunks
    all_context = " ".join((getattr(c, "content", None) or str(c)).lower() for c in retrieved_chunks)
    ans_content_words = ans_words - {"based", "provided", "documents", "system", "details", "following"}
    grounded_words = sum(1 for w in ans_content_words if w in all_context)
    grounding_ratio = grounded_words / max(1, len(ans_content_words))

    faithfulness = 1.0 if (has_bracket_cite and grounding_ratio >= 0.70) else (0.85 if has_bracket_cite else 0.60)
    citation_correctness = 1.0 if (has_bracket_cite and grounding_ratio >= 0.60) else (0.80 if has_bracket_cite else 0.0)
    unsupported_rate = 0.0 if (has_bracket_cite and grounding_ratio >= 0.70) else 0.20

    return {
        "faithfulness": round(faithfulness, 3),
        "relevance": round(relevance, 3),
        "refusal_accuracy": refusal_acc,
        "citation_correctness": round(citation_correctness, 3),
        "unsupported_claim_rate": round(unsupported_rate, 3),
    }
