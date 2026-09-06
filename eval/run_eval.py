import os
import sys
import json
import time
import logging
from typing import List, Dict, Any

# Ensure project root is in sys.path when running script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import settings
from db.database import Base
from db.models import Document, Chunk
from ingestion.loaders import DocumentLoaderFactory
from ingestion.chunking import ChunkingStrategyFactory
from embeddings.embed import get_embedding_service
from retrieval.hybrid_search import HybridSearchService
from generation.generate import GenerationService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_harness")


def setup_eval_db():
    """Sets up an isolated in-memory SQLite DB populated with sample evaluation documents."""
    eval_engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=eval_engine)
    Session = sessionmaker(bind=eval_engine)
    session = Session()

    # Ingest sample architecture document
    doc_path = os.path.join(os.path.dirname(__file__), "..", "sample_docs", "example_architecture.md")
    if os.path.exists(doc_path):
        with open(doc_path, "rb") as f:
            content = f.read()

        loaded_doc = DocumentLoaderFactory.load_bytes(content, "example_architecture.md", "text/markdown")
        chunker = ChunkingStrategyFactory.get_chunker("sentence", chunk_size=300, chunk_overlap=50)
        chunks = chunker.chunk(loaded_doc)

        embedding_svc = get_embedding_service()
        chunk_embeddings = embedding_svc.embed_documents([c.content for c in chunks])

        db_doc = Document(
            filename="example_architecture.md",
            content_type="text/markdown",
            file_hash="eval_hash_architecture",
            doc_metadata=loaded_doc.metadata,
        )
        session.add(db_doc)
        session.flush()

        for c, emb in zip(chunks, chunk_embeddings):
            session.add(
                Chunk(
                    document_id=db_doc.id,
                    content=c.content,
                    chunk_index=c.chunk_index,
                    page_number=c.page_number,
                    chunking_strategy="sentence",
                    embedding=emb,
                    chunk_metadata=c.metadata,
                )
            )

        session.commit()

    return session


def compute_metrics_for_query(
    expected_keywords: List[str],
    retrieved_chunks: List[Any],
    generated_answer: str,
    ground_truth_answer: str,
    top_k: int = 5,
) -> Dict[str, float]:
    """Computes Precision@k, Recall@k, and Faithfulness/Relevance."""
    chunks = retrieved_chunks[:top_k]
    if not chunks:
        return {"precision": 0.0, "recall": 0.0, "faithfulness": 0.0, "relevance": 0.0}

    # 1. Retrieval Metrics
    relevant_chunks = 0
    keywords_found = set()

    for c in chunks:
        content_lower = c.content.lower()
        chunk_has_match = False
        for kw in expected_keywords:
            if kw.lower() in content_lower:
                keywords_found.add(kw.lower())
                chunk_has_match = True
        if chunk_has_match:
            relevant_chunks += 1

    precision_at_k = relevant_chunks / len(chunks) if chunks else 0.0
    recall_at_k = len(keywords_found) / len(expected_keywords) if expected_keywords else 0.0

    # 2. Generation Quality (Faithfulness & Relevance)
    # Check if answer contains citation brackets [Chunk ...]
    has_citations = "[" in generated_answer and "]" in generated_answer
    ground_truth_tokens = set(ground_truth_answer.lower().split())
    answer_tokens = set(generated_answer.lower().split())
    token_overlap = len(ground_truth_tokens.intersection(answer_tokens)) / max(1, len(ground_truth_tokens))

    faithfulness = 1.0 if (has_citations and relevant_chunks > 0) else (0.8 if has_citations else 0.5)
    relevance = min(1.0, token_overlap * 1.5 + 0.3)

    return {
        "precision": precision_at_k,
        "recall": recall_at_k,
        "faithfulness": faithfulness,
        "relevance": relevance,
    }


def run_evaluation():
    test_set_path = os.path.join(os.path.dirname(__file__), "test_set.json")
    with open(test_set_path, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    logger.info(f"Loaded {len(test_set)} test questions from {test_set_path}")

    session = setup_eval_db()
    retriever = HybridSearchService()
    generator = GenerationService()

    pipelines = ["vector_only", "hybrid", "hybrid_rerank"]
    pipeline_results: Dict[str, Dict[str, Any]] = {}

    for pipeline in pipelines:
        logger.info(f"Evaluating pipeline: {pipeline} ...")
        precisions = []
        recalls = []
        faithfulness_scores = []
        relevance_scores = []
        latencies = []

        for item in test_set:
            q = item["question"]
            expected_kw = item["expected_keywords"]
            gt = item["ground_truth_answer"]

            t0 = time.time()
            retrieved = retriever.search(db=session, query=q, pipeline=pipeline, top_k=5)
            gen = generator.generate_answer(question=q, retrieved_chunks=retrieved)
            elapsed_ms = (time.time() - t0) * 1000

            metrics = compute_metrics_for_query(
                expected_keywords=expected_kw,
                retrieved_chunks=retrieved,
                generated_answer=gen.answer,
                ground_truth_answer=gt,
                top_k=5,
            )

            precisions.append(metrics["precision"])
            recalls.append(metrics["recall"])
            faithfulness_scores.append(metrics["faithfulness"])
            relevance_scores.append(metrics["relevance"])
            latencies.append(elapsed_ms)

        pipeline_results[pipeline] = {
            "precision": sum(precisions) / len(precisions) if precisions else 0.0,
            "recall": sum(recalls) / len(recalls) if recalls else 0.0,
            "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0,
            "relevance": sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        }

    # Print Markdown Benchmark Table
    print("\n" + "=" * 65)
    print("           RAG RETRIEVAL & GENERATION BENCHMARK RESULTS")
    print("=" * 65)
    print("\n| Pipeline | Precision@5 | Recall@5 | Faithfulness | Relevance | Avg Latency |")
    print("|---|---|---|---|---|---|")
    name_map = {
        "vector_only": "Vector-only (baseline)",
        "hybrid": "Hybrid search (BM25 + Vector)",
        "hybrid_rerank": "Hybrid + Cross-Encoder Re-rank",
    }
    for pipe, res in pipeline_results.items():
        print(
            f"| {name_map.get(pipe, pipe)} | {res['precision']:.2f} | {res['recall']:.2f} | "
            f"{res['faithfulness']:.2f} | {res['relevance']:.2f} | {res['avg_latency_ms']:.1f}ms |"
        )
    print("\n" + "=" * 65 + "\n")

    return pipeline_results


if __name__ == "__main__":
    run_evaluation()
