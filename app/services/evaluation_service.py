import os
import sys
import json
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import numpy as np

from app.core.config import settings
from db.database import Base
from db.models import Document, Chunk
from ingestion.loaders import DocumentLoaderFactory
from ingestion.chunking import ChunkingStrategyFactory
from embeddings.embed import get_embedding_service
from retrieval.hybrid_search import HybridSearchService
from generation.generate import GenerationService
from eval.metrics import compute_retrieval_metrics, compute_generation_metrics
from app.schemas.evaluation import EvalRow, BenchmarkRunResponse

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "eval", "results"))
LATEST_RESULTS_FILE = os.path.join(RESULTS_DIR, "latest_results.json")
BENCHMARK_DATASET_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "eval", "benchmark_dataset.json"))


class EvaluationService:
    def __init__(self):
        os.makedirs(RESULTS_DIR, exist_ok=True)

    def get_latest_results(self) -> Optional[BenchmarkRunResponse]:
        """Returns the most recent persisted evaluation run, or None if no evaluation has run."""
        if not os.path.exists(LATEST_RESULTS_FILE):
            return None
        try:
            with open(LATEST_RESULTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return BenchmarkRunResponse(**data)
        except Exception as e:
            logger.warning(f"Error reading latest evaluation results: {e}")
            return None

    def setup_evaluation_database(self) -> Any:
        """
        Creates an isolated in-memory database and indexes all sample documents
        from sample_docs/ for reproducible benchmarking.
        """
        eval_engine = create_engine("sqlite:///:memory:", echo=False, connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=eval_engine)
        Session = sessionmaker(bind=eval_engine)
        session = Session()

        sample_docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sample_docs"))
        embedding_svc = get_embedding_service()
        chunker = ChunkingStrategyFactory.get_chunker("sentence", chunk_size=300, chunk_overlap=50)

        for filename in os.listdir(sample_docs_dir):
            if filename.startswith(".") or filename.endswith(".py"):
                continue
            filepath = os.path.join(sample_docs_dir, filename)
            if not os.path.isfile(filepath):
                continue

            with open(filepath, "rb") as f:
                content = f.read()

            ext = filename.split(".")[-1].lower()
            mime = "application/pdf" if ext == "pdf" else ("text/html" if ext in ("html", "htm") else "text/markdown")
            try:
                loaded = DocumentLoaderFactory.load_bytes(content, filename, mime)
                chunks = chunker.chunk(loaded)
                embeddings = embedding_svc.embed_documents([c.content for c in chunks])

                db_doc = Document(
                    filename=filename,
                    content_type=mime,
                    file_hash=f"eval_hash_{filename}",
                    doc_metadata=loaded.metadata,
                )
                session.add(db_doc)
                session.flush()

                for c, emb in zip(chunks, embeddings):
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
            except Exception as e:
                logger.warning(f"Failed to ingest eval doc {filename}: {e}")

        return session

    def run_benchmark(
        self,
        limit: Optional[int] = None,
        pipelines: Optional[List[str]] = None,
    ) -> BenchmarkRunResponse:
        """
        Executes reproducible multi-pipeline benchmark across labeled questions,
        computes all retrieval, generation, citation, and robustness metrics,
        and persists timestamped results.
        """
        if not os.path.exists(BENCHMARK_DATASET_FILE):
            raise FileNotFoundError(f"Benchmark dataset not found at {BENCHMARK_DATASET_FILE}")

        with open(BENCHMARK_DATASET_FILE, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        if limit and limit > 0:
            dataset = dataset[:limit]

        eval_session = self.setup_evaluation_database()
        retriever = HybridSearchService()
        generator = GenerationService()

        eval_pipelines = pipelines or ["vector_only", "hybrid", "hybrid_rerank"]
        name_map = {
            "vector_only": "vector-only",
            "hybrid": "hybrid",
            "hybrid_rerank": "hybrid+rerank",
            "optimized_hybrid_rerank": "optimized-hybrid+rerank",
        }

        pipeline_rows: List[EvalRow] = []

        for pipe in eval_pipelines:
            logger.info(f"Running evaluation benchmark for pipeline: {pipe} ({len(dataset)} items)...")
            recalls_1, recalls_3, recalls_5 = [], [], []
            precisions_5 = []
            mrrs = []
            ndcgs_5 = []
            faithfulness_scores = []
            relevance_scores = []
            refusal_accuracies = []
            citation_correctness_scores = []
            unsupported_rates = []
            latencies = []

            for item in dataset:
                q = item["question"]
                target_docs = item.get("target_documents", [])
                expected_kw = item.get("expected_keywords", [])
                gt_answer = item.get("ground_truth_answer", "")
                is_ans = item.get("is_answerable", True)

                t0 = time.time()
                retrieved = retriever.search(db=eval_session, query=q, pipeline=pipe, top_k=5)
                gen = generator.generate_answer(question=q, retrieved_chunks=retrieved)
                elapsed_ms = (time.time() - t0) * 1000

                r_metrics = compute_retrieval_metrics(
                    retrieved_chunks=retrieved,
                    expected_keywords=expected_kw,
                    target_documents=target_docs,
                    is_answerable=is_ans,
                )
                g_metrics = compute_generation_metrics(
                    generated_answer=gen.answer,
                    ground_truth_answer=gt_answer,
                    retrieved_chunks=retrieved,
                    is_answerable=is_ans,
                )

                recalls_1.append(r_metrics["recall_at_1"])
                recalls_3.append(r_metrics["recall_at_3"])
                recalls_5.append(r_metrics["recall_at_5"])
                precisions_5.append(r_metrics["precision_at_5"])
                mrrs.append(r_metrics["mrr"])
                ndcgs_5.append(r_metrics["ndcg_at_5"])

                faithfulness_scores.append(g_metrics["faithfulness"])
                relevance_scores.append(g_metrics["relevance"])
                refusal_accuracies.append(g_metrics["refusal_accuracy"])
                citation_correctness_scores.append(g_metrics["citation_correctness"])
                unsupported_rates.append(g_metrics["unsupported_claim_rate"])
                latencies.append(elapsed_ms)

            n = len(dataset) or 1
            p50 = round(float(np.percentile(latencies, 50)), 1) if latencies else 0.0
            p95 = round(float(np.percentile(latencies, 95)), 1) if latencies else 0.0
            row = EvalRow(
                pipeline=name_map.get(pipe, pipe),
                precisionAt5=round(sum(precisions_5) / n, 2),
                recallAt5=round(sum(recalls_5) / n, 2),
                faithfulness=round(sum(faithfulness_scores) / n, 2),
                relevance=round(sum(relevance_scores) / n, 2),
                avgLatencyMs=round(sum(latencies) / n, 1),
                p50LatencyMs=p50,
                p95LatencyMs=p95,
                recallAt1=round(sum(recalls_1) / n, 2),
                recallAt3=round(sum(recalls_3) / n, 2),
                mrr=round(sum(mrrs) / n, 3),
                ndcgAt5=round(sum(ndcgs_5) / n, 3),
                citationCorrectness=round(sum(citation_correctness_scores) / n, 2),
                unsupportedClaimRate=round(sum(unsupported_rates) / n, 2),
                refusalAccuracy=round(sum(refusal_accuracies) / n, 2),
            )
            pipeline_rows.append(row)

        run_id = f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        timestamp_str = datetime.now(timezone.utc).isoformat()

        response = BenchmarkRunResponse(
            run_id=run_id,
            timestamp=timestamp_str,
            dataset_name="Enterprise RAG Ground-Truth Benchmark",
            dataset_size=len(dataset),
            has_run=True,
            results=pipeline_rows,
            summary={
                "top_precision_pipeline": max(pipeline_rows, key=lambda r: r.precisionAt5).pipeline,
                "top_recall_pipeline": max(pipeline_rows, key=lambda r: r.recallAt5).pipeline,
                "dataset_items_evaluated": len(dataset),
            },
        )

        # Persist results
        run_file = os.path.join(RESULTS_DIR, f"{run_id}.json")
        with open(run_file, "w", encoding="utf-8") as f:
            json.dump(response.model_dump(), f, indent=2)

        with open(LATEST_RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(response.model_dump(), f, indent=2)

        logger.info(f"Persisted benchmark run {run_id} to {run_file} and updated latest_results.json")
        return response


evaluation_service = EvaluationService()
