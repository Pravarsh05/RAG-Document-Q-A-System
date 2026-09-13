import os
import sys
import json
import time
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.evaluation_service import evaluation_service
from retrieval.hybrid_search import HybridSearchService
from eval.metrics import compute_retrieval_metrics
from app.services.query_rewriter import query_rewriter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("retrieval_experiment")


def run_retrieval_experiment(limit: int = None):
    logger.info("Initializing retrieval optimization benchmark experiment...")
    dataset_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "benchmark_dataset.json"))
    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if limit:
        dataset = dataset[:limit]

    session = evaluation_service.setup_evaluation_database()

    # Define Configurations
    configurations = [
        {
            "name": "Vector-Only (Baseline)",
            "pipeline": "vector_only",
            "candidate_k": 5,
            "top_k": 5,
            "rrf_k": 60,
            "rewrite": False,
        },
        {
            "name": "BM25-Only (Lexical)",
            "pipeline": "bm25_only",
            "candidate_k": 5,
            "top_k": 5,
            "rrf_k": 60,
            "rewrite": False,
        },
        {
            "name": "Hybrid (Vector + BM25 RRF k=60)",
            "pipeline": "hybrid",
            "candidate_k": 20,
            "top_k": 5,
            "rrf_k": 60,
            "rewrite": False,
        },
        {
            "name": "Hybrid + Cross-Encoder (ms-marco)",
            "pipeline": "hybrid_rerank",
            "candidate_k": 20,
            "top_k": 5,
            "rrf_k": 60,
            "rewrite": False,
        },
        {
            "name": "Optimized Hybrid + Rerank + Query Rewriter",
            "pipeline": "hybrid_rerank",
            "candidate_k": 25,
            "top_k": 5,
            "rrf_k": 40,
            "rewrite": True,
        },
    ]

    experiment_results = []

    for cfg in configurations:
        logger.info(f"Evaluating configuration: {cfg['name']} ...")
        retriever = HybridSearchService(rrf_k=cfg["rrf_k"])

        recalls_1 = []
        recalls_3 = []
        recalls_5 = []
        precisions_5 = []
        mrrs = []
        ndcgs_5 = []
        latencies = []

        for item in dataset:
            q = item["question"]
            target_docs = item.get("target_documents", [])
            expected_kw = item.get("expected_keywords", [])
            is_ans = item.get("is_answerable", True)

            query_to_run = q
            if cfg["rewrite"]:
                rewritten, expansions = query_rewriter.rewrite(q)
                if expansions:
                    query_to_run = rewritten

            t0 = time.time()
            retrieved = retriever.search(
                db=session,
                query=query_to_run,
                pipeline=cfg["pipeline"],
                top_k=cfg["top_k"],
                candidate_k=cfg["candidate_k"],
            )
            elapsed_ms = (time.time() - t0) * 1000

            metrics = compute_retrieval_metrics(
                retrieved_chunks=retrieved,
                expected_keywords=expected_kw,
                target_documents=target_docs,
                is_answerable=is_ans,
            )

            recalls_1.append(metrics["recall_at_1"])
            recalls_3.append(metrics["recall_at_3"])
            recalls_5.append(metrics["recall_at_5"])
            precisions_5.append(metrics["precision_at_5"])
            mrrs.append(metrics["mrr"])
            ndcgs_5.append(metrics["ndcg_at_5"])
            latencies.append(elapsed_ms)

        n = len(dataset) or 1
        res = {
            "configuration": cfg["name"],
            "pipeline": cfg["pipeline"],
            "candidate_k": cfg["candidate_k"],
            "top_k": cfg["top_k"],
            "rrf_k": cfg["rrf_k"],
            "query_rewriter": cfg["rewrite"],
            "recall_at_1": round(sum(recalls_1) / n, 3),
            "recall_at_3": round(sum(recalls_3) / n, 3),
            "recall_at_5": round(sum(recalls_5) / n, 3),
            "precision_at_5": round(sum(precisions_5) / n, 3),
            "mrr": round(sum(mrrs) / n, 3),
            "ndcg_at_5": round(sum(ndcgs_5) / n, 3),
            "avg_retrieval_latency_ms": round(sum(latencies) / n, 2),
        }
        experiment_results.append(res)

    out_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "experiment_results.json"))
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2)

    print("\n" + "=" * 88)
    print("                EMPIRICAL RETRIEVAL OPTIMIZATION EXPERIMENT")
    print("=" * 88)
    print("\n| Architecture Configuration | Recall@1 | Recall@5 | Precision@5 | MRR | nDCG@5 | Latency |")
    print("|---|---|---|---|---|---|---|")
    for r in experiment_results:
        print(
            f"| {r['configuration']:<42} | {r['recall_at_1']:.2f} | {r['recall_at_5']:.2f} | "
            f"{r['precision_at_5']:.2f} | {r['mrr']:.3f} | {r['ndcg_at_5']:.3f} | {r['avg_retrieval_latency_ms']:.1f}ms |"
        )
    print("\n" + "=" * 88 + "\n")
    return experiment_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_retrieval_experiment(limit=args.limit)
