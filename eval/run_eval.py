import os
import sys
import logging

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.evaluation_service import evaluation_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_harness")


def run_evaluation(limit: int = None):
    logger.info("Executing live RAG evaluation benchmark harness...")
    run_resp = evaluation_service.run_benchmark(limit=limit)

    print("\n" + "=" * 80)
    print(f"      ENTERPRISE RAG BENCHMARK RESULTS (Run ID: {run_resp.run_id})")
    print(f"      Dataset: {run_resp.dataset_name} ({run_resp.dataset_size} items)")
    print("=" * 80)
    print("\n| Pipeline | P@5 | R@1 | R@3 | R@5 | MRR | nDCG@5 | Faithfulness | Citation OK | Refusal Acc | Avg Latency |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in run_resp.results:
        print(
            f"| {r.pipeline:<16} | {r.precisionAt5:.2f} | {r.recallAt1:.2f} | {r.recallAt3:.2f} | {r.recallAt5:.2f} | "
            f"{r.mrr:.3f} | {r.ndcgAt5:.3f} | {r.faithfulness:.2f} | {r.citationCorrectness:.2f} | "
            f"{r.refusalAccuracy:.2f} | {r.avgLatencyMs:.1f}ms |"
        )
    print("\n" + "=" * 80 + "\n")
    return run_resp


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run RAG benchmark evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of dataset items to evaluate")
    args = parser.parse_args()
    run_evaluation(limit=args.limit)
