import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Query, HTTPException, status
from pydantic import BaseModel

from app.services.evaluation_service import evaluation_service
from app.schemas.evaluation import EvalRow, BenchmarkRunResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Evaluation"])


class EvalTriggerRequest(BaseModel):
    limit: Optional[int] = None
    pipelines: Optional[List[str]] = None


@router.get("/eval/results", response_model=List[EvalRow])
def get_eval_results():
    """
    Returns live metrics from the most recent actual benchmark run.
    If no evaluation has run yet, returns an empty list [].
    Does NOT return hardcoded or fabricated defaults.
    """
    latest = evaluation_service.get_latest_results()
    if not latest or not latest.results:
        return []
    return latest.results


@router.get("/eval/status", response_model=Dict[str, Any])
def get_eval_status():
    """Checks whether an empirical evaluation benchmark has been executed."""
    latest = evaluation_service.get_latest_results()
    if not latest:
        return {
            "has_run": False,
            "message": "No evaluation has been executed yet. Click 'Run Live Evaluation' to benchmark the system.",
            "latest_run": None,
        }
    return {
        "has_run": True,
        "run_id": latest.run_id,
        "timestamp": latest.timestamp,
        "dataset_size": latest.dataset_size,
        "latest_run": latest.model_dump(),
    }


@router.post("/eval/run", response_model=List[EvalRow])
def trigger_eval_run(req: Optional[EvalTriggerRequest] = None):
    """
    Triggers an evaluation benchmark run across ground-truth test items,
    computes real metrics (Recall@1/3/5, Precision@5, MRR, nDCG@5, Faithfulness,
    Citation Correctness, Refusal Accuracy), persists the results, and returns them.
    """
    try:
        limit = req.limit if req else None
        pipelines = req.pipelines if req else None
        benchmark_response = evaluation_service.run_benchmark(limit=limit, pipelines=pipelines)
        return benchmark_response.results
    except Exception as e:
        logger.error(f"Live evaluation benchmark failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation execution failed: {str(e)}",
        )
