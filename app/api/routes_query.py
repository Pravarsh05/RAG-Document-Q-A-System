import logging
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from db.database import get_db
from app.core.security import get_current_user_id
from app.core.rate_limit import rate_limit_dependency
from app.services.rag_service import rag_service
from app.schemas.query import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Query"])


@router.post("/query", response_model=QueryResponse)
def query_rag(
    request: QueryRequest,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
    _rate_limit = Depends(rate_limit_dependency),
):
    """
    Executes grounded RAG search & synthesis:
    - Multi-stage retrieval: Vector, BM25, Hybrid (RRF), Cross-Encoder Re-ranking
    - Query rewriting and expansion
    - Insufficient evidence & relevance threshold guardrails
    - LLM generation with strict chunk citations
    - Claim-evidence citation verification
    - Distributed query caching with index version invalidation
    """
    return rag_service.execute_query(request=request, db=db, user_id=user_id)
