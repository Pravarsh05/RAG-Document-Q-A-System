import logging
from fastapi import APIRouter, UploadFile, File, Form, Depends
from sqlalchemy.orm import Session

from app.core.config import settings
from db.database import get_db
from app.core.security import get_current_user_id
from app.core.rate_limit import rate_limit_dependency
from app.services.ingestion_service import ingestion_service
from app.schemas.document import IngestResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Ingestion"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    chunking_strategy: str = Form(settings.DEFAULT_CHUNKING_STRATEGY),
    chunk_size: int = Form(settings.DEFAULT_CHUNK_SIZE),
    chunk_overlap: int = Form(settings.DEFAULT_CHUNK_OVERLAP),
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
    _rate_limit = Depends(rate_limit_dependency),
):
    """
    Ingests and indexes enterprise documents:
    - Bounded streaming read (memory exhaustion protection)
    - Magic bytes format & extension validation
    - Multi-strategy chunking (sentence, semantic, fixed)
    - Dense vector embedding generation
    - Automatic cache invalidation
    """
    content_bytes, ext = await ingestion_service.read_and_validate_upload(file)

    return ingestion_service.ingest_bytes(
        content_bytes=content_bytes,
        filename=file.filename or "uploaded_file.txt",
        content_type=file.content_type,
        chunking_strategy=chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        user_id=user_id,
        db=db,
    )
