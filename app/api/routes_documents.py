import logging
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import settings
from db.database import get_db
from db.models import Document, Chunk, get_utc_now
from app.core.security import get_current_user_id
from app.services.cache_service import cache_service
from app.schemas.document import DocumentResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Documents"])


def format_doc_model(doc: Document) -> Dict[str, Any]:
    ext = doc.filename.split(".")[-1].lower() if "." in doc.filename else "txt"
    if ext in ["md", "markdown"]:
        file_type = "markdown"
    elif ext in ["htm", "html"]:
        file_type = "html"
    elif ext == "pdf":
        file_type = "pdf"
    else:
        file_type = ext

    strategy = doc.chunks[0].chunking_strategy if doc.chunks else settings.DEFAULT_CHUNKING_STRATEGY
    created_iso = doc.created_at.isoformat() if doc.created_at else get_utc_now().isoformat()
    count = len(doc.chunks) if doc.chunks else 0
    doc_meta = doc.doc_metadata or {}
    user_id = doc_meta.get("user_id")

    return {
        "id": doc.id,
        "filename": doc.filename,
        "fileType": file_type,
        "content_type": doc.content_type,
        "chunkCount": count,
        "chunk_count": count,
        "chunkingStrategy": strategy,
        "chunking_strategy": strategy,
        "ingestedAt": created_iso,
        "created_at": created_iso,
        "file_hash": doc.file_hash,
        "status": "ready",
        "doc_metadata": doc_meta,
        "user_id": user_id,
    }


@router.get("/documents", response_model=List[DocumentResponse])
def list_documents(
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    docs = db.query(Document).order_by(Document.created_at.desc()).all()
    # If user_id is specified and not default_user, filter docs
    if user_id != "default_user":
        docs = [d for d in docs if (d.doc_metadata or {}).get("user_id") == user_id]
    return [format_doc_model(d) for d in docs]


@router.get("/documents/{document_id}/chunks")
def get_document_chunks(
    document_id: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    chunks = db.query(Chunk).filter(Chunk.document_id == document_id).order_by(Chunk.chunk_index.asc()).all()
    return [c.to_dict() for c in chunks]


@router.delete("/documents/{document_id}")
def delete_document(
    document_id: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    db.delete(doc)
    db.commit()

    # Invalidate cache
    cache_service.invalidate_all()

    return {"message": "Document deleted successfully", "document_id": document_id}
