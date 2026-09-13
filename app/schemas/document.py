from typing import Optional, Dict, Any, List
from pydantic import BaseModel


class IngestResponse(BaseModel):
    document_id: str
    documentId: str
    filename: str
    content_type: str
    fileType: str
    chunk_count: int
    chunkCount: int
    chunking_strategy: str
    chunkingStrategy: str
    file_hash: str
    ingestion_time_ms: float
    status: str = "ready"
    user_id: str = "default_user"


class DocumentResponse(BaseModel):
    id: str
    filename: str
    fileType: str
    content_type: str
    chunkCount: int
    chunk_count: int
    chunkingStrategy: str
    chunking_strategy: str
    ingestedAt: str
    created_at: str
    file_hash: str
    status: str = "ready"
    doc_metadata: Dict[str, Any] = {}
    user_id: Optional[str] = None


class ChunkResponse(BaseModel):
    id: str
    document_id: str
    content: str
    chunk_index: int
    page_number: Optional[int] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    chunking_strategy: str
    chunk_metadata: Dict[str, Any] = {}
