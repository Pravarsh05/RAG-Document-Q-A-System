import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    Column,
    String,
    Integer,
    Text,
    DateTime,
    ForeignKey,
    JSON,
    Index,
)
from sqlalchemy.orm import relationship
from config import settings
from db.database import Base

try:
    from pgvector.sqlalchemy import Vector
    HAS_PGVECTOR = True
except ImportError:
    HAS_PGVECTOR = False
    Vector = None


def generate_uuid() -> str:
    return str(uuid.uuid4())


def get_utc_now():
    return datetime.now(timezone.utc)


class Document(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    filename = Column(String(255), nullable=False, index=True)
    content_type = Column(String(100), nullable=False)
    file_hash = Column(String(64), nullable=False, index=True)
    created_at = Column(DateTime, default=get_utc_now, nullable=False)
    doc_metadata = Column(JSON, default=dict, nullable=False)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "filename": self.filename,
            "content_type": self.content_type,
            "file_hash": self.file_hash,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "doc_metadata": self.doc_metadata,
            "chunk_count": len(self.chunks) if self.chunks else 0,
        }


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=True)
    start_char = Column(Integer, nullable=True)
    end_char = Column(Integer, nullable=True)
    chunking_strategy = Column(String(50), nullable=False, default="sentence")
    
    # Vector column if pgvector available, else JSON for embedding vector storage in SQLite tests
    if HAS_PGVECTOR and Vector is not None:
        embedding = Column(Vector(settings.EMBEDDING_DIMENSION), nullable=True)
    else:
        embedding = Column(JSON, nullable=True)
        
    chunk_metadata = Column(JSON, default=dict, nullable=False)

    document = relationship("Document", back_populates="chunks")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "filename": self.document.filename if self.document else None,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "chunking_strategy": self.chunking_strategy,
            "chunk_metadata": self.chunk_metadata,
        }
