import io
import time
import hashlib
import logging
from typing import Optional, Tuple
from fastapi import UploadFile, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import settings
from db.models import Document, Chunk
from ingestion.loaders import DocumentLoaderFactory
from ingestion.chunking import ChunkingStrategyFactory
from embeddings.embed import get_embedding_service
from app.services.cache_service import cache_service
from app.schemas.document import IngestResponse

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"pdf", "md", "markdown", "html", "htm", "txt"}
CHUNK_READ_SIZE = 64 * 1024  # 64 KB streaming chunks


class IngestionService:
    def __init__(self):
        self.embedding_service = get_embedding_service()

    async def read_and_validate_upload(
        self,
        file: UploadFile,
        max_size_mb: Optional[int] = None,
    ) -> Tuple[bytes, str]:
        """
        Reads uploaded file with bounded chunking to prevent memory exhaustion (DoS).
        Enforces size limits, extension check, and actual magic bytes/format validation.
        """
        limit_bytes = (max_size_mb or settings.MAX_UPLOAD_SIZE_MB) * 1024 * 1024
        filename = file.filename or "uploaded_file.txt"
        ext = filename.split(".")[-1].lower() if "." in filename else "txt"

        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file extension '.{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            )

        # Bounded streaming read
        buffer = io.BytesIO()
        total_read = 0

        while True:
            chunk = await file.read(CHUNK_READ_SIZE)
            if not chunk:
                break
            total_read += len(chunk)
            if total_read > limit_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds maximum allowed size of {max_size_mb or settings.MAX_UPLOAD_SIZE_MB} MB.",
                )
            buffer.write(chunk)

        content_bytes = buffer.getvalue()
        if not content_bytes or len(content_bytes.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        # Content/Magic bytes validation
        self._validate_content_format(content_bytes, ext, filename)

        return content_bytes, ext

    def _validate_content_format(self, content_bytes: bytes, ext: str, filename: str) -> None:
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file extension '.{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            )

        if ext == "pdf":

            # PDF magic bytes check
            if not content_bytes.startswith(b"%PDF-"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File '{filename}' has .pdf extension but lacks valid PDF magic header (%PDF-).",
                )
        elif ext in ("html", "htm"):
            head = content_bytes[:1024].lower()
            if not (b"<html" in head or b"<!doctype" in head or b"<body" in head or b"<div" in head or b"<p" in head):
                # Check if it's text
                try:
                    content_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"File '{filename}' contains invalid binary data for HTML format.",
                    )
        else:
            # Markdown and TXT must be valid UTF-8 and not binary
            if b"\x00" in content_bytes[:4096]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File '{filename}' contains binary null bytes and cannot be processed as text.",
                )
            try:
                content_bytes.decode("utf-8")
            except UnicodeDecodeError:
                # Try latin-1 fallback or raise
                try:
                    content_bytes.decode("latin-1")
                except Exception:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"File '{filename}' cannot be decoded as valid text.",
                    )

    def ingest_bytes(
        self,
        content_bytes: bytes,
        filename: str,
        content_type: Optional[str],
        chunking_strategy: str,
        chunk_size: int,
        chunk_overlap: int,
        user_id: str,
        db: Session,
    ) -> IngestResponse:
        start_time = time.time()
        file_hash = hashlib.sha256(content_bytes).hexdigest()

        # 1. Check for existing document with same hash
        existing_doc = db.query(Document).filter(Document.file_hash == file_hash).first()
        if existing_doc:
            db.delete(existing_doc)
            db.commit()

        # 2. Parse document
        loader = DocumentLoaderFactory.get_loader(filename, content_type)
        loaded_doc = loader.load(content_bytes, filename)

        # 3. Chunk
        chunker = ChunkingStrategyFactory.get_chunker(
            strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        raw_chunks = chunker.chunk(loaded_doc)

        if not raw_chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document contained no extractable text chunks.",
            )

        # 4. Create Document record
        new_doc = Document(
            filename=filename,
            content_type=content_type or loaded_doc.content_type,
            file_hash=file_hash,
            doc_metadata={**(loaded_doc.metadata or {}), "user_id": user_id},
        )
        db.add(new_doc)
        db.flush()

        # 5. Embed chunks
        chunk_texts = [c.content for c in raw_chunks]
        embeddings = self.embedding_service.embed_documents(chunk_texts)

        # 6. Save chunks
        db_chunks = []
        for c_data, emb in zip(raw_chunks, embeddings):
            db_chunk = Chunk(
                document_id=new_doc.id,
                content=c_data.content,
                chunk_index=c_data.chunk_index,
                page_number=c_data.page_number,
                start_char=c_data.start_char,
                end_char=c_data.end_char,
                chunking_strategy=chunking_strategy,
                embedding=emb,
                chunk_metadata={**(c_data.metadata or {}), "user_id": user_id},
            )
            db_chunks.append(db_chunk)

        db.add_all(db_chunks)
        db.commit()

        # Invalidate cache
        cache_service.invalidate_all()

        elapsed_ms = (time.time() - start_time) * 1000

        ext = filename.split(".")[-1].lower() if "." in filename else "txt"
        file_type = "markdown" if ext in ["md", "markdown"] else ("html" if ext in ["htm", "html"] else ("pdf" if ext == "pdf" else ext))

        return IngestResponse(
            document_id=new_doc.id,
            documentId=new_doc.id,
            filename=new_doc.filename,
            content_type=new_doc.content_type,
            fileType=file_type,
            chunk_count=len(db_chunks),
            chunkCount=len(db_chunks),
            chunking_strategy=chunking_strategy,
            chunkingStrategy=chunking_strategy,
            file_hash=file_hash,
            ingestion_time_ms=round(elapsed_ms, 2),
            status="ready",
            user_id=user_id,
        )


ingestion_service = IngestionService()
