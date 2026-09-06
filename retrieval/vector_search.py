import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import select, text
from db.models import Chunk, Document
from embeddings.embed import BaseEmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    chunk_id: str
    document_id: str
    filename: str
    content: str
    score: float
    page_number: Optional[int] = None
    chunk_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "content": self.content,
            "score": round(self.score, 4),
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }


class VectorSearchService:
    def __init__(self, embedding_service: BaseEmbeddingService):
        self.embedding_service = embedding_service

    def search(
        self,
        db: Session,
        query: str,
        top_k: int = 10,
        filter_document_id: Optional[str] = None,
    ) -> List[SearchResult]:
        query_vector = self.embedding_service.embed_query(query)
        return self.search_by_vector(db, query_vector, top_k, filter_document_id)

    def search_by_vector(
        self,
        db: Session,
        query_vector: List[float],
        top_k: int = 10,
        filter_document_id: Optional[str] = None,
    ) -> List[SearchResult]:
        bind = db.get_bind()
        is_postgres = bind.dialect.name == "postgresql"

        if is_postgres:
            try:
                # Use pgvector cosine distance: 1 - cosine_distance = cosine similarity
                filter_clause = "WHERE c.document_id = :doc_id" if filter_document_id else ""
                sql = f"""
                    SELECT 
                        c.id, c.document_id, c.content, c.chunk_index, c.page_number, 
                        c.chunk_metadata, d.filename,
                        1 - (c.embedding <=> :vector) AS similarity
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    {filter_clause}
                    ORDER BY c.embedding <=> :vector ASC
                    LIMIT :top_k
                """
                params = {"vector": str(query_vector), "top_k": top_k}
                if filter_document_id:
                    params["doc_id"] = filter_document_id

                results = db.execute(text(sql), params).fetchall()

                return [
                    SearchResult(
                        chunk_id=row[0],
                        document_id=row[1],
                        content=row[2],
                        chunk_index=row[3],
                        page_number=row[4],
                        metadata={**(row[5] or {}), "signal": "vector"},
                        filename=row[6],
                        score=float(row[7]) if row[7] is not None else 0.0,
                    )
                    for row in results
                ]
            except Exception as e:
                logger.warning(f"Postgres pgvector query failed ({e}), falling back to in-memory vector similarity.")

        # In-memory numpy fallback (works for SQLite / fallback)
        chunks = db.query(Chunk).join(Document).all()
        if filter_document_id:
            chunks = [c for c in chunks if c.document_id == filter_document_id]

        if not chunks:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        scored: List[SearchResult] = []
        for c in chunks:
            if c.embedding is None:
                continue
            c_vec = np.array(c.embedding, dtype=np.float32)
            c_norm = np.linalg.norm(c_vec)
            sim = float(np.dot(q_vec, c_vec) / (c_norm + 1e-9)) if c_norm > 0 else 0.0

            scored.append(
                SearchResult(
                    chunk_id=c.id,
                    document_id=c.document_id,
                    filename=c.document.filename if c.document else "unknown",
                    content=c.content,
                    chunk_index=c.chunk_index,
                    page_number=c.page_number,
                    metadata={**(c.chunk_metadata or {}), "signal": "vector"},
                    score=sim,
                )
            )

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]
