import re
import logging
from typing import List, Optional
from sqlalchemy.orm import Session
from rank_bm25 import BM25Okapi
from db.models import Chunk, Document
from retrieval.vector_search import SearchResult

logger = logging.getLogger(__name__)


class BM25KeywordSearchService:
    def __init__(self):
        self._bm25 = None
        self._chunk_ids = []
        self._corpus_version = None

    def _tokenize(self, text: str) -> List[str]:
        # Clean text and lower case alphanumeric tokens
        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens

    def search(
        self,
        db: Session,
        query: str,
        top_k: int = 10,
        filter_document_id: Optional[str] = None,
    ) -> List[SearchResult]:
        chunks = db.query(Chunk).join(Document).all()
        if filter_document_id:
            chunks = [c for c in chunks if c.document_id == filter_document_id]

        if not chunks:
            return []

        # Tokenize corpus and query
        tokenized_corpus = [self._tokenize(c.content) for c in chunks]
        tokenized_query = self._tokenize(query)

        if not tokenized_query:
            return []

        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenized_query)

        # Normalize BM25 scores between 0 and 1 using max score
        max_score = float(max(scores)) if (len(scores) > 0 and max(scores) > 0) else 0.0

        results: List[SearchResult] = []
        for c, score, tokens in zip(chunks, scores, tokenized_corpus):

            # Check direct token overlap for small corpus robustness
            overlap_count = sum(1 for q in tokenized_query if q in tokens)
            effective_score = float(score / max_score) if max_score > 0 else (0.5 if overlap_count > 0 else 0.0)
            if score > 0 or overlap_count > 0:
                results.append(
                    SearchResult(
                        chunk_id=c.id,
                        document_id=c.document_id,
                        filename=c.document.filename if c.document else "unknown",
                        content=c.content,
                        chunk_index=c.chunk_index,
                        page_number=c.page_number,
                        metadata={**(c.chunk_metadata or {}), "signal": "bm25"},
                        score=effective_score if effective_score > 0 else 0.5,
                    )
                )


        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
