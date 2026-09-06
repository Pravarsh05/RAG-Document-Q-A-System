import logging
from typing import List, Optional
import numpy as np
from config import settings
from retrieval.vector_search import SearchResult

logger = logging.getLogger(__name__)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


class BaseReranker:
    def rerank(self, query: str, results: List[SearchResult], top_k: Optional[int] = None) -> List[SearchResult]:
        raise NotImplementedError


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading CrossEncoder: {self.model_name} on {self.device}")
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(self, query: str, results: List[SearchResult], top_k: Optional[int] = None) -> List[SearchResult]:
        if not results:
            return []

        pairs = [[query, r.content] for r in results]
        try:
            raw_scores = self.model.predict(pairs, show_progress_bar=False)
            
            reranked: List[SearchResult] = []
            for r, score in zip(results, raw_scores):
                # Apply sigmoid normalization to raw logits
                norm_score = float(sigmoid(float(score)))
                reranked.append(
                    SearchResult(
                        chunk_id=r.chunk_id,
                        document_id=r.document_id,
                        filename=r.filename,
                        content=r.content,
                        chunk_index=r.chunk_index,
                        page_number=r.page_number,
                        metadata={**r.metadata, "initial_score": r.score, "rerank_raw": float(score)},
                        score=norm_score,
                    )
                )

            reranked.sort(key=lambda x: x.score, reverse=True)
            k = top_k if top_k is not None else len(reranked)
            return reranked[:k]

        except Exception as e:
            logger.warning(f"CrossEncoder prediction failed ({e}). Returning original ordering.")
            k = top_k if top_k is not None else len(results)
            return results[:k]


class MockReranker(BaseReranker):
    """Simple term-overlap reranker for lightweight mock/testing mode."""

    def rerank(self, query: str, results: List[SearchResult], top_k: Optional[int] = None) -> List[SearchResult]:
        if not results:
            return []

        query_tokens = set(query.lower().split())
        reranked = []
        for r in results:
            content_tokens = set(r.content.lower().split())
            overlap = len(query_tokens.intersection(content_tokens)) / max(1, len(query_tokens))
            combined_score = 0.5 * r.score + 0.5 * overlap
            reranked.append(
                SearchResult(
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    filename=r.filename,
                    content=r.content,
                    chunk_index=r.chunk_index,
                    page_number=r.page_number,
                    metadata={**r.metadata, "mock_reranked": True},
                    score=float(combined_score),
                )
            )

        reranked.sort(key=lambda x: x.score, reverse=True)
        k = top_k if top_k is not None else len(reranked)
        return reranked[:k]


def get_reranker() -> BaseReranker:
    try:
        return CrossEncoderReranker(
            model_name=settings.RERANKER_MODEL,
            device=settings.RERANKER_DEVICE,
        )
    except Exception as e:
        logger.warning(f"Could not load CrossEncoder ({e}). Using MockReranker.")
        return MockReranker()
