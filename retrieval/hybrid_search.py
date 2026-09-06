import logging
from typing import List, Dict, Optional, Literal
from sqlalchemy.orm import Session
from retrieval.vector_search import VectorSearchService, SearchResult
from retrieval.keyword_search import BM25KeywordSearchService
from retrieval.rerank import BaseReranker, get_reranker
from embeddings.embed import BaseEmbeddingService, get_embedding_service

logger = logging.getLogger(__name__)

RetrievalPipeline = Literal["vector_only", "bm25_only", "hybrid", "hybrid_rerank"]


class HybridSearchService:
    def __init__(
        self,
        embedding_service: Optional[BaseEmbeddingService] = None,
        reranker: Optional[BaseReranker] = None,
        rrf_k: int = 60,
    ):
        self.embedding_service = embedding_service or get_embedding_service()
        self.vector_search = VectorSearchService(self.embedding_service)
        self.bm25_search = BM25KeywordSearchService()
        self.reranker = reranker or get_reranker()
        self.rrf_k = rrf_k

    def reciprocal_rank_fusion(
        self,
        ranked_lists: List[List[SearchResult]],
        top_k: int = 10,
    ) -> List[SearchResult]:
        """
        Combines multiple ranked lists using Reciprocal Rank Fusion (RRF).
        RRF Score = sum(1 / (k + rank))
        """
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, SearchResult] = {}
        chunk_signals: Dict[str, set] = {}

        for r_list in ranked_lists:
            for rank, item in enumerate(r_list, start=1):
                chunk_id = item.chunk_id
                chunk_map[chunk_id] = item
                score_increment = 1.0 / (self.rrf_k + rank)
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + score_increment
                
                sig = item.metadata.get("signal", "vector")
                if chunk_id not in chunk_signals:
                    chunk_signals[chunk_id] = set()
                chunk_signals[chunk_id].add(sig)

        fused_results: List[SearchResult] = []
        for chunk_id, score in rrf_scores.items():
            base_item = chunk_map[chunk_id]
            signals = chunk_signals.get(chunk_id, set())
            if len(signals) > 1:
                resolved_signal = "hybrid"
            elif "bm25" in signals:
                resolved_signal = "bm25"
            else:
                resolved_signal = "vector"

            fused_results.append(
                SearchResult(
                    chunk_id=base_item.chunk_id,
                    document_id=base_item.document_id,
                    filename=base_item.filename,
                    content=base_item.content,
                    chunk_index=base_item.chunk_index,
                    page_number=base_item.page_number,
                    metadata={**base_item.metadata, "rrf_score": score, "signal": resolved_signal},
                    score=score,
                )
            )

        fused_results.sort(key=lambda x: x.score, reverse=True)
        return fused_results[:top_k]

    def search(
        self,
        db: Session,
        query: str,
        pipeline: RetrievalPipeline = "hybrid_rerank",
        top_k: int = 5,
        candidate_k: int = 20,
        filter_document_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Executes search based on the chosen pipeline:
        - 'vector_only': Pure dense vector similarity
        - 'bm25_only': Pure sparse BM25 keyword matching
        - 'hybrid': Reciprocal Rank Fusion of Vector + BM25
        - 'hybrid_rerank': Hybrid search candidates re-ranked by Cross-Encoder
        """
        if pipeline == "vector_only":
            return self.vector_search.search(
                db=db,
                query=query,
                top_k=top_k,
                filter_document_id=filter_document_id,
            )

        if pipeline == "bm25_only":
            return self.bm25_search.search(
                db=db,
                query=query,
                top_k=top_k,
                filter_document_id=filter_document_id,
            )

        # Retrieve candidates for hybrid search
        k_candidates = max(candidate_k, top_k * 2)
        vector_results = self.vector_search.search(
            db=db,
            query=query,
            top_k=k_candidates,
            filter_document_id=filter_document_id,
        )
        bm25_results = self.bm25_search.search(
            db=db,
            query=query,
            top_k=k_candidates,
            filter_document_id=filter_document_id,
        )

        hybrid_candidates = self.reciprocal_rank_fusion(
            [vector_results, bm25_results],
            top_k=k_candidates,
        )

        if pipeline == "hybrid":
            return hybrid_candidates[:top_k]

        if pipeline == "hybrid_rerank":
            # Pass hybrid candidates through cross-encoder re-ranker
            reranked = self.reranker.rerank(
                query=query,
                results=hybrid_candidates,
                top_k=top_k,
            )
            return reranked

        # Fallback to hybrid
        return hybrid_candidates[:top_k]
