import time
import logging
from typing import Optional, Dict, Any, List
from collections import Counter
from sqlalchemy.orm import Session

from app.core.config import settings
from retrieval.hybrid_search import HybridSearchService, RetrievalPipeline
from retrieval.vector_search import SearchResult
from generation.generate import GenerationService
from embeddings.embed import get_embedding_service
from app.services.cache_service import cache_service
from app.services.query_rewriter import query_rewriter
from app.services.citation_verifier import citation_verifier
from app.schemas.query import QueryRequest, QueryResponse, SourceChunkResponse, CitationResponse, ClaimVerificationItem

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.retriever = HybridSearchService(embedding_service=self.embedding_service)
        self.generator = GenerationService()

    def execute_query(
        self,
        request: QueryRequest,
        db: Session,
        user_id: str = "default_user",
    ) -> QueryResponse:
        total_start = time.time()
        effective_pipeline = request.get_pipeline()

        # 1. Structured Cache Lookup
        cache_key = cache_service.make_cache_key(
            query=request.question,
            pipeline=effective_pipeline,
            top_k=request.top_k,
            candidate_k=request.candidate_k,
            filter_document_id=request.filter_document_id,
            llm_model=settings.LLM_MODEL,
            embedding_model=settings.EMBEDDING_MODEL,
        )
        cached_result = cache_service.get_query(cache_key)
        if cached_result:
            cached_result["cached"] = True
            return QueryResponse(**cached_result)

        # 2. Query Rewriting (Standout Feature 1)
        do_rewriting = (
            request.enable_query_rewriting
            if request.enable_query_rewriting is not None
            else settings.QUERY_REWRITING_ENABLED
        )
        rewritten_q, expansions = (
            query_rewriter.rewrite(request.question)
            if do_rewriting
            else (request.question, [])
        )
        retrieval_query = rewritten_q if expansions else request.question

        # 3. Retrieval & Re-ranking
        retrieval_start = time.time()
        retrieved_results: List[SearchResult] = self.retriever.search(
            db=db,
            query=retrieval_query,
            pipeline=effective_pipeline,
            top_k=request.top_k,
            candidate_k=request.candidate_k,
            filter_document_id=request.filter_document_id,
        )
        retrieval_time_ms = (time.time() - retrieval_start) * 1000

        # 4. Relevance Thresholding & Insufficient Evidence Guardrail
        # If top score is below threshold, refuse rather than hallucinate
        top_score = retrieved_results[0].score if retrieved_results else 0.0
        insufficient_evidence = (
            not retrieved_results
            or (top_score < settings.RELEVANCE_THRESHOLD and effective_pipeline != "bm25_only")
        )

        gen_start = time.time()
        if insufficient_evidence:
            answer = "I cannot answer this question based on the provided documents as no sufficiently relevant context was found."
            citations = []
            gen_latency_ms = (time.time() - gen_start) * 1000
            model_name = settings.LLM_MODEL
            grounding_status = "insufficient_evidence"
            claim_verifications = []
        else:
            # 5. LLM Answer Synthesis with Strict Citations
            gen_result = self.generator.generate_answer(
                question=request.question,
                retrieved_chunks=retrieved_results,
            )
            answer = gen_result.answer
            citations = gen_result.citations
            gen_latency_ms = gen_result.generation_latency_ms
            model_name = gen_result.model_name

            # 6. Claim-Evidence Citation Verification (Standout Feature 3)
            do_verification = (
                request.enable_citation_verification
                if request.enable_citation_verification is not None
                else settings.CITATION_VERIFICATION_ENABLED
            )
            if do_verification:
                v_res = citation_verifier.verify(answer=answer, retrieved_chunks=retrieved_results)
                grounding_status = v_res["grounding_status"]
                claim_verifications = [ClaimVerificationItem(**c) for c in v_res["claim_verifications"]]
            else:
                grounding_status = "grounded" if citations else "insufficient_evidence"
                claim_verifications = []

        total_latency_ms = (time.time() - total_start) * 1000

        # 7. Multi-Document Provenance Tracking (Standout Feature 2)
        doc_provenance = Counter(r.filename for r in retrieved_results if r.filename)

        # 8. Build formatted chunk and citation objects
        formatted_chunks: List[SourceChunkResponse] = []
        chunk_by_id = {}
        for rank, r in enumerate(retrieved_results, start=1):
            sig = r.metadata.get("signal", "vector" if effective_pipeline == "vector_only" else ("bm25" if effective_pipeline == "bm25_only" else "hybrid"))
            c_resp = SourceChunkResponse(
                id=r.chunk_id,
                chunk_id=r.chunk_id,
                documentId=r.document_id,
                document_id=r.document_id,
                documentName=r.filename,
                filename=r.filename,
                page=r.page_number,
                page_number=r.page_number,
                position=r.chunk_index,
                chunk_index=r.chunk_index,
                text=r.content,
                content=r.content,
                rank=rank,
                signal=sig,
                score=round(float(r.score), 4),
                metadata=r.metadata,
            )
            formatted_chunks.append(c_resp)
            chunk_by_id[r.chunk_id] = c_resp

        formatted_citations: List[CitationResponse] = []
        for c in citations:
            matched = chunk_by_id.get(c.chunk_id)
            if matched:
                formatted_citations.append(
                    CitationResponse(
                        id=matched.id,
                        chunk_id=matched.chunk_id,
                        documentId=matched.documentId,
                        document_id=matched.document_id,
                        documentName=matched.documentName,
                        filename=matched.filename,
                        page=matched.page,
                        page_number=matched.page_number,
                        position=matched.position,
                        chunk_index=matched.chunk_index,
                        text=matched.text,
                        snippet=c.snippet or (matched.text[:150] + "..."),
                        rank=matched.rank,
                        signal=matched.signal,
                        score=matched.score,
                        is_verified=True,
                        support_score=1.0,
                    )
                )

        frontend_mode = (
            "hybrid_rerank"
            if effective_pipeline == "hybrid_rerank"
            else ("hybrid" if effective_pipeline == "hybrid" else "vector")
        )

        response_payload = {
            "question": request.question,
            "answer": answer,
            "pipeline": effective_pipeline,
            "retrievalMode": frontend_mode,
            "retrieval_mode": frontend_mode,
            "citations": [c.model_dump() for c in formatted_citations],
            "retrieved_chunks": [c.model_dump() for c in formatted_chunks],
            "retrievedChunks": [c.model_dump() for c in formatted_chunks],
            "latency_ms": {
                "retrieval_ms": round(retrieval_time_ms, 2),
                "generation_ms": round(gen_latency_ms, 2),
                "total_ms": round(total_latency_ms, 2),
            },
            "latencyMs": round(total_latency_ms, 2),
            "model_name": model_name,
            "modelName": model_name,
            "cached": False,
            "grounding_status": grounding_status,
            "groundingStatus": grounding_status,
            "rewritten_query": rewritten_q if expansions else None,
            "claim_verifications": [c.model_dump() for c in claim_verifications],
            "multi_doc_provenance": dict(doc_provenance),
        }

        # Cache response
        cache_service.set_query(cache_key, response_payload)

        return QueryResponse(**response_payload)


rag_service = RAGService()
