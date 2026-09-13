import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from retrieval.vector_search import SearchResult

logger = logging.getLogger(__name__)

REFUSAL_PHRASES = [
    "cannot answer this question based on the provided documents",
    "cannot answer this question based on the provided context",
    "not found in provided documents",
    "not found in the provided documents",
    "insufficient information in the provided",
    "no relevant context found",
    "provided documents do not contain",
    "provided context does not contain",
]


class CitationVerifier:
    """
    Evaluates factual grounding and citation fidelity:
    - Verifies whether each cited claim is actually supported by the cited chunk.
    - Flags unsupported claims and citation hallucinations.
    - Computes citation correctness, unsupported-claim rate, and grounding status.
    """

    STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
        "by", "about", "against", "between", "into", "through", "during", "before",
        "after", "above", "below", "from", "up", "down", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did", "this",
        "that", "these", "those", "it", "its", "they", "them", "their", "here", "there"
    }

    def is_refusal_answer(self, answer: str) -> bool:
        norm = answer.lower()
        return any(phrase in norm for phrase in REFUSAL_PHRASES)

    def extract_claims_with_citations(self, answer: str) -> List[Dict[str, Any]]:
        """
        Splits text into bullet points or sentences and extracts associated chunk citations.
        """
        lines = [l.strip() for l in answer.split("\n") if l.strip()]
        claims = []

        for line in lines:
            # Skip pure markdown headers
            if re.match(r"^#{1,6}\s+", line) and not re.search(r"\[\d+\]", line):
                continue

            # Split paragraphs into sentences if not bullet point
            if line.startswith(("-", "*", "•", "1.", "2.", "3.", "4.", "5.")):
                sentence_list = [line]
            else:
                sentence_list = re.split(r"(?<=[.!?])\s+", line)

            for s in sentence_list:
                clean_s = s.strip()
                if len(clean_s) < 15:
                    continue

                # Extract bracketed numbers like [1], [2], or [Chunk <id>]
                num_cites = [int(m) for m in re.findall(r"\[(\d+)\]", clean_s)]
                named_cites = re.findall(r"\[Chunk\s+([^\]]+)\]", clean_s, flags=re.IGNORECASE)

                # Clean claim text of citation tags for verification
                claim_text = re.sub(r"\[(?:\d+|Chunk\s+[^\]]+)\]", "", clean_s).strip()
                claim_text = re.sub(r"^[-*•\d.]+\s*", "", claim_text).strip()

                claims.append({
                    "raw": clean_s,
                    "claim": claim_text,
                    "numeric_cites": num_cites,
                    "named_cites": named_cites,
                })

        return claims

    def verify(
        self,
        answer: str,
        retrieved_chunks: List[SearchResult],
        min_support_score: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Verifies all claims against the retrieved chunks.
        Returns:
            - claim_verifications: List of verification results per claim
            - citation_correctness: float in [0.0, 1.0]
            - unsupported_claim_rate: float in [0.0, 1.0]
            - grounding_status: "grounded" | "citation_mismatch" | "insufficient_evidence" | "refusal"
        """
        if not answer.strip():
            return {
                "claim_verifications": [],
                "citation_correctness": 0.0,
                "unsupported_claim_rate": 1.0,
                "grounding_status": "insufficient_evidence",
            }

        if self.is_refusal_answer(answer):
            return {
                "claim_verifications": [],
                "citation_correctness": 1.0,
                "unsupported_claim_rate": 0.0,
                "grounding_status": "refusal",
            }

        if not retrieved_chunks:
            return {
                "claim_verifications": [],
                "citation_correctness": 0.0,
                "unsupported_claim_rate": 1.0,
                "grounding_status": "insufficient_evidence",
            }

        # Index chunks by 1-based index and by chunk_id
        index_map = {idx: c for idx, c in enumerate(retrieved_chunks, start=1)}
        chunk_map = {c.chunk_id: c for c in retrieved_chunks}

        claims = self.extract_claims_with_citations(answer)
        if not claims:
            # Fallback for single short answer
            claims = [{
                "raw": answer,
                "claim": answer,
                "numeric_cites": [int(m) for m in re.findall(r"\[(\d+)\]", answer)],
                "named_cites": re.findall(r"\[Chunk\s+([^\]]+)\]", answer, flags=re.IGNORECASE),
            }]

        claim_results = []
        supported_citations_count = 0
        total_citations_count = 0
        unsupported_claims_count = 0

        for item in claims:
            claim_text = item["claim"]
            claim_words = set(re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", claim_text.lower())) - self.STOP_WORDS

            # Resolve cited chunks for this claim
            cited_chunks: List[SearchResult] = []
            for num in item["numeric_cites"]:
                if num in index_map:
                    cited_chunks.append(index_map[num])
                    total_citations_count += 1
                else:
                    total_citations_count += 1  # Cited an out-of-range chunk number

            for cid in item["named_cites"]:
                clean_id = cid.strip()
                if clean_id.isdigit() and int(clean_id) in index_map:
                    cited_chunks.append(index_map[int(clean_id)])
                    total_citations_count += 1
                elif clean_id in chunk_map:
                    cited_chunks.append(chunk_map[clean_id])
                    total_citations_count += 1
                else:
                    total_citations_count += 1  # Cited unknown UUID

            if not cited_chunks:
                # Claim without any valid citation
                unsupported_claims_count += 1
                claim_results.append({
                    "claim": claim_text,
                    "cited_chunk_id": None,
                    "is_supported": False,
                    "confidence": 0.0,
                    "reason": "Claim has no citation attribution.",
                })
                continue

            # Verify against the cited chunks
            best_overlap = 0.0
            best_chunk_id = None
            for c in cited_chunks:
                chunk_words = set(re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", c.content.lower()))
                overlap = len(claim_words.intersection(chunk_words)) / max(1, len(claim_words))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_chunk_id = c.chunk_id

            is_supported = best_overlap >= min_support_score
            if is_supported:
                supported_citations_count += len(cited_chunks)
                reason = f"Claim supported by evidence (lexical overlap: {best_overlap:.2f})"
            else:
                unsupported_claims_count += 1
                reason = f"Cited chunk does not substantiate claim (support score: {best_overlap:.2f} < {min_support_score})"

            claim_results.append({
                "claim": claim_text,
                "cited_chunk_id": best_chunk_id,
                "is_supported": is_supported,
                "confidence": round(float(best_overlap), 3),
                "reason": reason,
            })

        citation_correctness = (
            supported_citations_count / total_citations_count if total_citations_count > 0 else 0.0
        )
        unsupported_claim_rate = (
            unsupported_claims_count / len(claims) if claims else 1.0
        )

        if total_citations_count == 0:
            status = "insufficient_evidence"
        elif unsupported_claim_rate == 0.0 and citation_correctness >= 0.80:
            status = "grounded"
        else:
            status = "citation_mismatch"

        return {
            "claim_verifications": claim_results,
            "citation_correctness": round(citation_correctness, 3),
            "unsupported_claim_rate": round(unsupported_claim_rate, 3),
            "grounding_status": status,
        }


citation_verifier = CitationVerifier()
