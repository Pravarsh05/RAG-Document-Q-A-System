import re
import logging
from typing import Tuple, List, Dict

logger = logging.getLogger(__name__)

CONVERSATIONAL_PREFIXES = [
    r"^(?:could|can)\s+you\s+(?:please\s+)?(?:tell|explain|show)\s+(?:me\s+)?(?:about\s+)?",
    r"^(?:i\s+(?:want|would\s+like)\s+to\s+know\s+)",
    r"^(?:i\s+was\s+wondering\s+(?:if|how|what)\s+)",
    r"^(?:what\s+(?:exactly\s+)?is\s+)",
    r"^(?:what\s+are\s+)",
    r"^(?:how\s+does\s+)",
    r"^(?:why\s+does\s+)",
    r"^(?:tell\s+me\s+(?:about\s+)?)",
    r"^(?:give\s+me\s+(?:a\s+summary\s+of\s+)?)",
    r"^(?:please\s+describe\s+)",
]

TECHNICAL_EXPANSIONS: Dict[str, List[str]] = {
    "pgvector": ["postgres", "vector", "similarity", "hnsw", "index"],
    "hnsw": ["hierarchical", "navigable", "small", "world", "approximate", "neighbors"],
    "rrf": ["reciprocal", "rank", "fusion", "rankings", "score"],
    "cross-encoder": ["cross", "encoder", "reranking", "passage", "ms-marco", "scores"],
    "bm25": ["bm25", "sparse", "lexical", "keyword", "frequency", "idf"],
    "chunking": ["chunk", "strategy", "sentence", "semantic", "overlap", "split"],
    "citations": ["grounded", "source", "chunks", "attribution", "evidence"],
    "hybrid": ["hybrid", "dense", "sparse", "vector", "bm25", "fusion"],
    "embedding": ["embeddings", "dense", "bge", "similarity", "dimension"],
    "latency": ["latency", "runtime", "speed", "ms", "performance", "throughput"],
    "cache": ["redis", "ttl", "invalidation", "hit", "miss"],
}


class QueryRewriter:
    """
    Transforms conversational, ambiguous, or keyword-sparse user queries
    into retrieval-optimized queries that maximize dense & sparse Recall@K.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def clean_conversational_noise(self, query: str) -> str:
        cleaned = query.strip()
        for pattern in CONVERSATIONAL_PREFIXES:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
        # Clean trailing question marks or punctuation
        cleaned = re.sub(r"[?!.]+$", "", cleaned).strip()
        return cleaned if cleaned else query.strip()

    def extract_expansions(self, text: str) -> List[str]:
        words = set(re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text.lower()))
        expansions = []
        for key, exp_list in TECHNICAL_EXPANSIONS.items():
            if key in words or any(k in words for k in key.split("-")):
                for term in exp_list:
                    if term not in words and term not in expansions:
                        expansions.append(term)
        return expansions[:5]

    def rewrite(self, query: str) -> Tuple[str, List[str]]:
        if not self.enabled or not query:
            return query, []

        core_query = self.clean_conversational_noise(query)
        expansions = self.extract_expansions(query)

        if expansions:
            # Build search-optimized query: core query plus high-relevance semantic tags
            rewritten = f"{core_query} ({' '.join(expansions)})"
        else:
            rewritten = core_query

        logger.debug(f"Rewrote query '{query}' -> '{rewritten}' (expansions: {expansions})")
        return rewritten, expansions


query_rewriter = QueryRewriter()
