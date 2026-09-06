import re
import time
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from config import settings
from retrieval.vector_search import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    chunk_id: str
    filename: Optional[str] = None
    page_number: Optional[int] = None
    snippet: Optional[str] = None


@dataclass
class GenerationResult:
    question: str
    answer: str
    citations: List[Citation]
    retrieved_chunks: List[SearchResult]
    generation_latency_ms: float
    model_name: str
    usage: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": [
                {
                    "chunk_id": c.chunk_id,
                    "filename": c.filename,
                    "page_number": c.page_number,
                    "snippet": c.snippet,
                }
                for c in self.citations
            ],
            "retrieved_chunks": [r.to_dict() for r in self.retrieved_chunks],
            "generation_latency_ms": round(self.generation_latency_ms, 2),
            "model_name": self.model_name,
            "usage": self.usage,
        }


SYSTEM_PROMPT = """You are an accurate, structured, and helpful Question-Answering assistant.
Your task is to answer the user's question STRICTLY and ONLY using the provided source context chunks.

FORMATTING & CLARITY RULES:
1. Grounding: Answer ONLY based on facts directly stated in the context chunks below. Do NOT assume or extrapolate.
2. Structure & Readability: Present information clearly using clean markdown. Use bullet points (- ) with bold conceptual titles (**Key Concept**) rather than clustered walls of text.
3. Citations: Every statement or bullet point MUST be cited using standard bracketed chunk numbers like [1], [2], [3] directly beside the claim. Do NOT output raw UUIDs or "Chunk <uuid>" strings.
4. Incomplete Information: If the provided context does not contain enough information to answer the question, state: "I cannot answer this question based on the provided documents."
5. Accuracy: Preserve exact formulas, metrics, model names, and parameter values.
"""


def format_context(chunks: List[SearchResult]) -> str:
    if not chunks:
        return "No relevant context found."

    formatted_parts = []
    for idx, c in enumerate(chunks, start=1):
        page_info = f", Page {c.page_number}" if c.page_number else ""
        header = f"--- [Chunk {idx}] (ID: {c.chunk_id}, Source: {c.filename}{page_info}) ---"
        formatted_parts.append(f"{header}\n{c.content.strip()}")

    return "\n\n".join(formatted_parts)


class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        pass


class AnthropicLLMClient(BaseLLMClient):
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key or settings.ANTHROPIC_API_KEY
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self.api_key)
        return self._client

    def generate(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        answer = response.content[0].text if response.content else ""
        return {
            "answer": answer,
            "model": self.model,
            "usage": {
                "input_tokens": response.usage.input_tokens if hasattr(response, "usage") else 0,
                "output_tokens": response.usage.output_tokens if hasattr(response, "usage") else 0,
            },
        }


class OpenAILLMClient(BaseLLMClient):
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def generate(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        answer = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        }
        return {
            "answer": answer,
            "model": self.model,
            "usage": usage,
        }


class MockLLMClient(BaseLLMClient):
    """
    Intelligent extractive QA synthesizer for offline evaluation, zero-key usage,
    and deterministic local environments. Extracts relevant sentences and facts
    directly from the retrieved context chunks, formatting structured, cited markdown.
    """

    STOP_WORDS = {
        "what", "is", "are", "the", "a", "an", "of", "and", "in", "to", "for",
        "with", "on", "at", "by", "from", "how", "does", "do", "which", "why",
        "who", "when", "where", "can", "could", "would", "should", "about",
        "tell", "me", "give", "list", "describe", "explain", "this", "that"
    }

    def generate(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        # 1. Parse question
        q_match = re.search(r"Question:\s*(.+?)(?:\n\s*Answer:|\Z)", user_prompt, re.DOTALL)
        question = q_match.group(1).strip() if q_match else ""
        query_words = set(re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", question.lower())) - self.STOP_WORDS

        # 2. Parse chunks from prompt: --- [Chunk X] (ID: Y, Source: Z) ---
        chunk_pattern = re.compile(
            r"---\s*\[Chunk\s+(\d+)\](?:\s*\(ID:\s*([^,]+),\s*Source:\s*([^)]*)\))?\s*---\n(.*?)(?=(?:---\s*\[Chunk|\n\nQuestion:|\Z))",
            re.DOTALL
        )
        chunks = []
        for m in chunk_pattern.finditer(user_prompt):
            c_index = int(m.group(1))
            c_id = (m.group(2) or "").strip()
            source = (m.group(3) or "").strip()
            content = m.group(4).strip()
            if content:
                chunks.append({"index": c_index, "id": c_id, "source": source, "content": content})

        # Fallback parser if old prompt header format
        if not chunks:
            fallback_pattern = re.compile(
                r"---\s*\[Chunk\s+([^\]]+)\](?:\s*\(Source:\s*([^)]*)\))?\s*---\n(.*?)(?=(?:---\s*\[Chunk|\n\nQuestion:|\Z))",
                re.DOTALL
            )
            for idx, m in enumerate(fallback_pattern.finditer(user_prompt), start=1):
                cid = m.group(1).strip()
                source = (m.group(2) or "").strip()
                content = m.group(3).strip()
                if content:
                    chunks.append({"index": idx, "id": cid, "source": source, "content": content})

        if not chunks:
            return {
                "answer": "I cannot answer this question based on the provided documents as no relevant context was found.",
                "model": "extractive-synthesizer-v1",
                "usage": {"input_tokens": 50, "output_tokens": 20},
            }

        # 3. Extract and score sentences across chunks with clear header separation
        scored_points = []
        for c in chunks:
            # First split by clean line blocks to preserve markdown headers
            raw_blocks = [b.strip() for b in c["content"].split("\n") if b.strip()]
            
            pending_heading = None
            for block in raw_blocks:
                clean_block = re.sub(r"^#+\s*", "", block).strip()
                # If short block with no terminal punctuation, treat as heading
                if len(clean_block) < 50 and not clean_block.endswith((".", "!", "?", ":")):
                    pending_heading = clean_block
                    continue

                # Split block into distinct sentences
                sentences = re.split(r"(?<=[.!?])\s+", clean_block)
                for s in sentences:
                    clean_s = s.strip()
                    if len(clean_s) < 15:
                        continue

                    # If we had a pending heading, attach it
                    if pending_heading:
                        clean_s = f"**{pending_heading}**: {clean_s}"
                        pending_heading = None

                    s_words = set(re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", clean_s.lower()))
                    matches = s_words & query_words
                    score = len(matches) * 2

                    # Boost keywords indicating key mechanisms / findings
                    if any(k in clean_s.lower() for k in ["supports", "includes", "provides", "features", "allows", "strategy", "architecture", "consists", "formula", "hybrid", "rerank", "fusion", "vector"]):
                        score += 1

                    scored_points.append({
                        "index": c["index"],
                        "chunk_id": c["id"] or str(c["index"]),
                        "source": c["source"],
                        "text": clean_s,
                        "score": score,
                    })

        # Sort by score
        scored_points.sort(key=lambda x: x["score"], reverse=True)
        top_points = [p for p in scored_points if p["score"] > 0][:6]

        if top_points:
            # Deduplicate similar text
            seen_texts = set()
            unique_points = []
            for p in top_points:
                prefix = p["text"][:35].lower()
                if prefix not in seen_texts:
                    seen_texts.add(prefix)
                    unique_points.append(p)

            formatted_bullets = []
            for p in unique_points[:4]:
                txt = p["text"]
                # Ensure bold lead or colon format
                if ":" in txt and not txt.startswith("**"):
                    parts = txt.split(":", 1)
                    if len(parts[0]) < 40:
                        txt = f"**{parts[0].strip()}**: {parts[1].strip()}"
                elif not txt.startswith("**"):
                    words = txt.split(" ")
                    if len(words) > 4:
                        lead = " ".join(words[:3])
                        rest = " ".join(words[3:])
                        txt = f"**{lead}** {rest}"

                formatted_bullets.append(f"- {txt} [{p['index']}]")

            lead_idx = unique_points[0]["index"]
            answer = (
                f"Based on the indexed documentation [{lead_idx}], here are the key details:\n\n"
                + "\n\n".join(formatted_bullets)
            )
        else:
            # Fallback: extract clean excerpt with proper bullet points
            lead_chunk = chunks[0]
            raw_lines = [l.strip() for l in lead_chunk["content"].split("\n") if len(l.strip()) > 20]
            lead_points = []
            for l in raw_lines[:3]:
                clean_l = re.sub(r"^#+\s*", "", l)
                lead_points.append(f"- {clean_l} [{lead_chunk['index']}]")

            answer = (
                f"Based on {lead_chunk['source'] or 'the indexed documents'} [{lead_chunk['index']}]:\n\n"
                + "\n\n".join(lead_points)
            )

        return {
            "answer": answer,
            "model": "extractive-synthesizer-v1",
            "usage": {
                "input_tokens": sum(len(c["content"].split()) for c in chunks),
                "output_tokens": len(answer.split()),
            },
        }


class GenerationService:
    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        self._llm_client = llm_client

    @property
    def llm_client(self) -> BaseLLMClient:
        if self._llm_client is not None:
            return self._llm_client
        return self._resolve_llm_client()

    @llm_client.setter
    def llm_client(self, client: Optional[BaseLLMClient]):
        self._llm_client = client

    def _resolve_llm_client(self) -> BaseLLMClient:
        provider = settings.LLM_PROVIDER.lower()
        if provider == "anthropic" and settings.ANTHROPIC_API_KEY:
            return AnthropicLLMClient(model=settings.LLM_MODEL)
        elif provider == "openai" and settings.OPENAI_API_KEY:
            return OpenAILLMClient(model=settings.LLM_MODEL or "gpt-4o-mini")
        else:
            return MockLLMClient()

    def _extract_citations(self, answer: str, chunks: List[SearchResult]) -> List[Citation]:
        chunk_map = {c.chunk_id: c for c in chunks}
        index_map = {idx: c for idx, c in enumerate(chunks, start=1)}

        citations: List[Citation] = []
        seen_ids = set()

        # 1. Match numeric citations e.g. [1], [2]
        numeric_matches = re.findall(r"\[(\d+)\]", answer)
        for num_str in numeric_matches:
            num = int(num_str)
            if num in index_map:
                chunk = index_map[num]
                if chunk.chunk_id not in seen_ids:
                    citations.append(
                        Citation(
                            chunk_id=chunk.chunk_id,
                            filename=chunk.filename,
                            page_number=chunk.page_number,
                            snippet=chunk.content[:160] + "..." if len(chunk.content) > 160 else chunk.content,
                        )
                    )
                    seen_ids.add(chunk.chunk_id)

        # 2. Match [Chunk <id>] or [Chunk <number>]
        named_matches = re.findall(r"\[Chunk\s+([^\]]+)\]", answer, flags=re.IGNORECASE)
        for cid in named_matches:
            clean_id = cid.strip()
            # If digit
            if clean_id.isdigit() and int(clean_id) in index_map:
                chunk = index_map[int(clean_id)]
                if chunk.chunk_id not in seen_ids:
                    citations.append(
                        Citation(
                            chunk_id=chunk.chunk_id,
                            filename=chunk.filename,
                            page_number=chunk.page_number,
                            snippet=chunk.content[:160] + "..." if len(chunk.content) > 160 else chunk.content,
                        )
                    )
                    seen_ids.add(chunk.chunk_id)
            elif clean_id in chunk_map and clean_id not in seen_ids:
                chunk = chunk_map[clean_id]
                citations.append(
                    Citation(
                        chunk_id=chunk.chunk_id,
                        filename=chunk.filename,
                        page_number=chunk.page_number,
                        snippet=chunk.content[:160] + "..." if len(chunk.content) > 160 else chunk.content,
                    )
                )
                seen_ids.add(chunk.chunk_id)

        # 3. Fallback if no explicit citations parsed
        if not citations and chunks:
            citations.append(
                Citation(
                    chunk_id=chunks[0].chunk_id,
                    filename=chunks[0].filename,
                    page_number=chunks[0].page_number,
                    snippet=chunks[0].content[:160] + "...",
                )
            )

        return citations

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[SearchResult],
    ) -> GenerationResult:
        context_str = format_context(retrieved_chunks)
        user_prompt = f"Context:\n{context_str}\n\nQuestion: {question}\n\nAnswer:"

        start_time = time.time()
        response_dict = self.llm_client.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        latency_ms = (time.time() - start_time) * 1000

        answer = response_dict.get("answer", "")
        model_name = response_dict.get("model", "unknown")
        usage = response_dict.get("usage", {})

        citations = self._extract_citations(answer, retrieved_chunks)

        return GenerationResult(
            question=question,
            answer=answer,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            generation_latency_ms=latency_ms,
            model_name=model_name,
            usage=usage,
        )
