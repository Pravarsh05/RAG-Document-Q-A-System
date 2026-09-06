import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from ingestion.loaders import LoadedDocument, DocumentPage


@dataclass
class ChunkData:
    content: str
    chunk_index: int
    page_number: Optional[int] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    chunking_strategy: str = "sentence"
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseChunker(ABC):
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk(self, doc: LoadedDocument) -> List[ChunkData]:
        pass


class FixedSizeChunker(BaseChunker):
    """Fixed-character or token length chunking with sliding window overlap."""

    def chunk(self, doc: LoadedDocument) -> List[ChunkData]:
        chunks: List[ChunkData] = []
        chunk_idx = 0

        for page in doc.pages:
            text = page.text.strip()
            if not text:
                continue

            start = 0
            text_len = len(text)

            while start < text_len:
                end = min(start + self.chunk_size, text_len)
                chunk_str = text[start:end].strip()

                if chunk_str:
                    chunks.append(
                        ChunkData(
                            content=chunk_str,
                            chunk_index=chunk_idx,
                            page_number=page.page_number,
                            start_char=start,
                            end_char=end,
                            chunking_strategy="fixed",
                            metadata={
                                "source": doc.filename,
                                "chunk_len": len(chunk_str),
                                **page.metadata,
                            },
                        )
                    )
                    chunk_idx += 1

                if end >= text_len:
                    break

                start += max(1, self.chunk_size - self.chunk_overlap)

        return chunks


class SentenceChunker(BaseChunker):
    """Chunks documents preserving complete sentence boundaries."""

    def _split_into_sentences(self, text: str) -> List[str]:
        # Split on sentence boundaries: ., !, ?, followed by whitespace or newline
        sentence_end = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])|\n\n+')
        sentences = [s.strip() for s in sentence_end.split(text) if s.strip()]
        return sentences

    def chunk(self, doc: LoadedDocument) -> List[ChunkData]:
        chunks: List[ChunkData] = []
        chunk_idx = 0

        for page in doc.pages:
            text = page.text.strip()
            if not text:
                continue

            sentences = self._split_into_sentences(text)
            if not sentences:
                continue

            current_chunk: List[str] = []
            current_len = 0
            sentence_start_indices: List[int] = []

            for sent in sentences:
                sent_len = len(sent)

                if current_len + sent_len + 1 > self.chunk_size and current_chunk:
                    chunk_content = " ".join(current_chunk)
                    chunks.append(
                        ChunkData(
                            content=chunk_content,
                            chunk_index=chunk_idx,
                            page_number=page.page_number,
                            chunking_strategy="sentence",
                            metadata={
                                "source": doc.filename,
                                "sentence_count": len(current_chunk),
                                **page.metadata,
                            },
                        )
                    )
                    chunk_idx += 1

                    # Create overlap by keeping the trailing sentences
                    overlap_sentences: List[str] = []
                    overlap_len = 0
                    for s in reversed(current_chunk):
                        if overlap_len + len(s) + 1 <= self.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_len += len(s) + 1
                        else:
                            break

                    current_chunk = overlap_sentences
                    current_len = sum(len(s) for s in current_chunk) + max(0, len(current_chunk) - 1)

                current_chunk.append(sent)
                current_len += sent_len + (1 if len(current_chunk) > 1 else 0)

            if current_chunk:
                chunk_content = " ".join(current_chunk)
                chunks.append(
                    ChunkData(
                        content=chunk_content,
                        chunk_index=chunk_idx,
                        page_number=page.page_number,
                        chunking_strategy="sentence",
                        metadata={
                            "source": doc.filename,
                            "sentence_count": len(current_chunk),
                            **page.metadata,
                        },
                    )
                )
                chunk_idx += 1

        return chunks


class SemanticChunker(BaseChunker):
    """
    Hierarchical recursive chunking: splits by markdown headings, paragraphs,
    then sentences, ensuring coherent semantic units.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        super().__init__(chunk_size, chunk_overlap)
        self.separators = ["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks: List[str] = []
        if not separators:
            return [text] if text else []

        sep = separators[0]
        splits = text.split(sep) if sep else list(text)

        current_doc: List[str] = []
        total = 0

        for piece in splits:
            if not piece.strip():
                continue
            piece_len = len(piece) + len(sep)
            if total + piece_len > self.chunk_size and current_doc:
                combined = sep.join(current_doc).strip()
                if len(combined) > self.chunk_size and len(separators) > 1:
                    # Recurse with finer separator
                    sub_chunks = self._split_text(combined, separators[1:])
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(combined)
                current_doc = []
                total = 0

            current_doc.append(piece)
            total += piece_len

        if current_doc:
            combined = sep.join(current_doc).strip()
            if len(combined) > self.chunk_size and len(separators) > 1:
                sub_chunks = self._split_text(combined, separators[1:])
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(combined)

        return final_chunks

    def chunk(self, doc: LoadedDocument) -> List[ChunkData]:
        chunks: List[ChunkData] = []
        chunk_idx = 0

        for page in doc.pages:
            text = page.text.strip()
            if not text:
                continue

            raw_pieces = self._split_text(text, self.separators)
            for piece in raw_pieces:
                p_clean = piece.strip()
                if p_clean:
                    chunks.append(
                        ChunkData(
                            content=p_clean,
                            chunk_index=chunk_idx,
                            page_number=page.page_number,
                            chunking_strategy="semantic",
                            metadata={
                                "source": doc.filename,
                                "length": len(p_clean),
                                **page.metadata,
                            },
                        )
                    )
                    chunk_idx += 1

        return chunks


class ChunkingStrategyFactory:
    _registry = {
        "fixed": FixedSizeChunker,
        "sentence": SentenceChunker,
        "semantic": SemanticChunker,
    }

    @classmethod
    def get_chunker(
        cls,
        strategy: str = "sentence",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> BaseChunker:
        strategy_key = strategy.lower().strip()
        chunker_cls = cls._registry.get(strategy_key, SentenceChunker)
        return chunker_cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
