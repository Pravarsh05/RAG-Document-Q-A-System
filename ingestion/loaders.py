import os
import re
import io
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class DocumentPage:
    page_number: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadedDocument:
    filename: str
    content_type: str
    pages: List[DocumentPage]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())


class BaseLoader(ABC):
    @abstractmethod
    def load(self, file_content: bytes, filename: str) -> LoadedDocument:
        pass


class PDFLoader(BaseLoader):
    def load(self, file_content: bytes, filename: str) -> LoadedDocument:
        pages: List[DocumentPage] = []
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(file_content))
            for idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                pages.append(
                    DocumentPage(
                        page_number=idx + 1,
                        text=text.strip(),
                        metadata={"total_pages": len(reader.pages)},
                    )
                )
        except Exception as e:
            # Fallback if pypdf fails or corrupt binary
            raw_text = file_content.decode("utf-8", errors="ignore")
            pages.append(
                DocumentPage(
                    page_number=1,
                    text=raw_text.strip(),
                    metadata={"error": str(e), "fallback": True},
                )
            )

        return LoadedDocument(
            filename=filename,
            content_type="application/pdf",
            pages=pages if pages else [DocumentPage(page_number=1, text="")],
            metadata={"num_pages": len(pages)},
        )


class MarkdownLoader(BaseLoader):
    def load(self, file_content: bytes, filename: str) -> LoadedDocument:
        text = file_content.decode("utf-8", errors="replace")
        
        # We can detect major markdown section headers to provide section hints
        headings = re.findall(r"^(#{1,6})\s+(.+)$", text, flags=re.MULTILINE)
        section_titles = [h[1] for h in headings]

        return LoadedDocument(
            filename=filename,
            content_type="text/markdown",
            pages=[
                DocumentPage(
                    page_number=1,
                    text=text.strip(),
                    metadata={"sections": section_titles},
                )
            ],
            metadata={"sections_count": len(section_titles)},
        )


class HTMLLoader(BaseLoader):
    def load(self, file_content: bytes, filename: str) -> LoadedDocument:
        html_str = file_content.decode("utf-8", errors="replace")
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_str, "html.parser")
            
            # Remove scripts, styles, meta
            for element in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                element.decompose()

            title = soup.title.string.strip() if soup.title and soup.title.string else filename
            clean_text = soup.get_text(separator="\n\n")
            # Normalize excessive newlines
            clean_text = re.sub(r"\n{3,}", "\n\n", clean_text).strip()
            
            return LoadedDocument(
                filename=filename,
                content_type="text/html",
                pages=[
                    DocumentPage(
                        page_number=1,
                        text=clean_text,
                        metadata={"title": title},
                    )
                ],
                metadata={"title": title},
            )
        except Exception:
            # Fallback simple regex strip
            clean_text = re.sub(r"<[^>]+>", " ", html_str)
            clean_text = re.sub(r"\s+", " ", clean_text).strip()
            return LoadedDocument(
                filename=filename,
                content_type="text/html",
                pages=[DocumentPage(page_number=1, text=clean_text)],
                metadata={},
            )


class TextLoader(BaseLoader):
    def load(self, file_content: bytes, filename: str) -> LoadedDocument:
        text = file_content.decode("utf-8", errors="replace").strip()
        return LoadedDocument(
            filename=filename,
            content_type="text/plain",
            pages=[DocumentPage(page_number=1, text=text)],
            metadata={},
        )


class DocumentLoaderFactory:
    _loaders: Dict[str, BaseLoader] = {
        "pdf": PDFLoader(),
        "md": MarkdownLoader(),
        "markdown": MarkdownLoader(),
        "html": HTMLLoader(),
        "htm": HTMLLoader(),
        "txt": TextLoader(),
    }

    @classmethod
    def get_loader(cls, filename: str, content_type: Optional[str] = None) -> BaseLoader:
        ext = os.path.splitext(filename)[1].lower().lstrip(".")
        if ext in cls._loaders:
            return cls._loaders[ext]

        if content_type:
            ct = content_type.lower()
            if "pdf" in ct:
                return cls._loaders["pdf"]
            if "markdown" in ct:
                return cls._loaders["md"]
            if "html" in ct:
                return cls._loaders["html"]

        return cls._loaders["txt"]

    @classmethod
    def load_bytes(cls, file_content: bytes, filename: str, content_type: Optional[str] = None) -> LoadedDocument:
        loader = cls.get_loader(filename, content_type)
        return loader.load(file_content, filename)
