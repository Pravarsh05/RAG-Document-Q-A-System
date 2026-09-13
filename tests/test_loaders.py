import pytest
from ingestion.loaders import DocumentLoaderFactory, PDFLoader, MarkdownLoader, HTMLLoader, TextLoader


def test_markdown_loader():
    md_bytes = b"# System Title\n\nThis is a section paragraph.\n\n## Subtitle\nMore details."
    doc = DocumentLoaderFactory.load_bytes(md_bytes, "doc.md", "text/markdown")
    assert doc.filename == "doc.md"
    assert doc.content_type == "text/markdown"
    assert len(doc.pages) == 1
    assert "System Title" in doc.pages[0].text
    assert doc.pages[0].metadata.get("sections") == ["System Title", "Subtitle"]


def test_html_loader():
    html_bytes = b"<html><head><title>Test Report</title></head><body><h1>Report Header</h1><p>Main content text.</p></body></html>"
    doc = DocumentLoaderFactory.load_bytes(html_bytes, "report.html", "text/html")
    assert doc.filename == "report.html"
    assert doc.metadata.get("title") == "Test Report"
    assert "Report Header" in doc.full_text
    assert "Main content text." in doc.full_text


def test_text_loader():
    txt_bytes = b"Plain text file line 1\nLine 2 text."
    doc = DocumentLoaderFactory.load_bytes(txt_bytes, "notes.txt", "text/plain")
    assert doc.filename == "notes.txt"
    assert "Line 2" in doc.full_text


def test_pdf_loader_with_sample_pdf():
    import os
    pdf_path = os.path.join(os.path.dirname(__file__), "..", "sample_docs", "example.pdf")
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        doc = DocumentLoaderFactory.load_bytes(pdf_bytes, "example.pdf", "application/pdf")
        assert doc.filename == "example.pdf"
        assert len(doc.pages) >= 1


def test_html_loader_strips_scripts_and_styles():
    html_bytes = b"<html><head><style>.danger { color: red; }</style></head><body><script>alert('hack');</script><p>Clean visible content.</p></body></html>"
    doc = DocumentLoaderFactory.load_bytes(html_bytes, "page.html", "text/html")
    assert "Clean visible content." in doc.full_text
    assert "alert" not in doc.full_text
    assert ".danger" not in doc.full_text


def test_markdown_loader_without_headings():
    md_bytes = b"Just plain paragraphs in markdown format without any hashes."
    doc = DocumentLoaderFactory.load_bytes(md_bytes, "plain.md", "text/markdown")
    assert "Just plain paragraphs" in doc.full_text
    assert doc.pages[0].metadata.get("sections") == []


def test_text_loader_unicode_handling():
    text_bytes = "UTF-8 text with symbols: ©, ®, ™, and emojis: 🚀, 💡".encode("utf-8")
    doc = DocumentLoaderFactory.load_bytes(text_bytes, "unicode.txt", "text/plain")
    assert "©" in doc.full_text
    assert "🚀" in doc.full_text


def test_loader_factory_case_insensitive_extension():
    loader_upper = DocumentLoaderFactory.get_loader("DOCUMENT.PDF")
    assert isinstance(loader_upper, PDFLoader)

    loader_md = DocumentLoaderFactory.get_loader("NOTES.MARKDOWN")
    assert isinstance(loader_md, MarkdownLoader)


def test_loader_factory_mime_type_resolution():
    loader = DocumentLoaderFactory.get_loader("unnamed_file", content_type="text/markdown; charset=utf-8")
    assert isinstance(loader, MarkdownLoader)

