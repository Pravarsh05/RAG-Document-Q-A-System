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
