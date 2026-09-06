"""
Generates a valid standalone PDF file in sample_docs/example.pdf for testing.
"""
import os

def create_sample_pdf(filepath: str = "sample_docs/example.pdf"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Raw minimal valid PDF format
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj\n"
        b"<< /Type /Catalog /Pages 2 0 R >>\n"
        b"endobj\n"
        b"2 0 obj\n"
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>\n"
        b"endobj\n"
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\n"
        b"endobj\n"
        b"4 0 obj\n"
        b"<< /Length 210 >>\n"
        b"stream\n"
        b"BT\n"
        b"/F1 16 Tf\n"
        b"50 720 Td\n"
        b"(RAG Document Q&A System - Sample PDF Document) Tj\n"
        b"/F1 12 Tf\n"
        b"0 -30 Td\n"
        b"(This PDF contains benchmark information regarding pgvector and cross-encoder re-ranking.) Tj\n"
        b"0 -20 Td\n"
        b"(Hybrid search combines BM25 keyword search with dense vector similarity.) Tj\n"
        b"ET\n"
        b"endstream\n"
        b"endobj\n"
        b"5 0 obj\n"
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\n"
        b"endobj\n"
        b"xref\n"
        b"0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000234 00000 n \n"
        b"0000000495 00000 n \n"
        b"trailer\n"
        b"<< /Size 6 /Root 1 0 R >>\n"
        b"startxref\n"
        b"568\n"
        b"%%EOF\n"
    )

    with open(filepath, "wb") as f:
        f.write(pdf_content)

    print(f"Generated sample PDF at {filepath}")


if __name__ == "__main__":
    create_sample_pdf()
