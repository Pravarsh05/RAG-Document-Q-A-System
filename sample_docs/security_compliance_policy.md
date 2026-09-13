# Security Architecture & API Compliance Policy

## 1. Authentication & Authorization
- **API Key Format**: High-entropy hexadecimal strings prefixed with `rag_sec_` followed by 32 cryptographic random bytes.
- **Header Transmission**: Clients transmit credentials via `X-API-Key: rag_sec_<token>` or `Authorization: Bearer rag_sec_<token>`.
- **Tenant Isolation**: Every ingested document and query execution is labeled with `user_id` or `tenant_id`. Documents uploaded by one tenant are strictly partitioned and cannot be retrieved across tenant boundaries.

## 2. File Upload Constraints & Defensive Ingestion
To mitigate Denial of Service (DoS) and malicious payload execution:
- **Maximum File Size**: 10 Megabytes (MB). Files exceeding this threshold receive HTTP 413.
- **Bounded Stream Reading**: Uploads are consumed in 64 Kilobyte chunks with running size accumulation, preventing memory exhaustion.
- **Allowed Formats**: `.pdf`, `.md`, `.markdown`, `.html`, `.htm`, `.txt`.
- **Magic Bytes Validation**:
  - PDF files must begin with the `%PDF-` signature.
  - HTML documents are inspected for structural tags (`<html>`, `<!DOCTYPE>`).
  - Text and Markdown documents are scanned for binary null-byte (`\x00`) injections and must decode cleanly under UTF-8.

## 3. CORS & Network Boundaries
- In production environments (`APP_ENV=production`), wildcard CORS (`*`) is explicitly prohibited.
- Permitted origins are configured via `CORS_ORIGINS` environment variables restricted to trusted dashboard domains.

## 4. Grounding and Refusal Mandate
- Models are prohibited from answering queries when retrieved evidence confidence falls below the `RELEVANCE_THRESHOLD` (0.20).
- When context is insufficient, the system must return an explicit refusal status `insufficient_evidence`.
- All factual claims must be cited using bracketed indices referencing verified source chunks.
