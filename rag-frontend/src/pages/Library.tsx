import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import {
  UploadCloud,
  FileText,
  FileCode,
  Globe,
  File,
  Trash2,
  Eye,
  Layers,
  HardDrive,
  Cpu,
} from "lucide-react";
import { useDocuments, useIngestDocument, useDeleteDocument } from "@/hooks/useApi";
import { StatusPill } from "@/components/StatusPill";
import { EmptyState } from "@/components/EmptyState";
import { ChunkInspectorDrawer } from "@/components/ChunkInspectorDrawer";
import type { ChunkingStrategy, DocumentSummary } from "@/types/api";

const strategies: ChunkingStrategy[] = ["sentence", "semantic", "fixed"];

const fileTypeBadge: Record<string, { style: string; icon: typeof FileText }> = {
  pdf: { style: "border-err/40 text-err bg-err/10", icon: FileText },
  markdown: { style: "border-vector/40 text-vector bg-vector/10", icon: FileCode },
  html: { style: "border-lexical/40 text-lexical bg-lexical/10", icon: Globe },
  txt: { style: "border-mist-400/40 text-mist-200 bg-ink-800", icon: File },
};

export function LibraryPage() {
  const { data: documents, isLoading, isError } = useDocuments();
  const ingest = useIngestDocument();
  const deleteDoc = useDeleteDocument();
  const [strategy, setStrategy] = useState<ChunkingStrategy>("sentence");
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [inspectDoc, setInspectDoc] = useState<DocumentSummary | null>(null);

  const onDrop = useCallback(
    (files: File[]) => {
      files.forEach((file) => ingest.mutate({ file, strategy }));
    },
    [ingest, strategy]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/pdf": [".pdf"],
      "text/markdown": [".md", ".markdown"],
      "text/html": [".html", ".htm"],
      "text/plain": [".txt"],
    },
  });

  const handleDelete = (docId: string, filename: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (window.confirm(`Delete "${filename}" and all its indexed vector/BM25 chunks?`)) {
      setDeletingId(docId);
      deleteDoc.mutate(docId, {
        onSettled: () => setDeletingId(null),
      });
    }
  };

  const totalChunks = documents?.reduce((acc, d) => acc + (d.chunkCount || 0), 0) ?? 0;
  const avgChunks = documents && documents.length > 0 ? Math.round(totalChunks / documents.length) : 0;

  return (
    <div className="mx-auto max-w-4xl px-6 py-8">
      {/* Header bar */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="font-display text-lg font-semibold text-mist-100 tracking-tight">Document Knowledge Base</h1>
            <span className="rounded bg-ink-800 px-2 py-0.5 font-mono text-[10px] text-mist-300 border border-ink-700">
              {documents?.length || 0} Files
            </span>
          </div>
          <p className="font-body text-xs text-mist-400">
            Multi-strategy indexed documents with dense vectors and sparse BM25 representations.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 rounded-md border border-ink-700 bg-ink-900 px-3 py-1.5 font-mono text-xs text-mist-300">
            <Cpu className="h-3.5 w-3.5 text-lexical" />
            <span className="text-mist-400 text-[11px]">Chunking:</span>
            <select
              value={strategy}
              onChange={(e) => setStrategy(e.target.value as ChunkingStrategy)}
              className="bg-transparent font-mono text-xs text-mist-100 focus:outline-none cursor-pointer"
            >
              {strategies.map((s) => (
                <option key={s} value={s} className="bg-ink-900 text-mist-100">
                  {s}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Stats Summary Row */}
      <div className="mt-6 grid grid-cols-1 gap-3 sm:grid-cols-3">
        <div className="rounded-lg border border-ink-800 bg-ink-900/80 p-4 shadow-subtle flex items-center gap-3.5">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md bg-vector/10 border border-vector/30 text-vector">
            <HardDrive className="h-5 w-5" />
          </div>
          <div>
            <div className="font-mono text-[10px] uppercase tracking-wider text-mist-400">Total Indexed Documents</div>
            <div className="font-mono text-xl font-semibold text-mist-100">{documents?.length || 0}</div>
          </div>
        </div>
        <div className="rounded-lg border border-ink-800 bg-ink-900/80 p-4 shadow-subtle flex items-center gap-3.5">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md bg-lexical/10 border border-lexical/30 text-lexical">
            <Layers className="h-5 w-5" />
          </div>
          <div>
            <div className="font-mono text-[10px] uppercase tracking-wider text-mist-400">Total Chunk Partitions</div>
            <div className="font-mono text-xl font-semibold text-mist-100">{totalChunks}</div>
          </div>
        </div>
        <div className="rounded-lg border border-ink-800 bg-ink-900/80 p-4 shadow-subtle flex items-center gap-3.5">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md bg-hybrid/10 border border-hybrid/30 text-hybrid">
            <Cpu className="h-5 w-5" />
          </div>
          <div>
            <div className="font-mono text-[10px] uppercase tracking-wider text-mist-400">Avg Chunks / File</div>
            <div className="font-mono text-xl font-semibold text-mist-100">{avgChunks}</div>
          </div>
        </div>
      </div>

      {/* Drag and Drop Zone */}
      <div
        {...getRootProps()}
        className={`mt-6 cursor-pointer rounded-lg border-2 border-dashed px-6 py-10 text-center transition-all ${
          isDragActive
            ? "border-lexical bg-lexical/10 shadow-glow-lexical scale-[1.01]"
            : "border-ink-700 hover:border-ink-600 bg-ink-900/40 hover:bg-ink-900/70"
        }`}
      >
        <input {...getInputProps()} />
        <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full border border-ink-700 bg-ink-850 text-lexical shadow-sm">
          <UploadCloud className="h-5 w-5" />
        </div>
        <p className="font-body text-sm font-medium text-mist-100">
          Drop PDF, Markdown, HTML, or TXT documents here, or <span className="text-lexical underline underline-offset-4">browse</span>
        </p>
        <p className="mt-1.5 font-mono text-[11px] text-mist-400">
          Parsed with <strong className="text-mist-200">"{strategy}"</strong> strategy &middot; Dual indexed into pgvector embeddings and BM25 index.
        </p>
      </div>

      {ingest.isPending && (
        <div className="mt-4 flex items-center gap-3 rounded-lg border border-lexical/40 bg-lexical/10 px-4 py-3 font-mono text-xs text-lexical animate-fadeIn">
          <span className="inline-block h-2.5 w-2.5 animate-ping rounded-full bg-lexical" />
          <span>Ingesting, partitioning chunks & generating embeddings for document...</span>
        </div>
      )}

      {ingest.isError && (
        <div className="mt-4 rounded-lg border border-err/40 bg-err/10 px-4 py-3 font-mono text-xs text-err shadow-sm">
          {(ingest.error as Error).message || "Failed to ingest document. Ensure the file format is supported."}
        </div>
      )}

      {/* Document Records */}
      <div className="mt-8">
        {isLoading && (
          <p className="font-mono text-xs text-mist-400">Loading indexed documents...</p>
        )}
        {isError && (
          <div className="rounded-lg border border-err/40 bg-err/10 p-4 font-mono text-xs text-err">
            Could not reach backend GET /documents. Make sure the backend server is running on port 8000.
          </div>
        )}
        {documents && documents.length === 0 && !isLoading && (
          <EmptyState
            title="No documents indexed yet"
            body="Drag and drop a PDF, Markdown, or TXT file above to start building your search index."
          />
        )}
        {documents && documents.length > 0 && (
          <div className="overflow-x-auto rounded-lg border border-ink-800 bg-ink-900/60 shadow-card">
            <table className="w-full border-collapse font-body text-sm">
              <thead>
                <tr className="border-b border-ink-700 bg-ink-900/90 text-left font-mono text-[11px] uppercase tracking-wide text-mist-400">
                  <th className="py-3 px-4 font-medium">Document</th>
                  <th className="py-3 px-3 font-medium">Format</th>
                  <th className="py-3 px-3 font-medium">Strategy</th>
                  <th className="py-3 px-3 font-medium text-right">Chunks</th>
                  <th className="py-3 px-3 font-medium">Ingested</th>
                  <th className="py-3 px-3 font-medium">Status</th>
                  <th className="py-3 px-4 font-medium text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-ink-850">
                {documents.map((doc) => {
                  const typeKey = (doc.fileType || "txt").toLowerCase();
                  const typeMeta = fileTypeBadge[typeKey] || fileTypeBadge.txt;
                  const Icon = typeMeta.icon;
                  const dateStr = doc.ingestedAt
                    ? new Date(doc.ingestedAt).toLocaleString(undefined, {
                        month: "short",
                        day: "numeric",
                        hour: "2-digit",
                        minute: "2-digit",
                      })
                    : "--";

                  return (
                    <tr
                      key={doc.id}
                      onClick={() => setInspectDoc(doc)}
                      className="cursor-pointer transition-colors hover:bg-ink-850/80 group"
                    >
                      <td className="py-3.5 px-4 font-medium text-mist-100 max-w-[220px] truncate group-hover:text-lexical transition-colors" title={doc.filename}>
                        <div className="flex items-center gap-2.5">
                          <Icon className="h-4 w-4 text-mist-400 shrink-0 group-hover:text-lexical" />
                          <span className="truncate">{doc.filename}</span>
                        </div>
                      </td>
                      <td className="py-3.5 px-3">
                        <span className={`inline-block rounded px-2 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-wider border ${typeMeta.style}`}>
                          {doc.fileType}
                        </span>
                      </td>
                      <td className="py-3.5 px-3 font-mono text-xs text-mist-300">
                        {doc.chunkingStrategy}
                      </td>
                      <td className="py-3.5 px-3 font-mono text-xs text-mist-100 text-right font-semibold">
                        {doc.chunkCount}
                      </td>
                      <td className="py-3.5 px-3 font-mono text-xs text-mist-400">
                        {dateStr}
                      </td>
                      <td className="py-3.5 px-3">
                        <StatusPill status={doc.status} />
                      </td>
                      <td className="py-3.5 px-4 text-right">
                        <div className="flex items-center justify-end gap-2">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setInspectDoc(doc);
                            }}
                            title="Inspect chunk partitions"
                            className="flex items-center gap-1 rounded border border-ink-700 bg-ink-850 px-2.5 py-1 font-mono text-xs text-mist-300 transition-all hover:border-lexical/60 hover:text-mist-100 cursor-pointer"
                          >
                            <Eye className="h-3 w-3" />
                            <span>Inspect</span>
                          </button>
                          <button
                            onClick={(e) => handleDelete(doc.id, doc.filename, e)}
                            disabled={deletingId === doc.id}
                            title="Delete document and chunks"
                            className="flex items-center gap-1 rounded border border-err/30 px-2 py-1 font-mono text-xs text-err/80 transition-all hover:bg-err/10 hover:border-err hover:text-err disabled:opacity-40 cursor-pointer"
                          >
                            <Trash2 className="h-3 w-3" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Chunk Inspector Drawer */}
      <ChunkInspectorDrawer
        document={inspectDoc}
        onClose={() => setInspectDoc(null)}
      />
    </div>
  );
}
