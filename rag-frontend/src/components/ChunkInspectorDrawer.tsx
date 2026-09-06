import { useEffect, useState } from "react";
import {
  Layers,
  Search,
  Copy,
  Check,
  X,
} from "lucide-react";
import { useDocumentChunks } from "@/hooks/useApi";
import type { DocumentSummary } from "@/types/api";

interface Props {
  document: DocumentSummary | null;
  onClose: () => void;
}

export function ChunkInspectorDrawer({ document, onClose }: Props) {
  const { data: chunks, isLoading, isError } = useDocumentChunks(document?.id || null);
  const [searchTerm, setSearchTerm] = useState("");
  const [copiedId, setCopiedId] = useState<string | null>(null);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  if (!document) return null;

  const filteredChunks = (chunks || []).filter((c) =>
    searchTerm ? c.content.toLowerCase().includes(searchTerm.toLowerCase()) : true
  );

  const handleCopy = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/70 backdrop-blur-sm transition-opacity animate-fadeIn">
      {/* Click outside backdrop to close */}
      <div className="flex-1" onClick={onClose} />

      <div className="flex h-full w-full max-w-2xl flex-col border-l border-ink-700/80 bg-ink-950 p-6 shadow-2xl overflow-hidden animate-slideInRight">
        {/* Header */}
        <div className="flex items-start justify-between border-b border-ink-800 pb-4">
          <div className="space-y-1 max-w-[80%]">
            <div className="flex items-center gap-2">
              <span className="flex items-center gap-1 font-mono text-xs uppercase tracking-wider text-lexical font-semibold">
                <Layers className="h-3.5 w-3.5" /> Chunk Partition Inspector
              </span>
              <span className="font-mono text-[11px] text-mist-400">
                &middot; {chunks?.length || document.chunkCount} partitions
              </span>
            </div>
            <h2 className="font-display text-base font-semibold text-mist-100 truncate" title={document.filename}>
              {document.filename}
            </h2>
            <div className="flex items-center gap-2 font-mono text-[11px] text-mist-400">
              <span className="rounded bg-ink-850 px-2 py-0.5 uppercase tracking-wider text-mist-300 border border-ink-800">
                {document.chunkingStrategy} strategy
              </span>
              <span>&middot;</span>
              <span className="uppercase text-vector font-semibold">{document.fileType}</span>
            </div>
          </div>
          <button
            onClick={onClose}
            title="Press Esc to close"
            className="flex items-center gap-1 rounded-md border border-ink-700 bg-ink-900 px-2.5 py-1 font-mono text-xs text-mist-400 transition-colors hover:border-ink-600 hover:text-mist-100 cursor-pointer"
          >
            <X className="h-3.5 w-3.5" />
            <span>Esc</span>
          </button>
        </div>

        {/* Search inside chunks */}
        <div className="relative my-4">
          <Search className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-mist-400" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search keyword within chunk content..."
            className="w-full rounded-md border border-ink-700 bg-ink-900/90 pl-9 pr-3.5 py-2.5 font-mono text-xs text-mist-100 placeholder:text-mist-400 focus:border-lexical focus:outline-none transition-colors"
          />
        </div>

        {/* Chunk list scroll area */}
        <div className="flex-1 space-y-3.5 overflow-y-auto pr-1">
          {isLoading && (
            <div className="py-12 text-center font-mono text-xs text-mist-400">
              <span className="inline-block h-2 w-2 animate-ping rounded-full bg-lexical mr-2" />
              Loading document chunk partitions...
            </div>
          )}

          {isError && (
            <div className="rounded-md border border-err/40 bg-err/10 p-4 font-mono text-xs text-err">
              Failed to retrieve chunks for document.
            </div>
          )}

          {!isLoading && filteredChunks.length === 0 && (
            <div className="py-12 text-center font-body text-xs text-mist-400">
              {searchTerm ? "No chunks match your search term." : "No chunks found in this document."}
            </div>
          )}

          {filteredChunks.map((chunk, idx) => {
            const approxTokens = Math.round(chunk.content.split(/\s+/).length * 1.3);
            const isCopied = copiedId === chunk.id;
            return (
              <div
                key={chunk.id || idx}
                className="rounded-lg border border-ink-800 bg-ink-900/80 p-4 transition-all hover:border-ink-700 shadow-subtle"
              >
                <div className="flex items-center justify-between border-b border-ink-850 pb-2 mb-2.5">
                  <div className="flex items-center gap-2 font-mono text-xs">
                    <span className="font-semibold text-vector">
                      Chunk #{chunk.chunk_index !== undefined ? chunk.chunk_index : idx + 1}
                    </span>
                    {chunk.page_number && (
                      <span className="text-mist-400 text-[11px]">
                        &middot; Page {chunk.page_number}
                      </span>
                    )}
                    {chunk.start_char !== undefined && chunk.end_char !== undefined && (
                      <span className="text-mist-400 text-[11px]">
                        &middot; Chars {chunk.start_char}:{chunk.end_char}
                      </span>
                    )}
                    <span className="text-mist-400 text-[11px]">
                      &middot; ~{approxTokens} tokens
                    </span>
                  </div>
                  <button
                    onClick={() => handleCopy(chunk.id, chunk.content)}
                    className="flex items-center gap-1 font-mono text-[11px] text-mist-400 hover:text-mist-200 transition-colors cursor-pointer"
                  >
                    {isCopied ? (
                      <>
                        <Check className="h-3 w-3 text-ok" />
                        <span className="text-ok">Copied</span>
                      </>
                    ) : (
                      <>
                        <Copy className="h-3 w-3" />
                        <span>Copy</span>
                      </>
                    )}
                  </button>
                </div>
                <pre className="font-mono text-xs leading-relaxed text-mist-200 whitespace-pre-wrap select-text font-normal">
                  {chunk.content}
                </pre>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
