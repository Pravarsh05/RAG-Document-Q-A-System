import clsx from "clsx";
import type { SourceChunk } from "@/types/api";

const signalColor: Record<string, string> = {
  bm25: "bg-lexical hover:bg-lexical/90 shadow-[0_0_8px_rgba(242,183,5,0.4)]",
  vector: "bg-vector hover:bg-vector/90 shadow-[0_0_8px_rgba(45,212,191,0.4)]",
  hybrid: "bg-hybrid hover:bg-hybrid/90 shadow-[0_0_8px_rgba(185,140,242,0.4)]",
  hybrid_rerank: "bg-hybrid hover:bg-hybrid/90 shadow-[0_0_8px_rgba(185,140,242,0.4)]",
};

const signalLabel: Record<string, string> = {
  bm25: "lexical (BM25)",
  vector: "semantic (vector)",
  hybrid: "hybrid (RRF)",
  hybrid_rerank: "hybrid + rerank",
};

/**
 * Signature element: a rank-ordered strip of the chunks that fed an
 * answer, colored by which retrieval signal surfaced each one.
 * Amber = lexical (BM25), Teal = semantic (Vector), Violet = hybrid / reranked.
 */
export function RetrievalTrace({
  chunks,
  selectedChunkId,
  onSelectChunk,
}: {
  chunks: SourceChunk[];
  selectedChunkId?: string | null;
  onSelectChunk?: (chunkId: string) => void;
}) {
  if (!chunks || chunks.length === 0) return null;

  const sorted = [...chunks].sort((a, b) => (a.rank || 1) - (b.rank || 1));

  return (
    <div className="rounded-md border border-ink-800 bg-ink-900/90 p-3 shadow-subtle">
      <div className="mb-2 flex items-center justify-between font-mono text-[11px] text-mist-400">
        <span className="flex items-center gap-1.5 font-semibold text-mist-200">
          <span className="h-1.5 w-1.5 rounded-full bg-hybrid" />
          Retrieval Trace ({sorted.length} candidate chunks)
        </span>
        <Legend />
      </div>
      <div className="flex gap-1.5 py-1">
        {sorted.map((chunk, idx) => {
          const sig = chunk.signal || "vector";
          const scoreVal = typeof chunk.score === "number" ? chunk.score.toFixed(3) : "1.000";
          const label = signalLabel[sig] || "semantic";
          const colorClass = signalColor[sig] || signalColor.vector;
          const isSelected = selectedChunkId === chunk.id;

          return (
            <button
              key={chunk.id || `trace-${idx}`}
              type="button"
              onClick={() => onSelectChunk?.(chunk.id)}
              title={`Rank #${chunk.rank || idx + 1}: ${chunk.documentName || "Document"} \u00b7 ${label} \u00b7 Score ${scoreVal}`}
              className={clsx(
                "group relative h-9 flex-1 rounded transition-all cursor-pointer border",
                colorClass,
                isSelected ? "ring-2 ring-mist-100 scale-105 border-white" : "border-transparent opacity-85 hover:opacity-100 hover:scale-105"
              )}
            >
              <span className="absolute inset-0 flex items-center justify-center font-mono text-[10px] font-bold text-ink-950">
                #{chunk.rank || idx + 1}
              </span>
            </button>
          );
        })}
      </div>
      <div className="mt-2 flex justify-between font-mono text-[10px] text-mist-400 border-t border-ink-850 pt-1.5">
        <span className="text-mist-300">#1 (Highest Cross-Encoder Score)</span>
        <span className="text-mist-400">#{sorted.length} Candidate cut-off</span>
      </div>
    </div>
  );
}

function Legend() {
  return (
    <div className="flex items-center gap-2.5 font-mono text-[10px]">
      <LegendItem color="bg-lexical" label="Amber: Lexical" />
      <LegendItem color="bg-vector" label="Teal: Semantic" />
      <LegendItem color="bg-hybrid" label="Violet: Hybrid" />
    </div>
  );
}

function LegendItem({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1">
      <span className={clsx("h-2 w-2 rounded-full shrink-0", color)} />
      <span className="text-mist-400">{label}</span>
    </span>
  );
}


