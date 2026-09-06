import clsx from "clsx";
import type { RetrievalMode } from "@/types/api";

const modes: { value: RetrievalMode; label: string; hint: string; activeClass: string; badge: string }[] = [
  {
    value: "hybrid_rerank",
    label: "Hybrid + Rerank",
    badge: "Violet",
    hint: "Cross-Encoder candidate re-ranking (Best accuracy)",
    activeClass: "bg-hybrid/20 text-hybrid border-hybrid/60 shadow-[0_0_12px_rgba(185,140,242,0.3)] font-semibold",
  },
  {
    value: "hybrid",
    label: "Hybrid",
    badge: "RRF",
    hint: "Vector + BM25 Reciprocal Rank Fusion",
    activeClass: "bg-gradient-to-r from-vector/20 to-lexical/20 text-mist-100 border-hybrid/50 shadow-sm font-semibold",
  },
  {
    value: "vector",
    label: "Vector",
    badge: "Dense",
    hint: "Dense embedding cosine similarity (pgvector)",
    activeClass: "bg-vector/20 text-vector border-vector/60 shadow-[0_0_12px_rgba(45,212,191,0.25)] font-semibold",
  },
  {
    value: "bm25",
    label: "BM25",
    badge: "Lexical",
    hint: "Sparse lexical keyword matching (BM25Okapi)",
    activeClass: "bg-lexical/20 text-lexical border-lexical/60 shadow-[0_0_12px_rgba(242,183,5,0.25)] font-semibold",
  },
];

export function RetrievalModeToggle({
  value,
  onChange,
  vertical = false,
}: {
  value: RetrievalMode;
  onChange: (mode: RetrievalMode) => void;
  vertical?: boolean;
}) {
  return (
    <div
      className={clsx(
        "rounded-lg border border-ink-750 bg-ink-950/90 p-1.5 backdrop-blur-sm",
        vertical ? "flex flex-col gap-1.5 w-full" : "inline-flex items-center gap-1"
      )}
    >
      {modes.map((mode) => {
        const isSelected = value === mode.value;
        return (
          <button
            key={mode.value}
            type="button"
            onClick={() => onChange(mode.value)}
            title={mode.hint}
            className={clsx(
              "rounded-md font-mono text-[11px] transition-all cursor-pointer border text-left",
              vertical ? "flex items-center justify-between px-3 py-2" : "px-2.5 py-1.5",
              isSelected
                ? mode.activeClass
                : "border-transparent text-mist-400 hover:text-mist-200 hover:bg-ink-850/80"
            )}
          >
            <span>{mode.label}</span>
            {vertical && (
              <span
                className={clsx(
                  "rounded px-1.5 py-0.2 text-[9px] uppercase tracking-wider font-mono",
                  isSelected
                    ? "bg-ink-900/90 text-mist-200"
                    : "bg-ink-900 text-mist-400"
                )}
              >
                {mode.badge}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}

