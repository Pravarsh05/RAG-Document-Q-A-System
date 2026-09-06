import { useState, useRef, useEffect } from "react";
import clsx from "clsx";
import type { SourceChunk } from "@/types/api";

const signalColor: Record<string, string> = {
  bm25: "border-lexical/60 text-lexical bg-lexical/10 hover:bg-lexical/20 hover:border-lexical shadow-[0_0_8px_rgba(242,183,5,0.2)]",
  vector: "border-vector/60 text-vector bg-vector/10 hover:bg-vector/20 hover:border-vector shadow-[0_0_8px_rgba(45,212,191,0.2)]",
  hybrid: "border-hybrid/60 text-hybrid bg-hybrid/10 hover:bg-hybrid/20 hover:border-hybrid shadow-[0_0_8px_rgba(185,140,242,0.2)]",
  hybrid_rerank: "border-hybrid/60 text-hybrid bg-hybrid/10 hover:bg-hybrid/20 hover:border-hybrid shadow-[0_0_8px_rgba(185,140,242,0.2)]",
};

export function CitationChip({
  chunk,
  onSelect,
  isHighlighted = false,
}: {
  chunk: SourceChunk;
  onSelect?: (chunkId: string) => void;
  isHighlighted?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    if (open) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [open]);

  const sig = chunk.signal || "vector";
  const colorClass = signalColor[sig] || signalColor.vector;
  const scoreText = typeof chunk.score === "number" ? chunk.score.toFixed(3) : "1.000";

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    setOpen((prev) => !prev);
    if (onSelect && chunk.id) {
      onSelect(chunk.id);
    }
  };

  return (
    <span ref={containerRef} className="relative inline-block align-baseline mx-0.5">
      <button
        type="button"
        onClick={handleClick}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        title={`Source #${chunk.rank || 1}: ${chunk.documentName || "Document"} (Score: ${scoreText})`}
        className={clsx(
          "inline-flex items-center gap-0.5 rounded px-1.5 py-0.2 font-mono text-[10px] font-semibold leading-none transition-all cursor-pointer border select-none",
          colorClass,
          isHighlighted && "ring-2 ring-mist-100 scale-105"
        )}
      >
        <span>[{chunk.rank || 1}]</span>
      </button>

      {open && (
        <div
          className="absolute bottom-full left-1/2 z-50 mb-2 w-72 -translate-x-1/2 rounded-md border border-ink-700 bg-ink-900/98 p-3 text-xs shadow-2xl backdrop-blur-md ring-1 ring-black/80 animate-fadeIn"
          onMouseEnter={() => setOpen(true)}
          onMouseLeave={() => setOpen(false)}
        >
          <div className="mb-1.5 flex items-center justify-between border-b border-ink-800 pb-1.5 font-mono text-[10px]">
            <span className="truncate max-w-[170px] font-semibold text-mist-100" title={chunk.documentName}>
              {chunk.documentName || "Source Document"}
              {chunk.page ? ` \u00b7 p.${chunk.page}` : ""}
            </span>
            <span className="shrink-0 rounded bg-ink-800 px-1.5 py-0.2 text-[9px] text-mist-300">
              score: {scoreText}
            </span>
          </div>
          <p className="max-h-36 overflow-y-auto font-mono text-[11px] leading-relaxed text-mist-300">
            {chunk.text || "No snippet content preview available."}
          </p>
          <div className="mt-2 flex items-center justify-between border-t border-ink-800/80 pt-1.5 font-mono text-[9px] text-mist-400">
            <span className="uppercase text-mist-400">Signal: {sig}</span>
            {onSelect && (
              <span className="text-lexical hover:underline cursor-pointer" onClick={() => onSelect(chunk.id)}>
                View in Right Inspector &rarr;
              </span>
            )}
          </div>
        </div>
      )}
    </span>
  );
}


