import { useEffect, useRef, useState, useCallback, ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import { useDropzone } from "react-dropzone";
import {
  Sparkles,
  Send,
  Bot,
  User,
  Copy,
  Check,
  RotateCcw,
  FileText,
  Zap,
  Clock,
  ShieldCheck,
  Target,
  Layers,
  UploadCloud,
  Trash2,
  Eye,
  Activity,
  Terminal,
  FileCode,
  Globe,
  File,
  Filter,
  Flame,
  AlertTriangle,
  ShieldAlert,
} from "lucide-react";
import {
  useAskQuestion,
  useDocuments,
  useIngestDocument,
  useDeleteDocument,
  useHealth,
  useEvalResults,
} from "@/hooks/useApi";
import { RetrievalModeToggle } from "@/components/RetrievalModeToggle";
import { RetrievalTrace } from "@/components/RetrievalTrace";
import { CitationChip } from "@/components/CitationChip";
import { ChunkInspectorDrawer } from "@/components/ChunkInspectorDrawer";
import type {
  RetrievalMode,
  SourceChunk,
  ChunkingStrategy,
  DocumentSummary,
} from "@/types/api";

interface Turn {
  question: string;
  answer: string;
  citations: SourceChunk[];
  retrievedChunks: SourceChunk[];
  latencyMs: number;
  latencyDetails?: { retrieval_ms?: number; generation_ms?: number; total_ms?: number };
  mode: RetrievalMode;
  modelName?: string;
  cached?: boolean;
  groundingStatus?: string;
  rewrittenQuery?: string;
}

const fileTypeIconMap: Record<string, { color: string; icon: typeof FileText }> = {
  pdf: { color: "text-rose-400 border-rose-500/30 bg-rose-500/10", icon: FileText },
  markdown: { color: "text-vector border-vector/30 bg-vector/10", icon: FileCode },
  html: { color: "text-lexical border-lexical/30 bg-lexical/10", icon: Globe },
  txt: { color: "text-mist-300 border-ink-700 bg-ink-800", icon: File },
};

const chunkingStrategies: ChunkingStrategy[] = ["sentence", "semantic", "fixed"];

export function QueryPage() {
  // State
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<RetrievalMode>("hybrid_rerank");
  const [selectedDocId, setSelectedDocId] = useState<string>("");
  const [chunkStrategy, setChunkStrategy] = useState<ChunkingStrategy>("sentence");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [copiedTurnIdx, setCopiedTurnIdx] = useState<number | null>(null);
  const [selectedChunkId, setSelectedChunkId] = useState<string | null>(null);
  const [inspectDoc, setInspectDoc] = useState<DocumentSummary | null>(null);
  const [copiedChunkId, setCopiedChunkId] = useState<string | null>(null);

  // APIs
  const ask = useAskQuestion();
  const { data: documents, isLoading: docsLoading } = useDocuments();
  const ingest = useIngestDocument();
  const deleteDoc = useDeleteDocument();
  const { data: health } = useHealth();
  const { data: evalRows } = useEvalResults();

  // Refs
  const inputRef = useRef<HTMLInputElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const chunkListRef = useRef<HTMLDivElement>(null);

  // Global Keyboard Accelerator: '/' to focus input, 'Esc' to blur
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const activeEl = document.activeElement;
      const isInput =
        activeEl?.tagName === "INPUT" ||
        activeEl?.tagName === "TEXTAREA" ||
        activeEl?.tagName === "SELECT";

      if (e.key === "/" && !isInput) {
        e.preventDefault();
        inputRef.current?.focus();
      } else if (e.key === "Escape" && document.activeElement === inputRef.current) {
        inputRef.current?.blur();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  // Dropzone setup
  const onDrop = useCallback(
    (files: File[]) => {
      files.forEach((file) => ingest.mutate({ file, strategy: chunkStrategy }));
    },
    [ingest, chunkStrategy]
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

  const handleCancel = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
  };

  const handleCopyAnswer = (idx: number, answerText: string) => {
    navigator.clipboard.writeText(answerText);
    setCopiedTurnIdx(idx);
    setTimeout(() => setCopiedTurnIdx(null), 2000);
  };

  const handleCopyChunk = (chunkId: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedChunkId(chunkId);
    setTimeout(() => setCopiedChunkId(null), 2000);
  };

  const handleSelectChunk = (chunkId: string) => {
    setSelectedChunkId(chunkId);
    // Smooth scroll to chunk in right column
    const element = document.getElementById(`source-chunk-${chunkId}`);
    if (element) {
      element.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  };

  const submit = (overrideQuestion?: string) => {
    const q = (overrideQuestion ?? question).trim();
    if (!q) return;

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    ask.mutate(
      {
        question: q,
        mode,
        filterDocumentId: selectedDocId || undefined,
        signal: controller.signal,
      },
      {
        onSuccess: (res) => {
          const newCitations = res.citations || [];
          const newRetrieved = res.retrievedChunks || res.retrieved_chunks || res.citations || [];
          const grounding = res.grounding_status || res.groundingStatus;
          const rewritten = res.rewritten_query;
          setTurns((prev) => [
            ...prev,
            {
              question: q,
              answer: res.answer,
              citations: newCitations,
              retrievedChunks: newRetrieved,
              latencyMs: res.latencyMs || 0,
              latencyDetails: res.latency_ms,
              mode: res.retrievalMode || res.retrieval_mode || mode,
              modelName: res.modelName || res.model_name,
              cached: res.cached,
              groundingStatus: grounding,
              rewrittenQuery: rewritten,
            },
          ]);
          setQuestion("");
          abortControllerRef.current = null;
          if (newRetrieved.length > 0) {
            setSelectedChunkId(newRetrieved[0].id);
          }
        },
        onError: () => {
          abortControllerRef.current = null;
        },
      }
    );
  };

  const handleDeleteDoc = (docId: string, filename: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (window.confirm(`Delete "${filename}" and all its vector/BM25 chunks?`)) {
      deleteDoc.mutate(docId);
      if (selectedDocId === docId) {
        setSelectedDocId("");
      }
    }
  };

  const latestTurn = turns.length > 0 ? turns[turns.length - 1] : null;
  const activeSources = latestTurn?.retrievedChunks || latestTurn?.citations || [];

  // Metrics from real evalRows if evaluation has been executed
  const evalRow = evalRows?.find((r) => {
    const pipe = r.pipeline.toLowerCase();
    if (mode === "hybrid_rerank") return pipe.includes("rerank");
    if (mode === "hybrid") return pipe === "hybrid";
    if (mode === "vector") return pipe.includes("vector");
    if (mode === "bm25") return pipe.includes("bm25");
    return false;
  });

  const precisionVal = evalRow?.precisionAt5 ?? null;
  const recallVal = evalRow?.recallAt5 ?? null;
  const faithfulnessVal = evalRow?.faithfulness ?? null;
  const latencyVal = latestTurn?.latencyMs ?? (evalRow?.avgLatencyMs ?? null);

  const samplePrompts = [
    { title: "Architecture & RRF", query: "How does the hybrid search and Reciprocal Rank Fusion pipeline work?" },
    { title: "Chunking Trade-offs", query: "What document chunking strategies are supported and how do they compare?" },
    { title: "Cross-Encoder Re-rank", query: "Why is Cross-Encoder re-ranking applied over candidate retrieval?" },
  ];

  // Helper to parse citations inline next to generated claims
  const renderInlineCitations = (
    content: ReactNode,
    citations: SourceChunk[],
    activeId: string | null,
    onSelect: (id: string) => void
  ): ReactNode => {
    if (typeof content === "string") {
      // Matches [1], [2], [Chunk 1], or [Chunk 377998fc-3a44-4d6c-bd73-548d8e3bc861]
      const citationRegex = /(\[(?:Chunk\s+)?(?:[a-f0-9-]{8,}|[0-9]+)\])/gi;
      const parts = content.split(citationRegex);
      if (parts.length === 1) return content;

      return parts.map((part, i) => {
        const match = part.match(/\[(?:Chunk\s+)?([a-f0-9-]{8,}|[0-9]+)\]/i);
        if (match) {
          const rawToken = match[1].trim();
          let matchedChunk: SourceChunk | undefined;

          if (/^\d+$/.test(rawToken)) {
            const rank = parseInt(rawToken, 10);
            matchedChunk = citations.find((c) => (c.rank || 1) === rank) || citations[rank - 1];
          } else {
            // UUID or ID match
            matchedChunk = citations.find(
              (c) => c.id === rawToken || c.chunk_id === rawToken || c.id.startsWith(rawToken)
            );
          }

          if (matchedChunk) {
            return (
              <CitationChip
                key={`cite-${i}`}
                chunk={matchedChunk}
                onSelect={onSelect}
                isHighlighted={activeId === matchedChunk.id}
              />
            );
          }
        }
        return part;
      });
    }
    if (Array.isArray(content)) {
      return content.map((item, i) => (
        <span key={i}>{renderInlineCitations(item, citations, activeId, onSelect)}</span>
      ));
    }
    return content;
  };

  return (
    <div className="flex h-screen w-full overflow-hidden bg-ink-950 text-mist-100 font-body select-none">
      {/* ========================================================================= */}
      {/* COLUMN 1 (LEFT): Knowledge Base + Document Ingestion + Retrieval Strategy */}
      {/* ========================================================================= */}
      <aside className="flex h-full w-[310px] shrink-0 flex-col border-r border-ink-800 bg-ink-950/95 backdrop-blur-md">
        {/* Left Header */}
        <div className="flex items-center justify-between border-b border-ink-800/80 px-4 py-3.5">
          <div className="flex items-center gap-2">
            <Terminal className="h-4 w-4 text-lexical" />
            <span className="font-mono text-xs font-semibold uppercase tracking-wider text-mist-200">
              Knowledge Base
            </span>
          </div>
          <span className="rounded bg-ink-850 px-2 py-0.5 font-mono text-[10px] text-mist-400 border border-ink-800">
            {documents?.length || 0} Docs
          </span>
        </div>

        {/* Scrollable controls */}
        <div className="flex-1 space-y-4 overflow-y-auto p-4">
          {/* Section 1: Retrieval Mode Selector */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="flex items-center gap-1.5 font-mono text-[11px] font-semibold text-mist-300">
                <Flame className="h-3 w-3 text-lexical" />
                <span>Retrieval Pipeline Mode</span>
              </label>
              <span className="font-mono text-[10px] text-hybrid uppercase">Active</span>
            </div>
            <RetrievalModeToggle value={mode} onChange={setMode} vertical={true} />
          </div>

          {/* Section 2: Document Ingestion Dropzone */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="flex items-center gap-1.5 font-mono text-[11px] font-semibold text-mist-300">
                <UploadCloud className="h-3 w-3 text-vector" />
                <span>Upload & Index Document</span>
              </label>
              <select
                value={chunkStrategy}
                onChange={(e) => setChunkStrategy(e.target.value as ChunkingStrategy)}
                className="rounded border border-ink-750 bg-ink-900 px-1.5 py-0.5 font-mono text-[10px] text-mist-300 focus:border-lexical focus:outline-none cursor-pointer"
                title="Chunking Strategy"
              >
                {chunkingStrategies.map((s) => (
                  <option key={s} value={s}>
                    {s} chunk
                  </option>
                ))}
              </select>
            </div>

            <div
              {...getRootProps()}
              className={`group flex flex-col items-center justify-center rounded-lg border border-dashed p-4 text-center transition-all cursor-pointer ${
                isDragActive
                  ? "border-lexical bg-lexical/10 shadow-glow-lexical"
                  : "border-ink-750 bg-ink-900/60 hover:border-vector/50 hover:bg-ink-900"
              }`}
            >
              <input {...getInputProps()} />
              <div className="mb-2 flex h-8 w-8 items-center justify-center rounded-full bg-ink-800 text-mist-300 group-hover:text-vector transition-colors">
                <UploadCloud className="h-4 w-4" />
              </div>
              <p className="font-mono text-[11px] font-medium text-mist-200">
                {isDragActive ? "Drop document to index..." : "Drag & drop files or click"}
              </p>
              <p className="mt-0.5 font-mono text-[10px] text-mist-400">
                PDF, Markdown, HTML, TXT
              </p>
              {ingest.isPending && (
                <div className="mt-2 flex items-center gap-1.5 font-mono text-[10px] text-lexical animate-pulse">
                  <span className="h-1.5 w-1.5 rounded-full bg-lexical" />
                  Embedding & Indexing chunks...
                </div>
              )}
            </div>
          </div>

          {/* Section 3: Document Filter & Library Items */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="flex items-center gap-1.5 font-mono text-[11px] font-semibold text-mist-300">
                <Filter className="h-3 w-3 text-mist-400" />
                <span>Scope: Filter Retrieval</span>
              </label>
              {selectedDocId && (
                <button
                  onClick={() => setSelectedDocId("")}
                  className="font-mono text-[10px] text-lexical hover:underline cursor-pointer"
                >
                  Reset Filter
                </button>
              )}
            </div>

            <div className="space-y-1.5">
              <button
                type="button"
                onClick={() => setSelectedDocId("")}
                className={`w-full flex items-center justify-between rounded-md px-2.5 py-2 font-mono text-xs transition-all border cursor-pointer ${
                  selectedDocId === ""
                    ? "border-vector/40 bg-vector/10 text-vector font-semibold shadow-sm"
                    : "border-transparent text-mist-400 hover:bg-ink-900 hover:text-mist-200"
                }`}
              >
                <div className="flex items-center gap-2 truncate">
                  <Globe className="h-3.5 w-3.5 shrink-0" />
                  <span className="truncate">All Indexed Corpus</span>
                </div>
                <span className="rounded bg-ink-850 px-1.5 py-0.2 text-[10px] text-mist-400">
                  {documents?.length || 0}
                </span>
              </button>

              {docsLoading && (
                <div className="py-4 text-center font-mono text-[11px] text-mist-400 animate-pulse">
                  Loading indexed files...
                </div>
              )}

              {documents?.map((doc) => {
                const isSelected = selectedDocId === doc.id;
                const fileType = doc.fileType || "txt";
                const typeStyle = fileTypeIconMap[fileType] || fileTypeIconMap.txt;
                const TypeIcon = typeStyle.icon;

                return (
                  <div
                    key={doc.id}
                    onClick={() => setSelectedDocId(isSelected ? "" : doc.id)}
                    className={`group relative flex items-center justify-between rounded-md p-2.5 transition-all border cursor-pointer ${
                      isSelected
                        ? "border-lexical/50 bg-lexical/10 text-mist-100 shadow-sm"
                        : "border-ink-800/80 bg-ink-900/60 text-mist-300 hover:border-ink-700 hover:bg-ink-850"
                    }`}
                  >
                    <div className="flex items-center gap-2 min-w-0 flex-1">
                      <div
                        className={`flex h-6 w-6 shrink-0 items-center justify-center rounded border ${typeStyle.color}`}
                      >
                        <TypeIcon className="h-3 w-3" />
                      </div>
                      <div className="min-w-0 flex-1">
                        <p className="truncate font-mono text-[11px] font-medium text-mist-200" title={doc.filename}>
                          {doc.filename}
                        </p>
                        <div className="flex items-center gap-2 font-mono text-[9px] text-mist-400">
                          <span>{doc.chunkCount} partitions</span>
                          <span>&middot;</span>
                          <span className="uppercase">{doc.chunkingStrategy || "sentence"}</span>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          setInspectDoc(doc);
                        }}
                        title="Inspect Chunk Partitions"
                        className="rounded p-1 text-mist-400 hover:bg-ink-800 hover:text-vector transition-colors"
                      >
                        <Eye className="h-3 w-3" />
                      </button>
                      <button
                        type="button"
                        onClick={(e) => handleDeleteDoc(doc.id, doc.filename, e)}
                        title="Delete Document"
                        className="rounded p-1 text-mist-400 hover:bg-ink-800 hover:text-rose-400 transition-colors"
                      >
                        <Trash2 className="h-3 w-3" />
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </aside>

      {/* ========================================================================= */}
      {/* COLUMN 2 (CENTER): Terminal QA + Grounded Answer + Inline Claims + Trace */}
      {/* ========================================================================= */}
      <main className="flex flex-1 flex-col h-full min-w-0 bg-ink-950/60">
        {/* Center Header */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-ink-800/80 bg-ink-950/90 px-6 py-3.5 backdrop-blur-md">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-ok shadow-[0_0_8px_rgba(52,211,153,0.8)]" />
              <span className="font-mono text-xs font-semibold text-mist-100 uppercase tracking-wide">
                Grounding QA Console
              </span>
            </div>
            <span className="font-mono text-[10px] text-mist-400">|</span>
            <div className="flex items-center gap-1.5 font-mono text-[11px] text-mist-300">
              <span className="text-mist-400">Mode:</span>
              <span className="rounded bg-hybrid/15 px-2 py-0.5 uppercase tracking-wider text-[10px] text-hybrid border border-hybrid/30 font-semibold">
                {mode}
              </span>
            </div>
            {selectedDocId && (
              <span className="flex items-center gap-1 rounded bg-lexical/10 px-2 py-0.5 font-mono text-[10px] text-lexical border border-lexical/30">
                <Filter className="h-2.5 w-2.5" /> Single Doc Filter
              </span>
            )}
          </div>

          <div className="flex items-center gap-3">
            {turns.length > 0 && (
              <button
                type="button"
                onClick={() => {
                  setTurns([]);
                  setSelectedChunkId(null);
                }}
                className="flex items-center gap-1 rounded-md border border-ink-750 bg-ink-900 px-2.5 py-1 font-mono text-[11px] text-mist-400 hover:border-ink-600 hover:text-mist-200 transition-colors cursor-pointer"
                title="Clear conversational session"
              >
                <RotateCcw className="h-3 w-3" />
                <span>Reset</span>
              </button>
            )}
            <div className="hidden sm:flex items-center gap-1 font-mono text-[10px] text-mist-400">
              <kbd className="rounded border border-ink-700 bg-ink-850 px-1 py-0.2 text-[9px] text-mist-300">/</kbd>
              <span>focus</span>
            </div>
          </div>
        </div>

        {/* Message Stream */}
        <div className="flex-1 space-y-6 overflow-y-auto px-6 py-6">
          {turns.length === 0 && !ask.isPending && (
            <div className="mx-auto mt-8 max-w-2xl text-center animate-fadeIn select-text">
              <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl border border-lexical/40 bg-gradient-to-br from-lexical/20 via-vector/10 to-hybrid/20 shadow-glow-lexical">
                <Sparkles className="h-6 w-6 text-lexical" />
              </div>
              <h2 className="font-display text-lg font-semibold text-mist-100 tracking-tight">
                Production Grounded Q&A Console
              </h2>
              <p className="mt-1 font-body text-xs text-mist-400 max-w-lg mx-auto leading-relaxed">
                Query enterprise documents across Dense Vector Embeddings (pgvector), Sparse BM25 Keyword Indexes, and Cross-Encoder Re-ranking with grounded citations.
              </p>

              {/* Sample Prompts */}
              <div className="mt-6 grid grid-cols-1 gap-2.5 sm:grid-cols-3 text-left">
                {samplePrompts.map((sample, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => submit(sample.query)}
                    className="group flex flex-col justify-between rounded-lg border border-ink-800 bg-ink-900/80 p-3 transition-all hover:border-lexical/60 hover:bg-ink-850/90 hover:shadow-card cursor-pointer"
                  >
                    <div className="flex items-center gap-1.5 font-mono text-[10px] font-semibold uppercase tracking-wider text-lexical group-hover:text-lexical">
                      <FileText className="h-3 w-3" />
                      <span>{sample.title}</span>
                    </div>
                    <p className="mt-2 font-body text-xs text-mist-300 line-clamp-2 leading-relaxed">
                      "{sample.query}"
                    </p>
                  </button>
                ))}
              </div>
            </div>
          )}

          {turns.map((turn, i) => (
            <div key={i} className="space-y-4 animate-fadeIn select-text">
              {/* User Question */}
              <div className="flex items-start justify-end gap-3">
                <div className="max-w-2xl flex flex-col items-end gap-1">
                  <div className="rounded-lg border border-ink-700 bg-ink-850 px-4 py-3 font-mono text-xs text-mist-100 shadow-sm leading-relaxed">
                    <span className="text-lexical font-bold mr-2">&gt;</span>
                    {turn.question}
                  </div>
                  {turn.rewrittenQuery &&
                    turn.rewrittenQuery.trim().toLowerCase() !== turn.question.trim().toLowerCase() && (
                      <div className="flex items-center gap-1.5 font-mono text-[10px] text-mist-400 bg-ink-900/80 border border-ink-800 rounded px-2 py-0.5">
                        <Sparkles className="h-2.5 w-2.5 text-lexical" />
                        <span>Rewritten for retrieval: <span className="text-mist-200 italic">"{turn.rewrittenQuery}"</span></span>
                      </div>
                    )}
                </div>
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-ink-800 border border-ink-700 text-mist-300">
                  <User className="h-4 w-4" />
                </div>
              </div>

              {/* Assistant Answer Card */}
              <div className="flex items-start gap-3">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-gradient-to-br from-lexical/20 via-vector/20 to-hybrid/20 border border-ink-700 text-lexical shadow-glow-lexical">
                  <Bot className="h-4 w-4" />
                </div>
                <div className="flex-1 space-y-3 min-w-0">
                  <div className="relative rounded-lg border border-ink-750 bg-ink-900/95 p-5 font-body text-sm leading-relaxed text-mist-200 shadow-card">
                    {/* Top Bar inside card */}
                    <div className="mb-3 flex items-center justify-between border-b border-ink-800 pb-2.5 font-mono text-[11px]">
                      <div className="flex items-center gap-2 text-mist-400">
                        {turn.groundingStatus === "grounded" ? (
                          <span className="flex items-center gap-1 text-ok font-semibold">
                            <ShieldCheck className="h-3.5 w-3.5 text-ok" /> Verified Grounded
                          </span>
                        ) : turn.groundingStatus === "citation_mismatch" ? (
                          <span className="flex items-center gap-1 text-amber-400 font-semibold">
                            <AlertTriangle className="h-3.5 w-3.5 text-amber-400" /> Citation Warning
                          </span>
                        ) : turn.groundingStatus === "insufficient_evidence" || turn.groundingStatus === "refusal" ? (
                          <span className="flex items-center gap-1 text-rose-400 font-semibold">
                            <ShieldAlert className="h-3.5 w-3.5 text-rose-400" /> Insufficient Evidence
                          </span>
                        ) : (
                          <span className="flex items-center gap-1 text-vector font-semibold">
                            <ShieldCheck className="h-3.5 w-3.5" /> Grounded Synthesis
                          </span>
                        )}
                        <span>&middot;</span>
                        <span className="text-mist-400">{turn.modelName || "Gemini 1.5 Flash"}</span>
                      </div>
                      <button
                        type="button"
                        onClick={() => handleCopyAnswer(i, turn.answer)}
                        className="flex items-center gap-1 rounded border border-ink-800 bg-ink-850 px-2 py-0.5 text-mist-400 hover:border-ink-700 hover:text-mist-200 transition-colors cursor-pointer"
                        title="Copy markdown text"
                      >
                        {copiedTurnIdx === i ? (
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

                    {/* Markdown Content with Citations attached directly to claims */}
                    <div className="leading-relaxed text-mist-200">
                      <ReactMarkdown
                        components={{
                          h1: ({ children }) => (
                            <h1 className="text-base font-bold text-mist-100 mt-4 mb-2.5 border-b border-ink-800 pb-1 flex items-center gap-1.5">
                              {children}
                            </h1>
                          ),
                          h2: ({ children }) => (
                            <h2 className="text-sm font-bold text-mist-100 mt-3.5 mb-2 flex items-center gap-1.5 text-lexical">
                              {children}
                            </h2>
                          ),
                          h3: ({ children }) => (
                            <h3 className="text-xs font-semibold text-vector uppercase tracking-wider mt-3 mb-1.5">
                              {children}
                            </h3>
                          ),
                          p: ({ children }) => (
                            <p className="mb-3.5 last:mb-0 leading-relaxed text-mist-200 text-[13px]">
                              {renderInlineCitations(children, turn.citations, selectedChunkId, handleSelectChunk)}
                            </p>
                          ),
                          ul: ({ children }) => (
                            <ul className="mb-4 space-y-2.5 pl-4 list-disc text-mist-200">{children}</ul>
                          ),
                          ol: ({ children }) => (
                            <ol className="mb-4 space-y-2.5 pl-4 list-decimal text-mist-200">{children}</ol>
                          ),
                          li: ({ children }) => (
                            <li className="leading-relaxed pl-1 text-[13px] text-mist-200">
                              {renderInlineCitations(children, turn.citations, selectedChunkId, handleSelectChunk)}
                            </li>
                          ),
                          strong: ({ children }) => (
                            <strong className="font-semibold text-mist-100 text-lexical/90">{children}</strong>
                          ),
                          code: ({ children }) => (
                            <code className="rounded bg-ink-800 px-1.5 py-0.5 font-mono text-xs text-vector border border-ink-700/60">
                              {children}
                            </code>
                          ),
                          pre: ({ children }) => (
                            <pre className="mb-3.5 overflow-x-auto rounded-md border border-ink-700 bg-ink-950 p-3.5 font-mono text-xs text-mist-200">
                              {children}
                            </pre>
                          ),
                        }}
                      >
                        {turn.answer}
                      </ReactMarkdown>
                    </div>

                    {/* Bottom Grounded Citations Bar */}
                    {turn.citations && turn.citations.length > 0 && (
                      <div className="mt-4 flex flex-wrap items-center gap-1.5 border-t border-ink-800/80 pt-3">
                        <span className="font-mono text-[11px] text-mist-400 mr-1 flex items-center gap-1">
                          <FileText className="h-3 w-3" /> Grounded Source Citations:
                        </span>
                        {turn.citations.map((c, idx) => (
                          <CitationChip
                            key={c.id || idx}
                            chunk={c}
                            onSelect={handleSelectChunk}
                            isHighlighted={selectedChunkId === c.id}
                          />
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Retrieval Trace Breakdown */}
                  {turn.retrievedChunks && turn.retrievedChunks.length > 0 && (
                    <RetrievalTrace
                      chunks={turn.retrievedChunks}
                      selectedChunkId={selectedChunkId}
                      onSelectChunk={handleSelectChunk}
                    />
                  )}

                  {/* Latency and Pipeline Badges */}
                  <div className="flex flex-wrap items-center gap-2 font-mono text-[11px] text-mist-400">
                    <span className="rounded bg-ink-850 px-2 py-0.5 uppercase tracking-wider text-[10px] text-mist-300 border border-ink-800 font-semibold">
                      {turn.mode}
                    </span>
                    <span>&middot;</span>
                    <span className="flex items-center gap-1 text-mist-300">
                      <Clock className="h-3 w-3 text-mist-400" />
                      {turn.latencyMs}ms
                    </span>
                    {turn.latencyDetails && (
                      <span className="text-mist-400/80">
                        (retrieval: {turn.latencyDetails.retrieval_ms || 0}ms &middot; gen:{" "}
                        {turn.latencyDetails.generation_ms || 0}ms)
                      </span>
                    )}
                    {turn.cached && (
                      <>
                        <span>&middot;</span>
                        <span className="flex items-center gap-1 text-ok font-medium">
                          <Zap className="h-3 w-3 text-ok" /> Cached
                        </span>
                      </>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ))}

          {/* Loading Indicator with Cancel */}
          {ask.isPending && (
            <div className="flex items-center justify-between rounded-lg border border-ink-700 bg-ink-900/90 p-4 font-mono text-xs text-mist-300 shadow-card animate-fadeIn">
              <div className="flex items-center gap-3">
                <span className="h-2.5 w-2.5 animate-ping rounded-full bg-lexical" />
                <span>Executing multi-stage retrieval, cross-encoder re-ranking & LLM synthesis...</span>
              </div>
              <button
                type="button"
                onClick={handleCancel}
                className="rounded border border-rose-500/40 bg-rose-500/10 px-3 py-1 text-rose-400 text-xs font-semibold hover:bg-rose-500/20 transition-colors cursor-pointer"
              >
                Cancel Query
              </button>
            </div>
          )}

          {ask.isError && (
            <div className="rounded-lg border border-rose-500/40 bg-rose-500/10 p-4 font-mono text-xs text-rose-400 shadow-sm">
              {ask.error instanceof Error && ask.error.name === "AbortError"
                ? "Query was cancelled by user."
                : (ask.error as Error).message || "Query failed. Ensure FastAPI backend is running on port 8000."}
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="border-t border-ink-800/80 bg-ink-950/90 p-4 backdrop-blur-md">
          <div className="relative flex gap-2">
            <div className="relative flex-1">
              <span className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 font-mono text-sm font-bold text-lexical">
                &gt;
              </span>
              <input
                ref={inputRef}
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && submit()}
                placeholder="Ask technical question across indexed corpus (pgvector, BM25, Cross-Encoder re-rank)..."
                className="w-full rounded-md border border-ink-750 bg-ink-900/90 pl-8 pr-10 py-3 font-mono text-xs text-mist-100 placeholder:text-mist-400 focus:border-lexical focus:outline-none focus:ring-1 focus:ring-lexical/50 transition-all shadow-inner"
              />
              <span className="pointer-events-none absolute right-3.5 top-1/2 -translate-y-1/2 rounded border border-ink-700 bg-ink-800 px-1.5 py-0.5 font-mono text-[10px] text-mist-400">
                /
              </span>
            </div>
            <button
              type="button"
              onClick={() => submit()}
              disabled={ask.isPending || !question.trim()}
              className="flex items-center gap-2 rounded-md bg-gradient-to-r from-lexical to-amber-500 px-5 py-3 font-mono text-xs font-bold text-ink-950 transition-all hover:brightness-110 disabled:opacity-40 cursor-pointer shadow-glow-lexical"
            >
              <span>Execute</span>
              <Send className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      </main>

      {/* ========================================================================= */}
      {/* COLUMN 3 (RIGHT): Retrieved Sources + Live Quality & Telemetry Metrics   */}
      {/* ========================================================================= */}
      <aside className="flex h-full w-[360px] shrink-0 flex-col border-l border-ink-800 bg-ink-950/95 backdrop-blur-md">
        {/* Right Header */}
        <div className="flex items-center justify-between border-b border-ink-800/80 px-4 py-3.5">
          <div className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-vector" />
            <span className="font-mono text-xs font-semibold uppercase tracking-wider text-mist-200">
              Live Sources & Metrics
            </span>
          </div>
          <span className="font-mono text-[10px] text-vector flex items-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-vector animate-ping" />
            Active Trace
          </span>
        </div>

        {/* Scrollable inspection and metrics */}
        <div className="flex-1 space-y-4 overflow-y-auto p-4" ref={chunkListRef}>
          {/* Quality & Benchmark Metric Cards */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="flex items-center gap-1.5 font-mono text-[11px] font-semibold text-mist-300">
                <Target className="h-3 w-3 text-lexical" />
                <span>RAG Triad & Latency Cards</span>
              </label>
              <span className="font-mono text-[10px] text-mist-400">Benchmark</span>
            </div>

            {/* 2x2 Metric Grid */}
            <div className="grid grid-cols-2 gap-2">
              {/* P@5 Card */}
              <div className="rounded-lg border border-ink-800 bg-ink-900/80 p-3 shadow-subtle">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[10px] uppercase tracking-wider text-mist-400">P@5</span>
                  <Target className="h-3.5 w-3.5 text-lexical" />
                </div>
                <div className="mt-1 font-mono text-xl font-bold text-lexical">
                  {precisionVal != null ? `${(precisionVal * 100).toFixed(0)}%` : "--"}
                </div>
                <div className="mt-1 h-1 w-full rounded-full bg-ink-800 overflow-hidden">
                  <div
                    className="h-full bg-lexical rounded-full transition-all"
                    style={{ width: `${precisionVal != null ? precisionVal * 100 : 0}%` }}
                  />
                </div>
                <p className="mt-1 font-mono text-[9px] text-mist-400">Precision @ Top 5</p>
              </div>

              {/* R@5 Card */}
              <div className="rounded-lg border border-ink-800 bg-ink-900/80 p-3 shadow-subtle">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[10px] uppercase tracking-wider text-mist-400">R@5</span>
                  <Zap className="h-3.5 w-3.5 text-vector" />
                </div>
                <div className="mt-1 font-mono text-xl font-bold text-vector">
                  {recallVal != null ? `${(recallVal * 100).toFixed(0)}%` : "--"}
                </div>
                <div className="mt-1 h-1 w-full rounded-full bg-ink-800 overflow-hidden">
                  <div
                    className="h-full bg-vector rounded-full transition-all"
                    style={{ width: `${recallVal != null ? recallVal * 100 : 0}%` }}
                  />
                </div>
                <p className="mt-1 font-mono text-[9px] text-mist-400">Recall @ Top 5</p>
              </div>

              {/* Faithfulness Card */}
              <div className="rounded-lg border border-ink-800 bg-ink-900/80 p-3 shadow-subtle">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[10px] uppercase tracking-wider text-mist-400">Faithfulness</span>
                  <ShieldCheck className="h-3.5 w-3.5 text-hybrid" />
                </div>
                <div className="mt-1 font-mono text-xl font-bold text-hybrid">
                  {faithfulnessVal != null ? `${(faithfulnessVal * 100).toFixed(0)}%` : "--"}
                </div>
                <div className="mt-1 h-1 w-full rounded-full bg-ink-800 overflow-hidden">
                  <div
                    className="h-full bg-hybrid rounded-full transition-all"
                    style={{ width: `${faithfulnessVal != null ? faithfulnessVal * 100 : 0}%` }}
                  />
                </div>
                <p className="mt-1 font-mono text-[9px] text-mist-400">Grounded Factuality</p>
              </div>

              {/* Latency Card */}
              <div className="rounded-lg border border-ink-800 bg-ink-900/80 p-3 shadow-subtle">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[10px] uppercase tracking-wider text-mist-400">Latency</span>
                  <Clock className="h-3.5 w-3.5 text-ok" />
                </div>
                <div className="mt-1 font-mono text-xl font-bold text-mist-100">
                  {latencyVal != null ? `${Math.round(latencyVal)}ms` : "--"}
                </div>
                <div className="mt-1 flex items-center justify-between font-mono text-[9px] text-mist-400">
                  <span>{latestTurn?.cached ? "Cached Hit" : "Realtime"}</span>
                  <span className="text-ok">{latestTurn?.cached ? "0ms gen" : "FastAPI"}</span>
                </div>
                <p className="mt-1 font-mono text-[9px] text-mist-400">End-to-End Latency</p>
              </div>
            </div>

            {!evalRow && (
              <div className="rounded border border-dashed border-ink-800 bg-ink-900/40 p-2 text-center">
                <p className="font-mono text-[10px] text-mist-400">
                  Baseline metrics appear after running an evaluation in the Eval dashboard.
                </p>
              </div>
            )}
          </div>

          {/* Section: Retrieved Sources Inspector */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="flex items-center gap-1.5 font-mono text-[11px] font-semibold text-mist-300">
                <Layers className="h-3 w-3 text-hybrid" />
                <span>Retrieved Candidate Chunks</span>
              </label>
              <span className="font-mono text-[10px] text-mist-400">
                {activeSources.length} partitions
              </span>
            </div>

            {activeSources.length === 0 ? (
              <div className="rounded-lg border border-ink-800 bg-ink-900/50 p-6 text-center">
                <Layers className="mx-auto h-8 w-8 text-mist-400/50 mb-2" />
                <p className="font-mono text-xs text-mist-300">No Query Active</p>
                <p className="mt-1 font-mono text-[10px] text-mist-400">
                  Submit a question in the center console to inspect candidate chunks and similarity scores.
                </p>
              </div>
            ) : (
              <div className="space-y-2.5">
                {activeSources.map((chunk, idx) => {
                  const isSelected = selectedChunkId === chunk.id;
                  const sig = chunk.signal || "vector";
                  const scoreVal = typeof chunk.score === "number" ? chunk.score.toFixed(3) : "1.000";
                  const isCopied = copiedChunkId === chunk.id;
                  const approxTokens = Math.round(chunk.text.split(/\s+/).length * 1.3);

                  const signalBadge =
                    sig === "bm25"
                      ? "border-lexical/40 bg-lexical/10 text-lexical"
                      : sig === "vector"
                      ? "border-vector/40 bg-vector/10 text-vector"
                      : "border-hybrid/40 bg-hybrid/10 text-hybrid";

                  return (
                    <div
                      key={chunk.id || idx}
                      id={`source-chunk-${chunk.id}`}
                      onClick={() => setSelectedChunkId(chunk.id)}
                      className={`group rounded-lg border p-3 transition-all cursor-pointer ${
                        isSelected
                          ? "border-mist-100/90 bg-ink-850 shadow-card ring-1 ring-mist-200/40"
                          : "border-ink-800 bg-ink-900/80 hover:border-ink-700 hover:bg-ink-850/80"
                      }`}
                    >
                      {/* Top Chunk Header */}
                      <div className="flex items-center justify-between border-b border-ink-850 pb-2 mb-2">
                        <div className="flex items-center gap-1.5 font-mono text-xs">
                          <span className="font-bold text-mist-100">
                            #{chunk.rank || idx + 1}
                          </span>
                          <span
                            className={`rounded px-1.5 py-0.2 font-mono text-[9px] uppercase tracking-wider font-semibold border ${signalBadge}`}
                          >
                            {sig}
                          </span>
                          <span className="font-mono text-[10px] text-mist-400">
                            score: {scoreVal}
                          </span>
                        </div>
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleCopyChunk(chunk.id, chunk.text);
                          }}
                          className="font-mono text-[10px] text-mist-400 hover:text-mist-200 flex items-center gap-1"
                          title="Copy chunk text"
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

                      {/* File and partition meta */}
                      <div className="mb-2 flex items-center justify-between font-mono text-[10px] text-mist-400">
                        <span className="truncate max-w-[200px] text-mist-300 font-medium" title={chunk.documentName}>
                          {chunk.documentName || "Document"}
                        </span>
                        <span>
                          {chunk.page ? `p.${chunk.page} \u00b7 ` : ""}
                          ~{approxTokens} tokens
                        </span>
                      </div>

                      {/* Text Snippet Preview */}
                      <p className="font-mono text-[11px] leading-relaxed text-mist-300 whitespace-pre-wrap line-clamp-4 select-text">
                        {chunk.text}
                      </p>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* System Health & Telemetry Card (Teal) */}
          <div className="rounded-lg border border-ink-800 bg-ink-900/90 p-3.5 shadow-subtle space-y-2">
            <div className="flex items-center justify-between border-b border-ink-850 pb-2">
              <div className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-vector shadow-[0_0_8px_rgba(45,212,191,0.8)]" />
                <span className="font-mono text-xs font-semibold text-mist-100">
                  System Health & Engines
                </span>
              </div>
              <span className="font-mono text-[10px] text-vector uppercase font-semibold">
                {health?.status || "Online"}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2 font-mono text-[10px] text-mist-400">
              <div>
                <span className="block text-mist-400/80">Database:</span>
                <span className="text-mist-200 font-medium">{health?.database || "Postgres (pgvector)"}</span>
              </div>
              <div>
                <span className="block text-mist-400/80">Cache:</span>
                <span className="text-mist-200 font-medium">{health?.cache || "Redis"}</span>
              </div>
              <div>
                <span className="block text-mist-400/80">Embedding:</span>
                <span className="text-mist-200 font-medium">{health?.embedding_model || "bge-small-en-v1.5"}</span>
              </div>
              <div>
                <span className="block text-mist-400/80">LLM Provider:</span>
                <span className="text-mist-200 font-medium">{health?.llm_provider || "Gemini 1.5 Flash"}</span>
              </div>
            </div>
          </div>
        </div>
      </aside>

      {/* Drawer: Inspect All Partitions for a Document */}
      {inspectDoc && (
        <ChunkInspectorDrawer
          document={inspectDoc}
          onClose={() => setInspectDoc(null)}
        />
      )}
    </div>
  );
}
