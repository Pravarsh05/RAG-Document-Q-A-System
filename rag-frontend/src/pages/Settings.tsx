import {
  Cpu,
  Bot,
  Zap,
  Database,
  Layers,
  Server,
  CheckCircle2,
} from "lucide-react";
import { useConfig, useUpdateConfig, useHealth } from "@/hooks/useApi";

export function SettingsPage() {
  const { data: config, isLoading: configLoading } = useConfig();
  const { data: health } = useHealth();
  const updateConfig = useUpdateConfig();

  const toggleCache = () => {
    if (!config) return;
    updateConfig.mutate({ cache_enabled: !config.cache_enabled });
  };

  const setEmbedding = (provider: "bge" | "openai") => {
    updateConfig.mutate({ embedding_provider: provider });
  };

  const setLLM = (provider: "anthropic" | "openai" | "mock") => {
    updateConfig.mutate({ llm_provider: provider });
  };

  return (
    <div className="mx-auto max-w-3xl px-6 py-8">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-ink-800 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="font-display text-lg font-semibold text-mist-100 tracking-tight">
              Settings & System Configuration
            </h1>
            <span className="rounded bg-ink-800 px-2 py-0.5 font-mono text-[10px] text-mist-300 border border-ink-700">
              Runtime Admin
            </span>
          </div>
          <p className="font-body text-xs text-mist-400">
            Configure embedding models, LLM synthesis providers, vector indices, and caching policies.
          </p>
        </div>
      </div>

      {configLoading ? (
        <div className="mt-8 py-12 text-center font-mono text-xs text-mist-400">
          <span className="inline-block h-2 w-2 animate-ping rounded-full bg-lexical mr-2" />
          Loading runtime settings...
        </div>
      ) : (
        <div className="mt-6 space-y-5">
          {/* Live System Health */}
          <Section icon={Server} title="Live Connectivity Status" hint="Current backend, database, and cache service health">
            <div className="flex flex-wrap items-center gap-3 font-mono text-xs">
              <span className="flex items-center gap-2 rounded-md border border-ok/30 bg-ok/10 px-3 py-1.5 text-ok">
                <span className="h-2 w-2 rounded-full bg-ok shadow-[0_0_6px_rgba(52,211,153,0.8)]" />
                API: {health?.status || "online"}
              </span>
              <span className="flex items-center gap-2 rounded-md border border-ink-700 bg-ink-900 px-3 py-1.5 text-mist-200">
                <Database className="h-3.5 w-3.5 text-mist-400" />
                DB: {health?.database || "connected"}
              </span>
              <span className="flex items-center gap-2 rounded-md border border-ink-700 bg-ink-900 px-3 py-1.5 text-mist-200">
                <Zap className="h-3.5 w-3.5 text-lexical" />
                Cache: {health?.cache || "active"}
              </span>
            </div>
          </Section>

          {/* Embedding Provider */}
          <Section icon={Cpu} title="Embedding Representation Model" hint="Select dense vector embedding model for indexing & query search">
            <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
              {(["bge", "openai"] as const).map((opt) => {
                const current = (config?.embedding_provider || "bge").toLowerCase();
                const isSelected = current.includes(opt);
                return (
                  <button
                    key={opt}
                    onClick={() => setEmbedding(opt)}
                    className={`flex items-center justify-between rounded-lg border p-3 font-mono text-xs text-left transition-all cursor-pointer ${
                      isSelected
                        ? "border-lexical bg-lexical/10 text-mist-100 shadow-glow-lexical"
                        : "border-ink-700 bg-ink-900 text-mist-300 hover:border-ink-600 hover:bg-ink-850"
                    }`}
                  >
                    <div>
                      <div className="font-semibold text-mist-100">
                        {opt === "bge" ? "Local BAAI/bge-small-en" : "OpenAI text-embedding-3-small"}
                      </div>
                      <div className="mt-0.5 text-[11px] text-mist-400">
                        {opt === "bge" ? "384 dimensions &middot; Zero API key" : "1536 dimensions &middot; OpenAI API"}
                      </div>
                    </div>
                    {isSelected && <CheckCircle2 className="h-4 w-4 text-lexical" />}
                  </button>
                );
              })}
            </div>
          </Section>

          {/* LLM Provider */}
          <Section icon={Bot} title="LLM Answer Generation Provider" hint="Select model provider for grounded multi-chunk answer synthesis">
            <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-3">
              {(["anthropic", "openai", "mock"] as const).map((opt) => {
                const current = (config?.llm_provider || "mock").toLowerCase();
                const isSelected = current.includes(opt);
                return (
                  <button
                    key={opt}
                    onClick={() => setLLM(opt)}
                    className={`flex flex-col justify-between rounded-lg border p-3 font-mono text-xs text-left transition-all cursor-pointer ${
                      isSelected
                        ? "border-vector bg-vector/10 text-mist-100 shadow-glow-vector"
                        : "border-ink-700 bg-ink-900 text-mist-300 hover:border-ink-600 hover:bg-ink-850"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-semibold text-mist-100">
                        {opt === "anthropic" ? "Anthropic Claude" : opt === "openai" ? "OpenAI GPT" : "Extractive Local"}
                      </span>
                      {isSelected && <CheckCircle2 className="h-4 w-4 text-vector" />}
                    </div>
                    <span className="text-[10px] text-mist-400">
                      {opt === "anthropic" ? "Claude 3.5 Sonnet" : opt === "openai" ? "GPT-4o mini" : "Grounded Synthesizer"}
                    </span>
                  </button>
                );
              })}
            </div>
          </Section>

          {/* Query Cache */}
          <Section icon={Zap} title="Query Caching Policy" hint="Caches identical query parameters in Redis or in-memory for sub-millisecond responses">
            <div className="flex items-center justify-between">
              <div>
                <div className="font-body text-xs text-mist-200">
                  {config?.cache_enabled ? "Query Cache is currently Active" : "Query Cache is currently Disabled"}
                </div>
                <div className="mt-0.5 font-mono text-[11px] text-mist-400">
                  TTL: {config?.cache_ttl_seconds || 3600} seconds &middot; Invalidation on document ingestion
                </div>
              </div>
              <button
                onClick={toggleCache}
                disabled={updateConfig.isPending}
                className={`rounded-md border px-4 py-2 font-mono text-xs font-semibold transition-all cursor-pointer ${
                  config?.cache_enabled
                    ? "border-ok bg-ok/15 text-ok"
                    : "border-ink-700 bg-ink-850 text-mist-400 hover:border-ink-600 hover:text-mist-200"
                }`}
              >
                {config?.cache_enabled ? "Enabled" : "Disabled"}
              </button>
            </div>
          </Section>

          {/* Ingestion Defaults */}
          <Section icon={Layers} title="Ingestion & Chunking Defaults" hint="Default parameters applied when new documents are uploaded">
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 font-mono text-xs text-mist-300">
              <div className="rounded-md border border-ink-800 bg-ink-950 p-3">
                <span className="text-[10px] uppercase text-mist-400 block mb-1">Default Strategy</span>
                <span className="text-mist-100 font-medium capitalize">{config?.default_chunking_strategy || "sentence"} Chunking</span>
              </div>
              <div className="rounded-md border border-ink-800 bg-ink-950 p-3">
                <span className="text-[10px] uppercase text-mist-400 block mb-1">Partition Window</span>
                <span className="text-mist-100 font-medium">{config?.default_chunk_size || 500} tokens / {config?.default_chunk_overlap || 50} overlap</span>
              </div>
            </div>
          </Section>
        </div>
      )}
    </div>
  );
}

function Section({
  icon: Icon,
  title,
  hint,
  children,
}: {
  icon: typeof Server;
  title: string;
  hint: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-ink-800 bg-ink-900/70 p-5 shadow-card">
      <div className="mb-3 flex items-start gap-3 border-b border-ink-800 pb-3">
        <div className="flex h-7 w-7 items-center justify-center rounded bg-ink-800 border border-ink-700 text-lexical shrink-0">
          <Icon className="h-4 w-4" />
        </div>
        <div>
          <div className="font-display text-sm font-semibold text-mist-100">{title}</div>
          <div className="font-body text-xs text-mist-400">{hint}</div>
        </div>
      </div>
      {children}
    </div>
  );
}
