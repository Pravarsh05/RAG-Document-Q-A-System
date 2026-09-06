import { NavLink } from "react-router-dom";
import clsx from "clsx";
import { MessageSquareCode, FolderKanban, LineChart, SlidersHorizontal, Layers, Sparkles } from "lucide-react";
import { useHealth } from "@/hooks/useApi";

const links = [
  { to: "/", label: "Query", icon: MessageSquareCode, badge: "Hybrid" },
  { to: "/library", label: "Library", icon: FolderKanban },
  { to: "/eval", label: "Benchmarks", icon: LineChart },
  { to: "/settings", label: "Settings", icon: SlidersHorizontal },
];

export function Sidebar() {
  const { data: health, isError, isLoading } = useHealth();

  const statusColor = isError ? "bg-err" : (isLoading ? "bg-lexical animate-pulse" : "bg-ok shadow-[0_0_8px_rgba(52,211,153,0.6)]");
  const statusText = isError ? "API Offline" : (isLoading ? "Connecting..." : "FastAPI Online");

  return (
    <aside className="flex h-screen w-60 shrink-0 flex-col border-r border-ink-700/80 bg-ink-950/95 backdrop-blur-md">
      {/* Brand Header */}
      <div className="flex items-center gap-3 px-5 py-5 border-b border-ink-850">
        <div className="flex h-9 w-9 items-center justify-center rounded-md bg-gradient-to-br from-lexical/20 to-vector/20 border border-ink-700 shadow-subtle">
          <Layers className="h-5 w-5 text-lexical" />
        </div>
        <div>
          <div className="flex items-center gap-1.5">
            <span className="font-display text-sm font-semibold text-mist-100 tracking-tight">RAG Console</span>
            <span className="rounded bg-ink-800 px-1 py-0.2 font-mono text-[9px] font-medium text-vector border border-ink-700">v1.0</span>
          </div>
          <div className="font-mono text-[10px] text-mist-400">pgvector + BM25</div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex flex-col gap-1.5 px-3 py-4">
        {links.map((link) => {
          const Icon = link.icon;
          return (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.to === "/"}
              className={({ isActive }) =>
                clsx(
                  "group flex items-center justify-between rounded-md px-3 py-2.5 font-body text-xs transition-all",
                  isActive
                    ? "bg-gradient-to-r from-ink-800 to-ink-850 text-mist-100 font-medium border-l-2 border-lexical shadow-sm"
                    : "text-mist-300 hover:bg-ink-900 hover:text-mist-100"
                )
              }
            >
              <div className="flex items-center gap-3">
                <Icon className="h-4 w-4 text-mist-400 group-hover:text-lexical transition-colors" />
                <span>{link.label}</span>
              </div>
              {link.badge && (
                <span className="rounded bg-hybrid/10 px-1.5 py-0.5 font-mono text-[10px] text-hybrid border border-hybrid/20">
                  {link.badge}
                </span>
              )}
            </NavLink>
          );
        })}
      </nav>

      {/* Health Status Card in Footer */}
      <div className="mt-auto p-3">
        <div className="rounded-md border border-ink-800 bg-ink-900/90 p-3 shadow-subtle">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className={clsx("h-2 w-2 rounded-full shrink-0", statusColor)} />
              <span className="font-mono text-xs font-medium text-mist-200">{statusText}</span>
            </div>
            <Sparkles className="h-3.5 w-3.5 text-mist-400" />
          </div>
          <div className="mt-2 space-y-1 border-t border-ink-800 pt-2 font-mono text-[10px] text-mist-400">
            <div className="flex justify-between">
              <span>Embedding:</span>
              <span className="text-mist-300">{health?.embedding_provider || "bge"}</span>
            </div>
            <div className="flex justify-between">
              <span>LLM:</span>
              <span className="text-mist-300">{health?.llm_provider || "local"}</span>
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
}

