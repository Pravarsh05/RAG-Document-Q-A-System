import clsx from "clsx";
import type { IngestionStatus } from "@/types/api";

const styles: Record<IngestionStatus, string> = {
  ready: "text-ok border-ok/40",
  processing: "text-lexical border-lexical/40",
  failed: "text-err border-err/40",
};

export function StatusPill({ status }: { status: IngestionStatus }) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 rounded-sm border px-2 py-0.5 font-mono text-[11px]",
        styles[status]
      )}
    >
      <span
        className={clsx("h-1.5 w-1.5 rounded-full", {
          "bg-ok": status === "ready",
          "bg-lexical animate-pulse": status === "processing",
          "bg-err": status === "failed",
        })}
      />
      {status}
    </span>
  );
}
