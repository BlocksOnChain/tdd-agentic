"use client";

import { useEffect, useRef, useState } from "react";

import { api } from "@/lib/api";
import { useUIStore } from "@/lib/store";

const colorFor = (agent: string) => {
  switch (agent) {
    case "project_manager":
      return "text-amber-400";
    case "researcher":
      return "text-cyan-400";
    case "backend_lead":
    case "frontend_lead":
      return "text-violet-400";
    case "backend_dev":
    case "frontend_dev":
    case "devops":
      return "text-emerald-400";
    case "qa":
      return "text-pink-400";
    default:
      return "text-zinc-400";
  }
};

export function AgentLog() {
  const { logs, clearLogs, selectedProjectId } = useUIStore();
  const [pending, setPending] = useState<string | null>(null);
  const scrollerRef = useRef<HTMLDivElement | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const [isAtBottom, setIsAtBottom] = useState(true);

  const recomputeIsAtBottom = () => {
    const el = scrollerRef.current;
    if (!el) return;
    const thresholdPx = 8;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    setIsAtBottom(distanceFromBottom <= thresholdPx);
  };

  const scrollToBottom = () => {
    bottomRef.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  };

  const resumeFrom = async (checkpointId: string) => {
    if (!selectedProjectId) return;
    setPending(checkpointId);
    try {
      await api.resumeFrom(selectedProjectId, checkpointId);
    } finally {
      setPending(null);
    }
  };

  useEffect(() => {
    if (isAtBottom) scrollToBottom();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [logs.length]);

  return (
    <div className="rounded border border-border bg-surface">
      <div className="flex items-center justify-between border-b border-border px-3 py-2">
        <div className="text-sm font-medium">Agent activity</div>
        <button onClick={clearLogs} className="text-xs text-zinc-400 hover:text-zinc-200">
          Clear
        </button>
      </div>
      <div className="relative">
        <div
          ref={scrollerRef}
          onScroll={recomputeIsAtBottom}
          className="max-h-[60vh] overflow-y-auto scrollbar p-3 font-mono text-xs leading-relaxed"
        >
          {logs.length === 0 && <div className="text-zinc-500">Waiting for events…</div>}
          {logs.map((l, i) => (
            <div
              key={l.id ?? `live-${l.ts}-${i}-${l.agent}`}
              className="group flex gap-2 hover:bg-muted/40 rounded px-1 py-0.5"
            >
              <span className="text-zinc-600 shrink-0">
                {new Date(l.ts * 1000).toLocaleTimeString()}
              </span>
              <span className={`shrink-0 ${colorFor(l.agent)}`}>{l.agent}</span>
              <span className="text-zinc-500 shrink-0">{l.kind}</span>
              <span className="text-zinc-300 break-all flex-1">{l.detail}</span>
              {l.checkpoint_id && selectedProjectId && (
                <button
                  title={`Resume from checkpoint ${l.checkpoint_id}`}
                  onClick={() => resumeFrom(l.checkpoint_id!)}
                  disabled={pending === l.checkpoint_id}
                  className="shrink-0 self-start rounded border border-border px-1.5 py-0.5 text-[10px] text-zinc-400 opacity-0 hover:bg-muted hover:text-zinc-200 group-hover:opacity-100 disabled:opacity-50"
                >
                  {pending === l.checkpoint_id ? "…" : "↶ resume"}
                </button>
              )}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>

        {!isAtBottom && logs.length > 0 && (
          <button
            type="button"
            onClick={scrollToBottom}
            title="Jump to latest"
            className="absolute bottom-3 right-3 rounded-full border border-border bg-surface/90 px-2 py-1 text-xs text-zinc-200 shadow-sm backdrop-blur hover:bg-surface"
          >
            ↓
          </button>
        )}
      </div>
    </div>
  );
}
