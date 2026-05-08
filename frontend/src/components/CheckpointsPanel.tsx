"use client";

import { useEffect, useRef, useState } from "react";

import { api, CheckpointT } from "@/lib/api";
import { useUIStore } from "@/lib/store";

const POLL_MS = 12000;

export function CheckpointsPanel() {
  const { selectedProjectId } = useUIStore();
  const [items, setItems] = useState<CheckpointT[]>([]);
  const [pending, setPending] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  /** Avoid overlapping polls (stacked timeouts while prior fetch is slow). */
  const inFlight = useRef(false);
  /** Cancel timed poll loops when switching project or unmounting. */
  const gen = useRef(0);

  const refresh = async (opts?: { force?: boolean }) => {
    if (!selectedProjectId) {
      setItems([]);
      return;
    }
    if (opts?.force !== true && inFlight.current) return;
    inFlight.current = true;
    setLoading(true);
    try {
      const r = await api.listCheckpoints(selectedProjectId, {
        force: opts?.force,
      });
      setItems(r.checkpoints);
    } catch {
      setItems([]);
    } finally {
      setLoading(false);
      inFlight.current = false;
    }
  };

  useEffect(() => {
    gen.current += 1;
    const myGen = gen.current;
    const pollTimerRef = { id: undefined as ReturnType<typeof setTimeout> | undefined };
    void refresh({ force: true });

    const tick = async () => {
      if (gen.current !== myGen) return;
      if (typeof document !== "undefined" && !document.hidden && selectedProjectId) {
        await refresh();
      }
      if (gen.current !== myGen) return;
      pollTimerRef.id = window.setTimeout(tick, POLL_MS);
    };

    pollTimerRef.id = window.setTimeout(tick, POLL_MS);

    const vis = () => {
      if (!document.hidden && gen.current === myGen && selectedProjectId) {
        void refresh();
      }
    };
    document.addEventListener("visibilitychange", vis);

    return () => {
      if (pollTimerRef.id !== undefined) window.clearTimeout(pollTimerRef.id);
      gen.current += 1;
      document.removeEventListener("visibilitychange", vis);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedProjectId]);

  const resume = async (cpId: string) => {
    if (!selectedProjectId) return;
    setPending(cpId);
    try {
      await api.resumeFrom(selectedProjectId, cpId);
    } finally {
      setPending(null);
    }
  };

  return (
    <div className="rounded border border-border bg-surface">
      <div className="flex items-center justify-between border-b border-border px-3 py-2">
        <div className="text-sm font-medium">Checkpoints</div>
        <button
          onClick={() => void refresh({ force: true })}
          className="text-xs text-zinc-400 hover:text-zinc-200"
          disabled={loading}
        >
          {loading ? "…" : "Refresh"}
        </button>
      </div>
      <div className="flex max-h-[60vh] flex-col gap-2 overflow-y-auto p-3 scrollbar">
        {items.length === 0 && (
          <div className="text-xs text-zinc-500">
            No checkpoints yet. Start an agent run to populate.
          </div>
        )}
        {items.map((cp, i) => (
          <div
            key={cp.checkpoint_id ?? i}
            className="rounded border border-border bg-muted p-2 text-xs"
          >
            <div className="flex items-center justify-between gap-2">
              <div className="text-zinc-300">
                {cp.created_at ? new Date(cp.created_at).toLocaleString() : "—"}
                {cp.step !== null && (
                  <span className="ml-2 text-zinc-500">step {cp.step}</span>
                )}
              </div>
              {cp.checkpoint_id && (
                <button
                  className="rounded border border-border px-2 py-1 text-zinc-300 hover:bg-surface disabled:opacity-50"
                  disabled={pending === cp.checkpoint_id}
                  onClick={() => resume(cp.checkpoint_id!)}
                >
                  {pending === cp.checkpoint_id ? "…" : "↶ Resume from here"}
                </button>
              )}
            </div>
            <div className="mt-1 text-zinc-400">
              {cp.wrote_nodes.length > 0 && (
                <span>
                  wrote:{" "}
                  <span className="text-zinc-200">{cp.wrote_nodes.join(", ")}</span>
                </span>
              )}
              {cp.next.length > 0 && (
                <span className="ml-2">
                  next:{" "}
                  <span className="text-zinc-200">{cp.next.join(", ")}</span>
                </span>
              )}
            </div>
            {cp.checkpoint_id && (
              <div className="mt-1 truncate font-mono text-[10px] text-zinc-600">
                id: {cp.checkpoint_id}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
