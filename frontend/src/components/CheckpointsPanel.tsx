"use client";

import { useEffect, useState } from "react";

import { api, CheckpointT } from "@/lib/api";
import { useUIStore } from "@/lib/store";

export function CheckpointsPanel() {
  const { selectedProjectId } = useUIStore();
  const [items, setItems] = useState<CheckpointT[]>([]);
  const [pending, setPending] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const refresh = async () => {
    if (!selectedProjectId) {
      setItems([]);
      return;
    }
    setLoading(true);
    try {
      const r = await api.listCheckpoints(selectedProjectId);
      setItems(r.checkpoints);
    } catch {
      setItems([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
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
          onClick={refresh}
          className="text-xs text-zinc-400 hover:text-zinc-200"
          disabled={loading}
        >
          {loading ? "…" : "Refresh"}
        </button>
      </div>
      <div className="max-h-[60vh] overflow-y-auto scrollbar p-3 flex flex-col gap-2">
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
                  wrote: <span className="text-zinc-200">{cp.wrote_nodes.join(", ")}</span>
                </span>
              )}
              {cp.next.length > 0 && (
                <span className="ml-2">
                  next: <span className="text-zinc-200">{cp.next.join(", ")}</span>
                </span>
              )}
            </div>
            {cp.checkpoint_id && (
              <div className="mt-1 font-mono text-[10px] text-zinc-600 truncate">
                id: {cp.checkpoint_id}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
