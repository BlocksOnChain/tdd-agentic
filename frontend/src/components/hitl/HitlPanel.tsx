"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { api } from "@/lib/api";
import { useUIStore } from "@/lib/store";

type InterruptItem = {
  ticket_id?: string;
  question: string;
  asked_by?: string;
  receivedAt: number;
  kind?: string;
  dismissed?: boolean;
  answered?: boolean;
  answer?: string;
  id: string;
};

// Color palette for interrupt types
const KIND_COLORS: Record<string, { pill: string; border: string; bg: string }> = {
  ask_human: {
    pill: "bg-amber-400/20 text-amber-300 border-amber-400/30",
    border: "border-amber-500/50",
    bg: "bg-amber-950/30",
  },
  question: {
    pill: "bg-cyan-400/20 text-cyan-300 border-cyan-400/30",
    border: "border-cyan-500/50",
    bg: "bg-cyan-950/30",
  },
};

const DEFAULT_COLORS = {
  pill: "bg-zinc-400/20 text-zinc-300 border-zinc-400/30",
  border: "border-zinc-500/50",
  bg: "bg-zinc-900/30",
};

function formatRelativeTime(ts: number): string {
  const diff = Date.now() - ts;
  const mins = Math.floor(diff / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

export function HitlPanel() {
  const {
    selectedProjectId,
    interrupts: wsInterrupts,
    pushInterrupt,
    popInterrupt,
    appendLog,
  } = useUIStore();

  // Local state for pending interrupts (mirrors wsInterrupts + local actions)
  const [localInterrups, setLocalInterrups] = useState<InterruptItem[]>([]);
  const [drafts, setDrafts] = useState<Record<number, string>>({});
  const [isLoading, setIsLoading] = useState(false);

  const textAreasRef = useRef<(HTMLTextAreaElement | null)[]>([]);

  // Sync WebSocket interrupts into local state
  useEffect(() => {
    setLocalInterrups((prev) => {
      const newOnes = wsInterrupts.filter(
        (wi) => !prev.some((p) => p.receivedAt === wi.receivedAt),
      );
      if (newOnes.length === 0) return prev;
      return [
        ...prev,
        ...newOnes.map((wi, i) => ({
          ...wi,
          id: `${wi.receivedAt}-${i}`,
          dismissed: false,
          answered: false,
        })),
      ];
    });
  }, [wsInterrupts]);

  // Fetch historical pending interrupts from backend on mount/project change
  useEffect(() => {
    if (!selectedProjectId) return;
    let cancelled = false;
    const fetchPending = async () => {
      setIsLoading(true);
      try {
        const res = await api.listInterrupts(selectedProjectId);
        const backendInterrupts = (res.interrupts || []) as Array<{
          kind?: string;
          question?: string;
          ticket_id?: string;
          asked_by?: string;
        }>;
        if (cancelled) return;
        // Only add if we have none yet (avoid duplicates with WS)
        if (backendInterrupts.length > 0 && localInterrups.length === 0) {
          const mapped = backendInterrupts.map((bi, i) => ({
            ticket_id: bi.ticket_id,
            question: bi.question ?? "(no question text)",
            asked_by: bi.asked_by ?? "system",
            receivedAt: Date.now() + i, // ensure ordering
            kind: bi.kind ?? "ask_human",
            dismissed: false,
            answered: false,
            id: `backend-${i}`,
          }));
          setLocalInterrups(mapped);
        }
      } catch {
        // Backend may not have interrupts endpoint yet — ignore
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    };
    fetchPending();
    return () => {
      cancelled = true;
    };
  }, [selectedProjectId]);

  const dismissInterrupt = useCallback(
    (idx: number) => {
      setLocalInterrups((prev) => prev.filter((_, i) => i !== idx));
      setDrafts((d) => {
        const next = { ...d };
        delete next[idx];
        return next;
      });
      // Also remove from WS store
      popInterrupt(idx);
    },
    [popInterrupt],
  );

  const submit = useCallback(
    async (idx: number) => {
      const i = localInterrups[idx];
      if (!i || i.dismissed || i.answered) return;
      const value = drafts[idx] ?? "";
      if (!value.trim()) return;

      // Persist to ticket if there's a ticket_id
      if (i.ticket_id) {
        try {
          await api.answerQuestion(i.ticket_id, 0, value);
        } catch {
          /* swallow */
        }
      }

      // Resume agent if there's a project
      if (selectedProjectId) {
        try {
          await api.resumeAgent(selectedProjectId, value);
          appendLog({
            ts: Date.now() / 1000,
            agent: "human",
            kind: "hitl_response",
            detail: JSON.stringify({ answer: value.slice(0, 200) }),
            checkpoint_id: null,
          });
        } catch {
          /* swallow */
        }
      }

      setLocalInterrups((prev) =>
        prev.map((item, j) =>
          j === idx ? { ...item, answered: true, answer: value, dismissed: true } : item,
        ),
      );
      setDrafts((d) => {
        const next = { ...d };
        delete next[idx];
        return next;
      });
    },
    [localInterrups, drafts, selectedProjectId, appendLog],
  );

  // Auto-focus first pending textarea on mount
  useEffect(() => {
    const firstPendingIdx = localInterrups.findIndex((i) => !i.dismissed && !i.answered);
    if (firstPendingIdx >= 0 && textAreasRef.current[firstPendingIdx]) {
      textAreasRef.current[firstPendingIdx]?.focus();
    }
  }, []);

  // Keyboard shortcut: Ctrl+Enter to submit
  const handleKeyDown = (idx: number, e: React.KeyboardEvent) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      submit(idx);
    }
  };

  // Sort: pending first, then answered, then dismissed — each group sorted by most recent
  const sorted = [...localInterrups].sort((a, b) => {
    const aPriority = a.answered ? 2 : a.dismissed ? 3 : 1;
    const bPriority = b.answered ? 2 : b.dismissed ? 3 : 1;
    if (aPriority !== bPriority) return aPriority - bPriority;
    return b.receivedAt - a.receivedAt;
  });

  const pendingCount = localInterrups.filter((i) => !i.dismissed && !i.answered).length;

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">Human Input Required</h2>
          <p className="text-xs text-zinc-400 mt-1">
            Agents may pause the graph to ask questions. Answer below to resume execution.
          </p>
        </div>
        {pendingCount > 0 && (
          <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-500/40 bg-amber-950/40 px-3 py-1 text-xs font-medium text-amber-300">
            <span className="h-2 w-2 animate-pulse rounded-full bg-amber-400" />
            {pendingCount} pending
          </span>
        )}
      </div>

      {/* Loading indicator */}
      {isLoading && (
        <div className="text-xs text-zinc-500">Loading pending interrupts...</div>
      )}

      {/* Interrupt list */}
      {sorted.length === 0 ? (
        <div className="rounded border border-border bg-surface p-6 text-center text-sm text-zinc-400">
          No pending questions. Agents will surface their questions here when they need your input.
        </div>
      ) : (
        <div className="flex flex-col gap-3">
          {sorted.map((i, idx) => {
            const colors = KIND_COLORS[i.kind ?? ""] ?? DEFAULT_COLORS;
            const value = drafts[localInterrups.indexOf(i)] ?? "";
            const isDone = i.dismissed || i.answered;
            const realIdx = localInterrups.indexOf(i);

            return (
              <div
                key={i.id}
                className={`rounded-lg border ${colors.border} ${colors.bg} p-4 transition-all ${
                  isDone ? "opacity-60" : ""
                }`}
              >
                {/* Type badge + metadata row */}
                <div className="flex items-center gap-2">
                  <span className={`rounded border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${colors.pill}`}>
                    {i.kind ?? "ask_human"}
                  </span>
                  {i.asked_by && (
                    <span className="text-xs text-zinc-400">
                      by <span className="text-zinc-300">{i.asked_by}</span>
                    </span>
                  )}
                  {i.ticket_id && (
                    <span className="text-xs text-zinc-400">
                      · ticket <span className="text-zinc-300">{i.ticket_id.slice(0, 8)}</span>
                    </span>
                  )}
                  <span className="ml-auto text-[10px] text-zinc-500">
                    {formatRelativeTime(i.receivedAt)}
                  </span>
                </div>

                {/* Question */}
                <div className="mt-3 text-sm leading-relaxed text-zinc-100">{i.question}</div>

                {/* Answer textarea */}
                {!isDone && (
                  <div className="mt-3">
                    <textarea
                      ref={(el) => {
                        textAreasRef.current[realIdx] = el;
                      }}
                      className="w-full rounded border border-border bg-muted/50 px-3 py-2 text-sm text-zinc-100 placeholder-zinc-500 focus:border-accent focus:outline-none"
                      rows={3}
                      placeholder="Your answer..."
                      value={value}
                      onChange={(e) => setDrafts((d) => ({ ...d, [realIdx]: e.target.value }))}
                      onKeyDown={(e) => handleKeyDown(realIdx, e)}
                    />
                    <div className="mt-1.5 flex items-center justify-between">
                      <span className="text-[10px] text-zinc-500">
                        Ctrl+Enter to submit
                      </span>
                      <div className="flex gap-2">
                        <button
                          className="rounded border border-accent/50 bg-accent/15 px-4 py-2 text-xs font-medium text-accent transition hover:bg-accent/25"
                          onClick={() => submit(realIdx)}
                          disabled={!value.trim()}
                        >
                          Send &amp; resume
                        </button>
                        <button
                          className="rounded border border-border px-3 py-2 text-xs text-zinc-400 transition hover:bg-muted hover:text-zinc-200"
                          onClick={() => dismissInterrupt(realIdx)}
                        >
                          Dismiss
                        </button>
                      </div>
                    </div>
                  </div>
                )}

                {/* Answered status */}
                {isDone && (
                  <div className="mt-3 flex items-center gap-2 rounded border border-green-500/30 bg-green-950/20 px-3 py-2">
                    <span className="text-sm text-green-400">✓</span>
                    <span className="text-xs text-green-300">
                      Answered: <span className="text-zinc-300">{i.answer?.slice(0, 100)}</span>
                      {i.answer && i.answer.length > 100 ? "..." : ""}
                    </span>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
