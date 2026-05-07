"use client";

import { useState } from "react";

import { api } from "@/lib/api";
import { useUIStore } from "@/lib/store";

export function InterruptPanel() {
  const { interrupts, popInterrupt, selectedProjectId } = useUIStore();
  const [drafts, setDrafts] = useState<Record<number, string>>({});

  if (interrupts.length === 0) {
    return (
      <div className="rounded border border-border bg-surface p-4 text-sm text-zinc-400">
        No pending questions. Agents will surface their questions here.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      {interrupts.map((i, idx) => {
        const value = drafts[idx] ?? "";
        const submit = async () => {
          if (!value.trim()) return;
          if (i.ticket_id) {
            // Persisted question on a ticket → POST to answer endpoint
            try {
              await api.answerQuestion(i.ticket_id, 0, value);
            } catch {
              /* swallow */
            }
          }
          if (selectedProjectId) {
            try {
              await api.resumeAgent(selectedProjectId, value);
            } catch {
              /* swallow */
            }
          }
          popInterrupt(idx);
        };
        return (
          <div key={`${i.receivedAt}-${idx}`} className="rounded border border-accent/40 bg-accent/10 p-4">
            <div className="text-xs uppercase tracking-wide text-accent">
              {i.asked_by ?? "agent"} asks
              {i.ticket_id ? ` · ticket ${i.ticket_id.slice(0, 8)}` : ""}
            </div>
            <div className="mt-2 text-sm text-zinc-100">{i.question}</div>
            <textarea
              className="mt-3 w-full rounded border border-border bg-muted px-3 py-2 text-sm"
              rows={3}
              placeholder="Your answer…"
              value={value}
              onChange={(e) => setDrafts((d) => ({ ...d, [idx]: e.target.value }))}
            />
            <div className="mt-2 flex gap-2">
              <button
                className="rounded border border-accent bg-accent/20 px-3 py-2 text-xs text-accent"
                onClick={submit}
              >
                Send & resume
              </button>
              <button
                className="rounded border border-border px-3 py-2 text-xs"
                onClick={() => popInterrupt(idx)}
              >
                Dismiss
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
}
