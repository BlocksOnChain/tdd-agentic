"use client";

import { useState } from "react";

import { api } from "@/lib/api";
import { unansweredTicketQuestions } from "@/lib/interrupts";
import { useUIStore } from "@/lib/store";

export function InterruptPanel() {
  const { interrupts, popInterrupt, selectedProjectId, tickets, setTickets } = useUIStore();
  const [drafts, setDrafts] = useState<Record<number, string>>({});
  const [errors, setErrors] = useState<Record<number, string>>({});

  if (!selectedProjectId) {
    return (
      <div className="rounded border border-border bg-surface p-4 text-sm text-zinc-400">
        Select a project to answer agent questions and resume the graph.
      </div>
    );
  }

  if (interrupts.length === 0) {
    return (
      <div className="rounded border border-border bg-surface p-4 text-sm text-zinc-400">
        No pending graph interrupts. When an agent calls <code className="text-zinc-300">ask_human</code>,
        the paused question appears here and in the banner above.
      </div>
    );
  }

  const questionIndexForTicket = (ticketId: string, question: string) => {
    const ticket = tickets.find((t) => t.id === ticketId);
    if (!ticket) return 0;
    const idx = ticket.questions.findIndex((q) => q.question === question && !q.answer);
    return idx >= 0 ? idx : 0;
  };

  return (
    <div className="flex flex-col gap-3">
      {interrupts.map((i, idx) => {
        const value = drafts[idx] ?? "";
        const submit = async () => {
          if (!value.trim() || !selectedProjectId) return;
          setErrors((prev) => {
            const next = { ...prev };
            delete next[idx];
            return next;
          });
          try {
            if (i.ticket_id) {
              const questionIndex =
                typeof i.question_index === "number"
                  ? i.question_index
                  : questionIndexForTicket(i.ticket_id, i.question);
              await api.answerQuestion(i.ticket_id, questionIndex, value.trim());
              const refreshed = await api.listTickets(selectedProjectId);
              setTickets(refreshed);
            }
            await api.resumeAgent(selectedProjectId, value.trim());
            popInterrupt(idx);
            setDrafts((prev) => {
              const next = { ...prev };
              delete next[idx];
              return next;
            });
          } catch (err) {
            setErrors((prev) => ({
              ...prev,
              [idx]: err instanceof Error ? err.message : "Could not resume agent",
            }));
          }
        };
        return (
          <div key={`${i.receivedAt}-${idx}`} className="rounded border border-accent/40 bg-accent/10 p-4">
            <div className="text-xs uppercase tracking-wide text-accent">
              {i.asked_by ?? "agent"} asks
              {i.ticket_id ? ` · ticket ${i.ticket_id.slice(0, 8)}` : ""}
              {i.kind ? ` · ${i.kind}` : ""}
            </div>
            <div className="mt-2 text-sm text-zinc-100">{i.question}</div>
            <textarea
              className="mt-3 w-full rounded border border-border bg-muted px-3 py-2 text-sm"
              rows={3}
              placeholder="Your answer…"
              value={value}
              onChange={(e) => setDrafts((d) => ({ ...d, [idx]: e.target.value }))}
            />
            {errors[idx] && <div className="mt-2 text-xs text-red-400">{errors[idx]}</div>}
            <div className="mt-2 flex gap-2">
              <button
                type="button"
                className="rounded border border-accent bg-accent/20 px-3 py-2 text-xs text-accent"
                onClick={submit}
              >
                Send & resume
              </button>
              <button
                type="button"
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

export function useHumanInputCount() {
  const interrupts = useUIStore((s) => s.interrupts.length);
  const ticketQuestions = useUIStore((s) => unansweredTicketQuestions(s.tickets).length);
  return interrupts + ticketQuestions;
}
