"use client";

import { useState } from "react";

import { api } from "@/lib/api";
import { unansweredTicketQuestions } from "@/lib/interrupts";
import { useUIStore } from "@/lib/store";

export function TicketQuestionsPanel() {
  const { tickets, selectedProjectId, setTickets } = useUIStore();
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const open = unansweredTicketQuestions(tickets);

  if (!selectedProjectId) {
    return (
      <div className="rounded border border-border bg-surface p-4 text-sm text-zinc-400">
        Select a project to see ticket questions.
      </div>
    );
  }

  if (open.length === 0) {
    return (
      <div className="rounded border border-border bg-surface p-4 text-sm text-zinc-400">
        No unanswered ticket questions. Agents add them when they need clarification on a ticket.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      {open.map((row) => {
        const key = `${row.ticket_id}:${row.question_index}`;
        const value = drafts[key] ?? "";
        const submit = async () => {
          if (!value.trim()) return;
          setErrors((prev) => {
            const next = { ...prev };
            delete next[key];
            return next;
          });
          try {
            await api.answerQuestion(row.ticket_id, row.question_index, value.trim());
            if (selectedProjectId) {
              const refreshed = await api.listTickets(selectedProjectId);
              setTickets(refreshed);
              await api.resumeAgent(selectedProjectId, value.trim());
            }
            setDrafts((prev) => {
              const next = { ...prev };
              delete next[key];
              return next;
            });
          } catch (err) {
            setErrors((prev) => ({
              ...prev,
              [key]: err instanceof Error ? err.message : "Could not save answer",
            }));
          }
        };
        return (
          <div key={key} className="rounded border border-border bg-surface p-4">
            <div className="text-xs uppercase tracking-wide text-zinc-500">
              {row.asked_by} · ticket {row.ticket_id.slice(0, 8)}
            </div>
            <div className="mt-1 text-sm font-medium text-zinc-100">{row.ticket_title}</div>
            <div className="mt-2 text-sm text-zinc-200">{row.question}</div>
            <textarea
              className="mt-3 w-full rounded border border-border bg-muted px-3 py-2 text-sm"
              rows={3}
              placeholder="Your answer…"
              value={value}
              onChange={(e) => setDrafts((prev) => ({ ...prev, [key]: e.target.value }))}
            />
            {errors[key] && <div className="mt-2 text-xs text-red-400">{errors[key]}</div>}
            <div className="mt-2">
              <button
                type="button"
                className="rounded border border-accent bg-accent/20 px-3 py-2 text-xs text-accent"
                onClick={submit}
              >
                Save answer & resume
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
}
