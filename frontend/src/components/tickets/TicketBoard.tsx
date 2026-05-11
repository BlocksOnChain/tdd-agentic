"use client";

import { useEffect, useState } from "react";

import { api, TestCaseSpec, TestType, TicketStatus, TicketT } from "@/lib/api";
import { cn } from "@/lib/cn";
import { useUIStore } from "@/lib/store";

const TEST_TYPE_BADGE: Record<TestType, string> = {
  unit: "bg-emerald-900/40 text-emerald-300 ring-emerald-700",
  integration: "bg-amber-900/40 text-amber-300 ring-amber-700",
  functional: "bg-violet-900/40 text-violet-300 ring-violet-700",
};

function TestCaseItem({ spec }: { spec: TestCaseSpec }) {
  const badge = TEST_TYPE_BADGE[spec.test_type] ?? TEST_TYPE_BADGE.unit;
  return (
    <li>
      <div className="flex flex-wrap items-baseline gap-x-2">
        <span className={cn("rounded px-1.5 py-0.5 text-[10px] font-medium uppercase ring-1", badge)}>
          {spec.test_type}
        </span>
        <span className="text-zinc-300">
          <span className="text-zinc-500">given</span> {spec.given}
        </span>
      </div>
      <div className="ml-1 mt-0.5">
        <span className="text-zinc-500">should</span> {spec.should}
      </div>
      <div className="ml-1 mt-0.5 font-mono text-[11px] text-zinc-400">
        <span className="text-zinc-500">expected</span> {spec.expected}
      </div>
      {spec.notes && (
        <div className="ml-1 mt-0.5 text-[11px] italic text-zinc-500">{spec.notes}</div>
      )}
    </li>
  );
}

const COLUMNS: { id: TicketStatus; label: string }[] = [
  { id: "draft", label: "Draft" },
  { id: "in_review", label: "In review" },
  { id: "questions_pending", label: "Questions" },
  { id: "todo", label: "To do" },
  { id: "in_progress", label: "In progress" },
  { id: "done", label: "Done" },
];

export function TicketBoard() {
  const { tickets, selectedProjectId, setTickets } = useUIStore();
  const [selected, setSelected] = useState<TicketT | null>(null);

  useEffect(() => {
    if (!selectedProjectId) return;
    api.listTickets(selectedProjectId).then(setTickets).catch(() => undefined);
  }, [selectedProjectId, setTickets]);

  const grouped: Record<string, TicketT[]> = {};
  COLUMNS.forEach((c) => (grouped[c.id] = []));
  tickets.forEach((t) => {
    if (grouped[t.status]) grouped[t.status].push(t);
  });

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-3 2xl:grid-cols-6">
        {COLUMNS.map((c) => (
          <div key={c.id} className="rounded border border-border bg-surface p-3 min-h-[200px]">
            <div className="mb-2 flex items-center justify-between">
              <div className="text-xs font-medium uppercase tracking-wide text-zinc-400">
                {c.label}
              </div>
              <div className="text-xs text-zinc-500">{grouped[c.id].length}</div>
            </div>
            <div className="flex flex-col gap-2">
              {grouped[c.id].map((t) => (
                <button
                  key={t.id}
                  onClick={() => setSelected(t)}
                  className={cn(
                    "rounded border border-border bg-muted p-2 text-left text-sm hover:border-accent",
                    selected?.id === t.id && "border-accent"
                  )}
                >
                  <div className="font-medium line-clamp-2">{t.title}</div>
                  <div className="mt-1 text-xs text-zinc-400">
                    {t.subtasks.length} subtasks
                  </div>
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>

      {selected && <TicketDetail ticket={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}

function TicketDetail({ ticket, onClose }: { ticket: TicketT; onClose: () => void }) {
  const { selectedProjectId, setTickets } = useUIStore();
  const [drafts, setDrafts] = useState<Record<number, string>>({});
  const [errors, setErrors] = useState<Record<number, string>>({});

  const submitAnswer = async (questionIndex: number) => {
    const value = drafts[questionIndex]?.trim();
    if (!value) return;
    setErrors((prev) => {
      const next = { ...prev };
      delete next[questionIndex];
      return next;
    });
    try {
      await api.answerQuestion(ticket.id, questionIndex, value);
      if (selectedProjectId) {
        const refreshed = await api.listTickets(selectedProjectId);
        setTickets(refreshed);
        await api.resumeAgent(selectedProjectId, value);
      }
      setDrafts((prev) => {
        const next = { ...prev };
        delete next[questionIndex];
        return next;
      });
    } catch (err) {
      setErrors((prev) => ({
        ...prev,
        [questionIndex]: err instanceof Error ? err.message : "Could not save answer",
      }));
    }
  };

  return (
    <div className="rounded border border-border bg-surface p-4">
      <div className="flex items-start justify-between">
        <div>
          <div className="text-xs uppercase tracking-wide text-zinc-500">{ticket.status}</div>
          <h2 className="text-lg font-semibold">{ticket.title}</h2>
        </div>
        <button onClick={onClose} className="text-xs text-zinc-400 hover:text-zinc-200">
          Close
        </button>
      </div>
      <p className="mt-2 text-sm text-zinc-300">{ticket.description}</p>

      <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2">
        <RequirementsList title="Business" items={ticket.business_requirements} />
        <RequirementsList title="Technical" items={ticket.technical_requirements} />
      </div>

      {ticket.questions.length > 0 && (
        <div className="mt-4">
          <div className="mb-2 text-xs font-medium uppercase tracking-wide text-amber-400">
            Open questions
          </div>
          <ul className="flex flex-col gap-2">
            {ticket.questions.map((q, i) => (
              <li key={i} className="rounded border border-border bg-muted p-2 text-sm">
                <div className="text-zinc-200">{q.question}</div>
                {q.answer ? (
                  <div className="mt-1 text-xs text-zinc-400">→ {q.answer}</div>
                ) : (
                  <div className="mt-2 flex flex-col gap-2">
                    <textarea
                      className="w-full rounded border border-border bg-surface px-2 py-1 text-xs"
                      rows={2}
                      placeholder="Your answer…"
                      value={drafts[i] ?? ""}
                      onChange={(e) => setDrafts((prev) => ({ ...prev, [i]: e.target.value }))}
                    />
                    {errors[i] && <div className="text-xs text-red-400">{errors[i]}</div>}
                    <button
                      type="button"
                      className="self-start rounded border border-accent bg-accent/20 px-2 py-1 text-xs text-accent"
                      onClick={() => submitAnswer(i)}
                    >
                      Save answer & resume
                    </button>
                  </div>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="mt-4">
        <div className="mb-2 text-xs font-medium uppercase tracking-wide text-zinc-400">
          Subtasks ({ticket.subtasks.length})
        </div>
        <div className="flex flex-col gap-2">
          {ticket.subtasks.map((s) => (
            <div key={s.id} className="rounded border border-border bg-muted p-3">
              <div className="flex items-center justify-between">
                <div className="text-sm font-medium">
                  #{s.order_index} · {s.title}
                </div>
                <div className="text-xs text-zinc-400">
                  {s.assigned_to} · {s.status}
                </div>
              </div>
              {s.test_cases.length > 0 && (
                <div className="mt-2">
                  <div className="text-xs uppercase tracking-wide text-zinc-500">
                    Test cases (TDD · RITE)
                  </div>
                  <ul className="ml-4 list-disc space-y-1.5 text-xs text-zinc-300">
                    {s.test_cases.map((t, i) => (
                      <TestCaseItem key={i} spec={t} />
                    ))}
                  </ul>
                </div>
              )}
              {s.todos.length > 0 && (
                <div className="mt-2">
                  <div className="text-xs uppercase tracking-wide text-zinc-500">Todos</div>
                  <ul className="ml-4 list-disc text-xs text-zinc-300">
                    {s.todos.map((t) => (
                      <li key={t.id}>
                        <span className={t.status === "done" ? "line-through text-zinc-500" : ""}>
                          {t.title}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function RequirementsList({ title, items }: { title: string; items: string[] }) {
  return (
    <div>
      <div className="mb-1 text-xs font-medium uppercase tracking-wide text-zinc-400">
        {title} requirements
      </div>
      {items.length === 0 ? (
        <div className="text-xs text-zinc-500">none</div>
      ) : (
        <ul className="ml-4 list-disc text-sm text-zinc-200">
          {items.map((r, i) => (
            <li key={i}>{r}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
