import { PendingInterrupt } from "./store";

function asRecord(value: unknown): Record<string, unknown> | null {
  if (typeof value !== "object" || value === null) return null;
  return value as Record<string, unknown>;
}

function payloadFromRaw(raw: unknown): Record<string, unknown> | null {
  const top = asRecord(raw);
  if (!top) return null;
  const nested = asRecord(top.value);
  return nested ?? top;
}

export function normalizeInterrupt(raw: unknown): PendingInterrupt | null {
  const payload = payloadFromRaw(raw);
  if (!payload) {
    if (typeof raw === "string" && raw.trim()) {
      return { question: raw, receivedAt: Date.now() };
    }
    return null;
  }

  const question =
    typeof payload.question === "string"
      ? payload.question
      : typeof payload.message === "string"
        ? payload.message
        : null;
  if (!question?.trim()) return null;

  const questionIndex =
    typeof payload.question_index === "number" ? payload.question_index : undefined;

  return {
    ticket_id: typeof payload.ticket_id === "string" ? payload.ticket_id : undefined,
    question,
    asked_by: typeof payload.asked_by === "string" ? payload.asked_by : undefined,
    kind: typeof payload.kind === "string" ? payload.kind : undefined,
    question_index: questionIndex,
    receivedAt: Date.now(),
  };
}

export function unansweredTicketQuestions(
  tickets: { id: string; title: string; questions: { question: string; answer: string | null; asked_by: string }[] }[],
) {
  const rows: {
    ticket_id: string;
    ticket_title: string;
    question_index: number;
    question: string;
    asked_by: string;
  }[] = [];
  for (const ticket of tickets) {
    ticket.questions.forEach((q, question_index) => {
      if (q.answer) return;
      rows.push({
        ticket_id: ticket.id,
        ticket_title: ticket.title,
        question_index,
        question: q.question,
        asked_by: q.asked_by,
      });
    });
  }
  return rows;
}
