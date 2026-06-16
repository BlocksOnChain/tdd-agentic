/**
 * Turns raw agent-event payloads into human-readable, traceable summaries
 * and provides consistent agent/kind styling across the Logs UI.
 *
 * Events arrive in two shapes:
 *   1. emitted events  → { agent, kind, ...fields }   (tool_result, turn_start…)
 *   2. graph updates    → { node, update, checkpoint_id }  (kind defaults to "log")
 */

export type LogTone = "info" | "success" | "warn" | "error" | "muted" | "accent";

export interface HumanizedLog {
  icon: string;
  /** Friendly name for the event kind, e.g. "Tool result". */
  kindLabel: string;
  /** One-line, plain-English description of what happened. */
  summary: string;
  tone: LogTone;
}

const AGENT_LABELS: Record<string, string> = {
  project_manager: "Project Manager",
  researcher: "Researcher",
  lead: "Lead",
  coordinator: "Coordinator",
  backend_dev: "Backend Dev",
  frontend_dev: "Frontend Dev",
  devops: "DevOps",
  qa: "QA",
  system: "System",
  human: "Human",
};

/** Human-friendly display name for an agent/node id. */
export function agentLabel(agent: string): string {
  return AGENT_LABELS[agent] ?? agent;
}

const AGENT_TEXT: Record<string, string> = {
  project_manager: "text-amber-400",
  researcher: "text-cyan-400",
  lead: "text-violet-400",
  coordinator: "text-pink-400",
  backend_dev: "text-emerald-400",
  frontend_dev: "text-sky-400",
  devops: "text-orange-400",
  qa: "text-lime-400",
  human: "text-yellow-300",
  system: "text-zinc-400",
};

export function agentTextColor(agent: string): string {
  return AGENT_TEXT[agent] ?? "text-zinc-300";
}

/** Tailwind classes for a small agent pill (neutral bg + colored, bold text). */
export function agentPillClasses(agent: string): string {
  return `inline-flex items-center rounded border border-zinc-700 bg-zinc-800/70 px-1.5 py-0.5 text-[10px] font-semibold ${agentTextColor(
    agent,
  )}`;
}

const TONE_TEXT: Record<LogTone, string> = {
  info: "text-sky-400",
  success: "text-emerald-400",
  warn: "text-amber-400",
  error: "text-red-400",
  muted: "text-zinc-500",
  accent: "text-violet-400",
};

export function toneTextColor(tone: LogTone): string {
  return TONE_TEXT[tone];
}

interface KindMeta {
  icon: string;
  label: string;
  tone: LogTone;
}

const KIND_META: Record<string, KindMeta> = {
  turn_start: { icon: "▶", label: "Turn started", tone: "muted" },
  turn_end: { icon: "■", label: "Turn ended", tone: "muted" },
  tool_call: { icon: "🔧", label: "Tool call", tone: "info" },
  tool_result: { icon: "✓", label: "Tool result", tone: "info" },
  handoff: { icon: "➜", label: "Handoff", tone: "accent" },
  route: { icon: "↳", label: "Routed", tone: "accent" },
  route_fallback: { icon: "↳", label: "Route fallback", tone: "warn" },
  verification_gate_failed: { icon: "⛔", label: "Verification failed", tone: "error" },
  crash: { icon: "✖", label: "Crash", tone: "error" },
  error: { icon: "✖", label: "Error", tone: "error" },
  transient_error: { icon: "⚠", label: "Transient error", tone: "warn" },
  hitl_response: { icon: "💬", label: "Human reply", tone: "accent" },
  stopped: { icon: "⏹", label: "Stopped", tone: "warn" },
  log: { icon: "•", label: "Update", tone: "muted" },
};

export function kindMeta(kind: string): KindMeta {
  return KIND_META[kind] ?? { icon: "•", label: kind, tone: "muted" };
}

function asString(v: unknown): string {
  if (v == null) return "";
  if (typeof v === "string") return v;
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}

function squeeze(s: string, max = 240): string {
  const t = s.replace(/\s+/g, " ").trim();
  return t.length > max ? `${t.slice(0, max)}…` : t;
}

/** Pull readable text out of a (model_dump'd) LangChain message content. */
function messageText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((block) => {
        if (typeof block === "string") return block;
        if (block && typeof block === "object") {
          const b = block as Record<string, unknown>;
          if (typeof b.text === "string") return b.text;
        }
        return "";
      })
      .filter(Boolean)
      .join(" ");
  }
  return "";
}

/** Best-effort one-line description of a graph node update payload. */
function describeUpdate(update: unknown): string {
  if (!update || typeof update !== "object") return "";
  const u = update as Record<string, unknown>;
  const messages = u.messages;
  if (Array.isArray(messages) && messages.length > 0) {
    const last = messages[messages.length - 1] as Record<string, unknown>;
    const text = messageText(last?.content);
    if (text) return text;
  }
  const parts: string[] = [];
  if (typeof u.next_agent === "string") parts.push(`next → ${agentLabel(u.next_agent)}`);
  if (typeof u.active_ticket_id === "string") parts.push(`ticket ${u.active_ticket_id}`);
  if (typeof u.active_subtask_id === "string") parts.push(`subtask ${u.active_subtask_id}`);
  return parts.join(", ");
}

/**
 * Produce a human-readable rendering of a single log entry from its kind +
 * full payload.
 */
export function humanizeLog(entry: {
  kind: string;
  agent: string;
  payload?: Record<string, unknown>;
  detail?: string;
}): HumanizedLog {
  const kind = entry.kind || "log";
  const p = entry.payload ?? {};
  const meta = kindMeta(kind);

  let summary = "";
  switch (kind) {
    case "turn_start":
      summary = "started working";
      break;
    case "turn_end":
      summary = "finished its turn";
      break;
    case "tool_call": {
      const name = asString(p.name || p.tool);
      const args = p.args ?? p.arguments;
      const argStr = args ? squeeze(asString(args), 140) : "";
      summary = `called ${name || "a tool"}${argStr ? `(${argStr})` : ""}`;
      break;
    }
    case "tool_result": {
      const name = asString(p.name);
      const pv = squeeze(asString(p.preview ?? p.result ?? p.content), 240);
      summary = `${name ? `${name} → ` : ""}${pv || "completed"}`;
      break;
    }
    case "handoff": {
      summary = squeeze(asString(p.summary ?? p.content ?? p.detail), 240) || "handed off to the Project Manager";
      break;
    }
    case "route":
    case "route_fallback": {
      const target = asString(p.target ?? p.next_agent ?? p.to);
      summary = target ? `routed to ${agentLabel(target)}` : "made a routing decision";
      break;
    }
    case "verification_gate_failed": {
      const sub = asString(p.subtask_id);
      const detail = squeeze(asString(p.detail), 200);
      summary = `subtask ${sub || "?"} kept blocked${detail ? `: ${detail}` : ""}`;
      break;
    }
    case "crash":
    case "error":
    case "transient_error": {
      const attempt = p.attempt ? ` (attempt ${asString(p.attempt)})` : "";
      summary = `${squeeze(asString(p.error ?? p.detail), 240) || "an error occurred"}${attempt}`;
      break;
    }
    case "stopped":
      summary = "run stopped";
      break;
    case "hitl_response":
      summary = squeeze(asString(p.response ?? p.answer), 240) || "human answered";
      break;
    default: {
      const desc = describeUpdate(p.update);
      summary = desc || squeeze(asString(p.detail ?? p), 240);
    }
  }

  return { icon: meta.icon, kindLabel: meta.label, summary, tone: meta.tone };
}
