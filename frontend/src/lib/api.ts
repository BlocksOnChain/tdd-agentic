import { formatAgentLogDetail } from "./agentLog";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type TicketStatus =
  | "draft"
  | "in_review"
  | "questions_pending"
  | "todo"
  | "in_progress"
  | "done"
  | "blocked";

export type SubtaskStatus = "pending" | "in_progress" | "done" | "blocked";

export type AgentRole =
  | "project_manager"
  | "researcher"
  | "backend_lead"
  | "frontend_lead"
  | "backend_dev"
  | "frontend_dev"
  | "devops"
  | "qa";

export interface TodoT {
  id: string;
  subtask_id: string;
  title: string;
  detail: string;
  order_index: number;
  status: "pending" | "done";
}

export type TestType = "unit" | "integration" | "functional";

export interface TestCaseSpec {
  given: string;
  should: string;
  expected: string;
  test_type: TestType;
  notes: string;
}

export interface SubtaskT {
  id: string;
  ticket_id: string;
  title: string;
  description: string;
  required_functionality: string;
  test_cases: TestCaseSpec[];
  assigned_to: AgentRole;
  order_index: number;
  status: SubtaskStatus;
  todos: TodoT[];
}

export interface TicketQuestion {
  question: string;
  answer: string | null;
  asked_by: string;
  answered_by: string | null;
  ts: number | null;
}

export interface TicketT {
  id: string;
  project_id: string;
  title: string;
  description: string;
  business_requirements: string[];
  technical_requirements: string[];
  status: TicketStatus;
  questions: TicketQuestion[];
  order_index: number;
  subtasks: SubtaskT[];
}

export interface ProjectT {
  id: string;
  name: string;
  description: string;
  goal: string;
}

/** One line in the Logs panel (live WebSocket or replayed from Postgres). */
export interface AgentLogEntry {
  id?: string;
  ts: number;
  agent: string;
  kind: string;
  detail: string;
  checkpoint_id?: string | null;
}

/** Row from ``GET /api/agents/logs/:projectId``. */
export interface PersistedAgentLog {
  id: string;
  created_at: string | null;
  ts: number | null;
  agent: string;
  kind: string;
  payload: Record<string, unknown>;
  detail?: string;
  checkpoint_id: string | null;
}

export interface LogItemResponse {
  log: {
    id: string;
    project_id: string | null;
    created_at: string | null;
    ts: number | null;
    agent: string;
    kind: string;
    payload: Record<string, unknown>;
    checkpoint_id: string | null;
  };
  checkpoint: {
    checkpoint_id: string;
    created_at: string | null;
    bytes: number | null;
    metadata: Record<string, unknown>;
  } | null;
}

export function persistedLogToEntry(row: PersistedAgentLog): AgentLogEntry {
  const p = row.payload || {};
  const node = typeof p.node === "string" ? p.node : null;
  const agentField = typeof p.agent === "string" ? p.agent : null;
  const ts =
    typeof row.ts === "number" && !Number.isNaN(row.ts)
      ? row.ts
      : row.created_at
        ? Date.parse(row.created_at) / 1000
        : Date.now() / 1000;
  const cp = row.checkpoint_id ?? (typeof p.checkpoint_id === "string" ? p.checkpoint_id : null);
  const kind = row.kind || (typeof p.kind === "string" ? p.kind : "log");
  return {
    id: row.id,
    ts,
    agent: node ?? agentField ?? row.agent,
    kind,
    detail: row.detail ?? formatAgentLogDetail(kind, p),
    checkpoint_id: cp,
  };
}

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return (await res.json()) as T;
}

export interface CheckpointT {
  checkpoint_id: string | null;
  parent_checkpoint_id: string | null;
  created_at: string | null;
  source: string | null;
  step: number | null;
  wrote_nodes: string[];
  next: string[];
}

export const api = {
  listProjects: () => http<ProjectT[]>("/api/projects"),
  createProject: (data: Partial<ProjectT>) =>
    http<ProjectT>("/api/projects", { method: "POST", body: JSON.stringify(data) }),
  deleteProject: (projectId: string) =>
    http(`/api/projects/${projectId}`, { method: "DELETE" }),
  listTickets: (projectId?: string) =>
    http<TicketT[]>(`/api/tickets${projectId ? `?project_id=${projectId}` : ""}`),
  listSubtasks: (params: { project_id: string; ticket_id?: string; assigned_to?: AgentRole; status?: SubtaskStatus }) => {
    const q = new URLSearchParams();
    q.set("project_id", params.project_id);
    if (params.ticket_id) q.set("ticket_id", params.ticket_id);
    if (params.assigned_to) q.set("assigned_to", params.assigned_to);
    if (params.status) q.set("status", params.status);
    return http<SubtaskT[]>(`/api/tickets/subtasks?${q.toString()}`);
  },
  getTicket: (id: string) => http<TicketT>(`/api/tickets/${id}`),
  answerQuestion: (ticketId: string, questionIndex: number, answer: string) =>
    http<TicketT>(`/api/tickets/${ticketId}/answer`, {
      method: "POST",
      body: JSON.stringify({ question_index: questionIndex, answer, answered_by: "human" }),
    }),
  startAgent: (projectId: string, goal: string) =>
    http(`/api/agents/start`, {
      method: "POST",
      body: JSON.stringify({ project_id: projectId, goal }),
    }),
  resumeAgent: (projectId: string, response: string) =>
    http(`/api/agents/resume`, {
      method: "POST",
      body: JSON.stringify({ project_id: projectId, response }),
    }),
  listInterrupts: (projectId: string) =>
    http<{ interrupts: unknown[] }>(`/api/agents/interrupts/${projectId}`),
  retryAgent: (projectId: string) =>
    http(`/api/agents/retry`, {
      method: "POST",
      body: JSON.stringify({ project_id: projectId }),
    }),
  stopAgent: (projectId: string) =>
    http(`/api/agents/stop`, {
      method: "POST",
      body: JSON.stringify({ project_id: projectId }),
    }),
  resumeFrom: (projectId: string, checkpointId: string) =>
    http(`/api/agents/resume_from`, {
      method: "POST",
      body: JSON.stringify({ project_id: projectId, checkpoint_id: checkpointId }),
    }),
  listCheckpoints: (projectId: string, opts?: { force?: boolean }) => {
    const q = opts?.force ? "?force=true" : "";
    return http<{ checkpoints: CheckpointT[] }>(
      `/api/agents/checkpoints/${projectId}${q}`,
    );
  },
  listAgentLogs: (projectId: string, limit = 5000) =>
    http<{ logs: PersistedAgentLog[] }>(`/api/agents/logs/${projectId}?limit=${limit}`),
  getAgentLogItem: (logId: string) =>
    http<LogItemResponse>(`/api/agents/logs/item/${logId}`),
};
