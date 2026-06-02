"use client";

import { create } from "zustand";

import { AgentLogEntry, TicketT } from "./api";

export type { AgentLogEntry } from "./api";

export interface PendingInterrupt {
  ticket_id?: string;
  question: string;
  asked_by?: string;
  receivedAt: number;
  kind?: string;        // "ask_human" | "question"
  dismissed?: boolean;
  answered?: boolean;
  answer?: string;
  id?: string;          // unique ID for replay safety
}

export interface CrashState {
  message: string;
  attempt?: number;
  willRetryInSeconds?: number | null;
  ts: number;
}

interface UIState {
  selectedProjectId: string | null;
  setSelectedProject: (id: string | null) => void;

  tickets: TicketT[];
  setTickets: (t: TicketT[]) => void;

  logs: AgentLogEntry[];
  setLogsFromHistory: (entries: AgentLogEntry[]) => void;
  appendLog: (l: AgentLogEntry) => void;
  clearLogs: () => void;

  interrupts: PendingInterrupt[];
  pushInterrupt: (i: PendingInterrupt) => void;
  popInterrupt: (idx: number) => void;

  crash: CrashState | null;
  setCrash: (c: CrashState | null) => void;

  // Log filters
  logSearch: string;
  setLogSearch: (s: string) => void;
  logKindFilter: string;
  setLogKindFilter: (k: string) => void;
  autoScrollEnabled: boolean;
  setAutoScrollEnabled: (v: boolean) => void;

  // Agent activity tracking
  agentLastSeen: Record<string, number>;  // agent name → last seen timestamp
  updateAgentLastSeen: (agent: string) => void;
  activeAgents: string[];
  setActiveAgents: (agents: string[]) => void;

  // Current phase
  currentPhase: string | null;
  setCurrentPhase: (phase: string | null) => void;
}

const MAX_LOGS_IN_MEMORY = 8000;

export const useUIStore = create<UIState>((set) => ({
  selectedProjectId: null,
  setSelectedProject: (id) => set({ selectedProjectId: id }),

  tickets: [],
  setTickets: (t) => set({ tickets: t }),

  logs: [],
  setLogsFromHistory: (entries) =>
    set({
      logs: entries.length > MAX_LOGS_IN_MEMORY ? entries.slice(-MAX_LOGS_IN_MEMORY) : entries,
    }),
  appendLog: (l) =>
    set((s) => ({ logs: [...s.logs.slice(-(MAX_LOGS_IN_MEMORY - 1)), l] })),
  clearLogs: () => set({ logs: [] }),

  interrupts: [],
  pushInterrupt: (i) => set((s) => ({ interrupts: [...s.interrupts, i] })),
  popInterrupt: (idx) =>
    set((s) => ({ interrupts: s.interrupts.filter((_, i) => i !== idx) })),

  crash: null,
  setCrash: (c) => set({ crash: c }),

  // Log filters
  logSearch: "",
  setLogSearch: (s) => set({ logSearch: s }),
  logKindFilter: "all",
  setLogKindFilter: (k) => set({ logKindFilter: k }),
  autoScrollEnabled: true,
  setAutoScrollEnabled: (v) => set({ autoScrollEnabled: v }),

  // Agent activity
  agentLastSeen: {},
  updateAgentLastSeen: (agent) =>
    set((s) => ({
      agentLastSeen: { ...s.agentLastSeen, [agent]: Date.now() / 1000 },
    })),
  activeAgents: [],
  setActiveAgents: (agents) => set({ activeAgents: agents }),

  // Current phase
  currentPhase: null,
  setCurrentPhase: (phase) => set({ currentPhase: phase }),
}));
