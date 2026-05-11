"use client";

import { create } from "zustand";

import { AgentLogEntry, TicketT } from "./api";

export type { AgentLogEntry } from "./api";

export interface PendingInterrupt {
  ticket_id?: string;
  question: string;
  asked_by?: string;
  kind?: string;
  question_index?: number;
  receivedAt: number;
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
  setInterrupts: (items: PendingInterrupt[]) => void;
  pushInterrupt: (i: PendingInterrupt) => void;
  popInterrupt: (idx: number) => void;

  crash: CrashState | null;
  setCrash: (c: CrashState | null) => void;
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
  setInterrupts: (items) => set({ interrupts: items }),
  pushInterrupt: (i) =>
    set((s) => {
      const duplicate = s.interrupts.some(
        (existing) =>
          existing.question === i.question &&
          existing.ticket_id === i.ticket_id &&
          existing.kind === i.kind,
      );
      if (duplicate) return s;
      return { interrupts: [...s.interrupts, i] };
    }),
  popInterrupt: (idx) =>
    set((s) => ({ interrupts: s.interrupts.filter((_, i) => i !== idx) })),

  crash: null,
  setCrash: (c) => set({ crash: c }),
}));
