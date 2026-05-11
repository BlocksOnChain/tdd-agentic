"use client";

import { useEffect } from "react";

import { formatAgentLogDetail } from "@/lib/agentLog";
import { api, TicketT, persistedLogToEntry } from "@/lib/api";
import { normalizeInterrupt } from "@/lib/interrupts";
import { useUIStore } from "@/lib/store";
import { useEventStream, WSEvent } from "@/lib/websocket";

/**
 * Listens to the global WebSocket event stream and fans out into the
 * Zustand store: log feed, tickets, interrupts. Mounted once at the root
 * of the layout tree so every page sees live state.
 */
export function EventBridge() {
  const {
    appendLog,
    pushInterrupt,
    setInterrupts,
    setTickets,
    selectedProjectId,
    setCrash,
    setLogsFromHistory,
  } = useUIStore();

  useEventStream((e: WSEvent) => {
    if (e.type === "agent") {
      const activeProjectId = useUIStore.getState().selectedProjectId;
      if (activeProjectId && e.project_id && e.project_id !== activeProjectId) return;
      const p = e.payload as Record<string, unknown>;
      const node = (p.node as string) ?? (p.agent as string) ?? "system";
      const kind = (p.kind as string) ?? "log";
      appendLog({
        ts: e.ts ?? Date.now() / 1000,
        agent: node,
        kind,
        detail: formatAgentLogDetail(kind, p),
        checkpoint_id: (p.checkpoint_id as string | undefined) ?? null,
      });
      if (kind === "crash") {
        setCrash({
          message: (p.error as string) ?? "Agent run crashed",
          attempt: p.attempt as number | undefined,
          willRetryInSeconds: null,
          ts: Date.now(),
        });
      } else if (kind === "transient_error") {
        setCrash({
          message: (p.error as string) ?? "Transient error",
          attempt: p.attempt as number | undefined,
          willRetryInSeconds: p.will_retry_in_seconds as number | null | undefined,
          ts: Date.now(),
        });
      }
    }
    if (e.type === "interrupt") {
      if (selectedProjectId && e.project_id && e.project_id !== selectedProjectId) return;
      const normalized = normalizeInterrupt(e.payload);
      if (normalized) pushInterrupt(normalized);
    }
    if (e.type === "ticket" && selectedProjectId) {
      api
        .listTickets(selectedProjectId)
        .then((tickets: TicketT[]) => setTickets(tickets))
        .catch(() => undefined);
    }
    if (e.type === "project") {
      const p = e.payload as Record<string, unknown>;
      if (p.action === "deleted" && p.id === selectedProjectId) {
        setTickets([]);
        setLogsFromHistory([]);
      }
    }
  });

  // Tickets + persisted agent logs when project changes (survives full page reload)
  useEffect(() => {
    if (!selectedProjectId) {
      setLogsFromHistory([]);
      setInterrupts([]);
      return;
    }
    const pid = selectedProjectId;
    useUIStore.getState().clearLogs();
    api
      .listTickets(pid)
      .then((tickets: TicketT[]) => {
        if (useUIStore.getState().selectedProjectId !== pid) return;
        setTickets(tickets);
      })
      .catch(() => undefined);
    api
      .listAgentLogs(pid)
      .then(({ logs }) => {
        if (useUIStore.getState().selectedProjectId !== pid) return;
        setLogsFromHistory(logs.map(persistedLogToEntry));
      })
      .catch(() => undefined);
    api
      .listInterrupts(pid)
      .then(({ interrupts }) => {
        if (useUIStore.getState().selectedProjectId !== pid) return;
        setInterrupts(
          interrupts
            .map((item) => normalizeInterrupt(item))
            .filter((item): item is NonNullable<typeof item> => item !== null),
        );
      })
      .catch(() => undefined);
  }, [selectedProjectId, setTickets, setLogsFromHistory, setInterrupts]);

  return null;
}
