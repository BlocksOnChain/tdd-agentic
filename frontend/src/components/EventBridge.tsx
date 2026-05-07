"use client";

import { useEffect } from "react";

import { api, TicketT, persistedLogToEntry } from "@/lib/api";
import { useUIStore } from "@/lib/store";
import { useEventStream, WSEvent } from "@/lib/websocket";

/**
 * Listens to the global WebSocket event stream and fans out into the
 * Zustand store: log feed, tickets, interrupts. Mounted once at the root
 * of the layout tree so every page sees live state.
 */
export function EventBridge() {
  const { appendLog, pushInterrupt, setTickets, selectedProjectId, setCrash, setLogsFromHistory } =
    useUIStore();

  useEventStream((e: WSEvent) => {
    if (e.type === "agent") {
      const p = e.payload as Record<string, unknown>;
      const node = (p.node as string) ?? (p.agent as string) ?? "system";
      const kind = (p.kind as string) ?? "log";
      appendLog({
        ts: e.ts ?? Date.now() / 1000,
        agent: node,
        kind,
        detail: JSON.stringify(p).slice(0, 1200),
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
      const p = e.payload as Record<string, unknown>;
      pushInterrupt({
        ticket_id: p.ticket_id as string | undefined,
        question: (p.question as string) ?? "(no question text)",
        asked_by: p.asked_by as string | undefined,
        receivedAt: Date.now(),
      });
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
  }, [selectedProjectId, setTickets, setLogsFromHistory]);

  return null;
}
