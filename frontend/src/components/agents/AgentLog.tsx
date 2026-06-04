"use client";

import { type ReactNode, useEffect, useRef, useState } from "react";

import { api, AgentLogEntry, LogItemResponse } from "@/lib/api";
import { useUIStore } from "@/lib/store";
import { LogFilterBar } from "@/components/logs/LogFilterBar";
import {
  agentLabel,
  agentPillClasses,
  humanizeLog,
  kindMeta,
  toneTextColor,
} from "@/lib/logFormat";

function formatRelativeTime(ts: number): string {
  const now = Date.now() / 1000;
  const diff = now - ts;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return new Date(ts * 1000).toLocaleString();
}

function formatAbsoluteTime(ts: number): string {
  return new Date(ts * 1000).toLocaleString();
}

/** A single human-readable log row. */
function LogRow({
  entry,
  onOpen,
  onResume,
  resuming,
  canResume,
}: {
  entry: AgentLogEntry;
  onOpen: () => void;
  onResume: () => void;
  resuming: boolean;
  canResume: boolean;
}) {
  const human = humanizeLog(entry);
  return (
    <div
      className="group flex cursor-pointer items-start gap-2 rounded px-2 py-1.5 hover:bg-muted/40"
      onClick={onOpen}
      title="Click to view full details"
    >
      <span
        className="w-16 shrink-0 pt-0.5 text-right tabular-nums text-zinc-600"
        title={formatAbsoluteTime(entry.ts)}
      >
        {formatRelativeTime(entry.ts)}
      </span>

      <span className={`shrink-0 ${agentPillClasses(entry.agent)}`}>
        {agentLabel(entry.agent)}
      </span>

      <span
        className={`shrink-0 inline-flex items-center gap-1 pt-0.5 text-[11px] font-medium ${toneTextColor(
          human.tone,
        )}`}
        title={entry.kind}
      >
        <span aria-hidden>{human.icon}</span>
        {human.kindLabel}
      </span>

      <span className="min-w-0 flex-1 pt-0.5 text-zinc-300 break-words">
        {human.summary || "—"}
      </span>

      {canResume && (
        <button
          title={`Resume from checkpoint ${entry.checkpoint_id}`}
          onClick={(e: { stopPropagation(): void }) => {
            e.stopPropagation();
            onResume();
          }}
          disabled={resuming}
          className="shrink-0 self-start rounded border border-border px-1.5 py-0.5 text-[10px] text-zinc-400 opacity-0 hover:bg-muted hover:text-zinc-200 group-hover:opacity-100 disabled:opacity-50"
        >
          {resuming ? "…" : "↶ resume"}
        </button>
      )}
    </div>
  );
}

/** Full-detail modal. Renders complete, untruncated payload. */
function DetailModal({
  entry,
  serverDetail,
  loading,
  error,
  onClose,
}: {
  entry: AgentLogEntry;
  serverDetail: LogItemResponse | null;
  loading: boolean;
  error: string | null;
  onClose: () => void;
}) {
  // Prefer the authoritative server payload when available, else the live one.
  const payload = serverDetail?.log.payload ?? entry.payload ?? {};
  const human = humanizeLog({ ...entry, payload });
  const meta = kindMeta(entry.kind);
  const checkpointId = serverDetail?.log.checkpoint_id ?? entry.checkpoint_id ?? null;
  const createdAt = serverDetail?.log.created_at ?? formatAbsoluteTime(entry.ts);

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/60 p-6"
      onClick={onClose}
    >
      <div
        className="flex max-h-[85vh] w-full max-w-3xl flex-col rounded-lg border border-border bg-surface shadow-xl"
        onClick={(e: { stopPropagation(): void }) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-3">
          <div className="flex items-center gap-2">
            <span className={agentPillClasses(entry.agent)}>
              {agentLabel(entry.agent)}
            </span>
            <span className={`inline-flex items-center gap-1 text-sm font-medium ${toneTextColor(meta.tone)}`}>
              <span aria-hidden>{meta.icon}</span>
              {meta.label}
            </span>
          </div>
          <button
            className="text-xs text-zinc-400 hover:text-zinc-200"
            onClick={onClose}
          >
            Close
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 space-y-3 overflow-auto p-4 text-xs">
          {/* Human-readable summary */}
          <div className="rounded border border-border bg-muted/40 p-3">
            <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-zinc-500">
              Summary
            </div>
            <p className="whitespace-pre-wrap break-words text-sm text-zinc-200">
              {human.summary || "(no summary)"}
            </p>
          </div>

          {/* Metadata grid */}
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Meta label="Agent" value={agentLabel(entry.agent)} />
            <Meta label="Kind" value={entry.kind} />
            <Meta label="Time" value={createdAt ?? "—"} />
            <Meta label="Checkpoint" value={checkpointId ?? "—"} mono />
          </div>

          {/* Notable fields surfaced for quick scanning */}
          <FieldHighlights payload={payload} />

          {/* Checkpoint enrichment (server only) */}
          {serverDetail?.checkpoint && (
            <Section title="Checkpoint">
              <div className="space-y-0.5 text-zinc-400">
                <div>id: <span className="font-mono text-zinc-300">{serverDetail.checkpoint.checkpoint_id}</span></div>
                <div>created_at: {serverDetail.checkpoint.created_at ?? "—"}</div>
                <div>size: {serverDetail.checkpoint.bytes ?? "—"} bytes</div>
              </div>
              <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap break-words rounded bg-black/30 p-2 text-[11px] text-zinc-300">
                {JSON.stringify(serverDetail.checkpoint.metadata, null, 2)}
              </pre>
            </Section>
          )}

          {/* Full raw payload — never truncated */}
          <Section title="Full payload">
            {loading && <div className="text-zinc-500">Loading authoritative payload…</div>}
            {error && <div className="text-amber-400">Showing live payload ({error}).</div>}
            <pre className="max-h-[40vh] overflow-auto whitespace-pre-wrap break-words rounded bg-black/30 p-2 font-mono text-[11px] leading-relaxed text-zinc-300">
              {JSON.stringify(payload, null, 2)}
            </pre>
          </Section>
        </div>
      </div>
    </div>
  );
}

function Meta({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="rounded border border-border bg-muted/30 px-2 py-1.5">
      <div className="text-[10px] uppercase tracking-wide text-zinc-500">{label}</div>
      <div className={`truncate text-zinc-200 ${mono ? "font-mono text-[10px]" : ""}`} title={value}>
        {value}
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="rounded border border-border bg-muted/40 p-3">
      <div className="mb-2 text-[10px] font-semibold uppercase tracking-wide text-zinc-500">
        {title}
      </div>
      {children}
    </div>
  );
}

/** Pull a few commonly-useful fields out of the payload for fast scanning. */
function FieldHighlights({ payload }: { payload: Record<string, unknown> }) {
  const rows: Array<[string, string]> = [];
  const add = (key: string, label: string) => {
    const v = payload[key];
    if (v == null || v === "") return;
    const s = typeof v === "string" ? v : JSON.stringify(v);
    rows.push([label, s]);
  };
  add("name", "Tool");
  add("preview", "Result preview");
  add("detail", "Detail");
  add("error", "Error");
  add("subtask_id", "Subtask");
  add("ticket_id", "Ticket");
  add("next_agent", "Next agent");
  if (rows.length === 0) return null;
  return (
    <Section title="Highlights">
      <div className="space-y-1">
        {rows.map(([label, value]) => (
          <div key={label} className="flex gap-2">
            <span className="w-28 shrink-0 text-zinc-500">{label}</span>
            <span className="min-w-0 flex-1 whitespace-pre-wrap break-words text-zinc-300">
              {value}
            </span>
          </div>
        ))}
      </div>
    </Section>
  );
}

export function AgentLog() {
  const {
    logs,
    clearLogs,
    selectedProjectId,
    logSearch,
    logKindFilter,
    autoScrollEnabled,
    setAutoScrollEnabled,
  } = useUIStore();
  const [pending, setPending] = useState<string | null>(null);
  const [selected, setSelected] = useState<AgentLogEntry | null>(null);
  const [serverDetail, setServerDetail] = useState<LogItemResponse | null>(null);
  const [selectedLoading, setSelectedLoading] = useState(false);
  const [selectedError, setSelectedError] = useState<string | null>(null);
  const scrollerRef = useRef<HTMLDivElement | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const [isAtBottom, setIsAtBottom] = useState(true);

  const filteredLogs = logs.filter((l) => {
    if (logKindFilter !== "all" && l.kind !== logKindFilter) return false;
    if (logSearch) {
      const q = logSearch.toLowerCase();
      return (
        l.agent.toLowerCase().includes(q) ||
        l.detail.toLowerCase().includes(q) ||
        humanizeLog(l).summary.toLowerCase().includes(q)
      );
    }
    return true;
  });

  const recomputeIsAtBottom = () => {
    const el = scrollerRef.current;
    if (!el) return;
    const thresholdPx = 8;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    setIsAtBottom(distanceFromBottom <= thresholdPx);
  };

  const scrollToBottom = () => {
    bottomRef.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  };

  useEffect(() => {
    if (autoScrollEnabled && isAtBottom) scrollToBottom();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [logs.length, autoScrollEnabled]);

  const resumeFrom = async (checkpointId: string) => {
    if (!selectedProjectId) return;
    setPending(checkpointId);
    try {
      await api.resumeFrom(selectedProjectId, checkpointId);
    } finally {
      setPending(null);
    }
  };

  const openDetails = async (entry: AgentLogEntry) => {
    setSelected(entry);
    setServerDetail(null);
    setSelectedError(null);
    // Live events have no DB id; render them straight from the in-memory
    // payload. Persisted rows fetch the authoritative full payload + any
    // associated checkpoint metadata.
    if (!entry.id) return;
    setSelectedLoading(true);
    try {
      const r = await api.getAgentLogItem(entry.id);
      setServerDetail(r);
    } catch (e) {
      setSelectedError(e instanceof Error ? e.message : "failed to load full record");
    } finally {
      setSelectedLoading(false);
    }
  };

  return (
    <div className="rounded border border-border bg-surface">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border px-3 py-2">
        <div className="text-sm font-medium">Agent activity</div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-zinc-500">
            {filteredLogs.length} / {logs.length}
          </span>
          <button
            onClick={() => setAutoScrollEnabled(!autoScrollEnabled)}
            className="rounded border border-border px-2 py-1 text-xs text-zinc-400 hover:bg-muted hover:text-zinc-200"
            title={autoScrollEnabled ? "Auto-scroll on" : "Auto-scroll off"}
          >
            {autoScrollEnabled ? "↓ Auto" : "↓ Paused"}
          </button>
          <button onClick={clearLogs} className="text-xs text-zinc-400 hover:text-zinc-200">
            Clear
          </button>
        </div>
      </div>

      {/* Filter bar */}
      <LogFilterBar />

      {/* Log list */}
      <div className="relative">
        <div
          ref={scrollerRef}
          onScroll={recomputeIsAtBottom}
          className="max-h-[60vh] overflow-y-auto scrollbar p-2 text-xs leading-relaxed"
        >
          {filteredLogs.length === 0 && (
            <div className="p-2 text-zinc-500">
              {logs.length === 0 ? "Waiting for events…" : "No logs match current filters."}
            </div>
          )}
          {filteredLogs.map((l, i) => (
            <LogRow
              key={l.id ?? `live-${l.ts}-${i}-${l.agent}`}
              entry={l}
              onOpen={() => void openDetails(l)}
              onResume={() => l.checkpoint_id && resumeFrom(l.checkpoint_id)}
              resuming={pending === l.checkpoint_id}
              canResume={Boolean(l.checkpoint_id && selectedProjectId)}
            />
          ))}
          <div ref={bottomRef} />
        </div>

        {!isAtBottom && filteredLogs.length > 0 && autoScrollEnabled && (
          <button
            type="button"
            onClick={scrollToBottom}
            title="Jump to latest"
            className="absolute bottom-3 right-3 rounded-full border border-border bg-surface/90 px-2 py-1 text-xs text-zinc-200 shadow-sm backdrop-blur hover:bg-surface"
          >
            ↓
          </button>
        )}
      </div>

      {/* Detail modal */}
      {selected && (
        <DetailModal
          entry={selected}
          serverDetail={serverDetail}
          loading={selectedLoading}
          error={selectedError}
          onClose={() => {
            setSelected(null);
            setServerDetail(null);
            setSelectedError(null);
          }}
        />
      )}
    </div>
  );
}
