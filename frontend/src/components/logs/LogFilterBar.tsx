"use client";

import { useEffect, useState } from "react";

import { cn } from "@/lib/cn";
import { useUIStore } from "@/lib/store";
import { agentLabel, agentTextColor } from "@/lib/logFormat";

const KIND_OPTIONS = [
  "all",
  "tool_call",
  "tool_result",
  "log",
  "handoff",
  "turn_start",
  "turn_end",
  "verification_gate_failed",
  "subtask_incomplete",
  "crash",
  "error",
  "route",
  "route_fallback",
  "transient_error",
  "hitl_response",
  "stopped",
];

export function LogFilterBar() {
  const { logSearch, setLogSearch, logKindFilter, setLogKindFilter, autoScrollEnabled, setAutoScrollEnabled, logs } =
    useUIStore();
  const [debouncedSearch, setDebouncedSearch] = useState(logSearch);

  // Debounce search input
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedSearch(logSearch), 300);
    return () => clearTimeout(timer);
  }, [logSearch]);

  // Compute unique agents with their colors
  const agents = [...new Set(logs.map((l) => l.agent))];

  // Filtered count
  const filteredCount = logs.filter((l) => {
    if (logKindFilter !== "all" && l.kind !== logKindFilter) return false;
    if (debouncedSearch) {
      const q = debouncedSearch.toLowerCase();
      return (
        l.agent.toLowerCase().includes(q) ||
        l.detail.toLowerCase().includes(q)
      );
    }
    return true;
  }).length;

  const hasFilters = logSearch || logKindFilter !== "all";

  return (
    <div className="mb-4 space-y-3">
      {/* Search + controls row */}
      <div className="flex flex-wrap items-center gap-3">
        {/* Search */}
        <div className="relative flex-1 min-w-[200px]">
          <input
            type="text"
            placeholder="Search logs by agent name or keyword..."
            value={logSearch}
            onChange={(e) => setLogSearch(e.target.value)}
            className="w-full rounded border border-border bg-muted/50 px-3 py-2 pr-8 text-sm text-zinc-100 placeholder-zinc-500 focus:border-accent focus:outline-none"
          />
          {logSearch && (
            <button
              onClick={() => setLogSearch("")}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-zinc-300"
            >
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>

        {/* Auto-scroll toggle */}
        <button
          onClick={() => setAutoScrollEnabled(!autoScrollEnabled)}
          className={cn(
            "inline-flex items-center gap-1.5 rounded border px-3 py-2 text-xs transition",
            autoScrollEnabled
              ? "border-accent/50 bg-accent/10 text-accent"
              : "border-border text-zinc-400 hover:bg-muted",
          )}
          title={autoScrollEnabled ? "Auto-scroll enabled" : "Auto-scroll paused"}
        >
          <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d={autoScrollEnabled ? "M19 14l-7 7m0 0l-7-7m7 7V3" : "M19 14l-7 7m0 0l-7-7m7 7V3m0 11V3"}
            />
          </svg>
          <span>{autoScrollEnabled ? "Auto" : "Paused"}</span>
        </button>

        {/* Filtered count */}
        {hasFilters && (
          <span className="text-xs text-zinc-500">
            {filteredCount} of {logs.length}
          </span>
        )}

        {/* Clear filters */}
        {hasFilters && (
          <button
            onClick={() => {
              setLogSearch("");
              setLogKindFilter("all");
            }}
            className="rounded border border-border px-3 py-2 text-xs text-zinc-400 transition hover:bg-muted hover:text-zinc-200"
          >
            Clear
          </button>
        )}
      </div>

      {/* Kind filter pills */}
      <div className="flex flex-wrap gap-1.5">
        {KIND_OPTIONS.map((kind) => (
          <button
            key={kind}
            onClick={() => setLogKindFilter(kind)}
            className={cn(
              "rounded px-2.5 py-1 text-[11px] font-medium uppercase tracking-wide transition",
              logKindFilter === kind
                ? "bg-accent/20 text-accent border border-accent/40"
                : "bg-muted/50 text-zinc-400 border border-transparent hover:bg-muted hover:text-zinc-300",
            )}
          >
            {kind}
          </button>
        ))}
      </div>

      {/* Agent color legend */}
      {agents.length > 0 && (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
          <span className="text-zinc-500">Agents:</span>
          {agents.map((agent) => (
            <span key={agent} className={cn("font-medium", agentTextColor(agent))}>
              {agentLabel(agent)}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
