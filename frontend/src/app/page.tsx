"use client";

import { useEffect, useState } from "react";

import { AgentLog } from "@/components/agents/AgentLog";
import { CheckpointsPanel } from "@/components/CheckpointsPanel";
import { CrashBanner } from "@/components/CrashBanner";
import { FileExplorer } from "@/components/files/FileExplorer";
import { ProjectPicker } from "@/components/ProjectPicker";
import { api } from "@/lib/api";
import { useUIStore } from "@/lib/store";

const AGENT_COLORS: Record<string, string> = {
  project_manager: "text-amber-400",
  researcher: "text-cyan-400",
  backend_lead: "text-violet-400",
  frontend_lead: "text-violet-400",
  backend_dev: "text-emerald-400",
  frontend_dev: "text-emerald-400",
  devops: "text-orange-400",
  qa: "text-pink-400",
  human: "text-yellow-300",
};

function getAgentColor(agent: string): string {
  return AGENT_COLORS[agent] ?? "text-zinc-400";
}

function formatRelativeTime(ts: number): string {
  const diff = Date.now() - ts * 1000;
  if (diff < 60_000) return "just now";
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
  return `${Math.floor(diff / 3_600_000)}h ago`;
}

function AgentActivityPanel() {
  const { activeAgents, agentLastSeen, selectedProjectId } = useUIStore();
  const [phase, setPhase] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedProjectId) {
      setPhase(null);
      return;
    }
    let cancelled = false;
    const poll = async () => {
      try {
        const res = await api.getAgentState(selectedProjectId);
        if (cancelled) return;
        const values = res.values as Record<string, unknown> | undefined;
        const next = res.next as string[] | undefined;
        if (values && typeof values === "object") {
          const nextAgent = values.next_agent as string | undefined;
          if (nextAgent) {
            setPhase(nextAgent);
            return;
          }
        }
        if (next && next.length > 0) {
          setPhase(next[0]);
        }
      } catch {
        /* agent state not available yet */
      }
    };
    poll();
    const interval = setInterval(poll, 10_000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [selectedProjectId]);

  // Compute active agents (last seen within 15 min)
  const now = Date.now() / 1000;
  const activeList = activeAgents
    .map((name) => ({
      name,
      lastSeen: agentLastSeen[name] ?? 0,
      status: now - (agentLastSeen[name] ?? 0) < 900 ? "Active" : "Idle",
    }))
    .sort((a, b) => b.lastSeen - a.lastSeen);

  return (
    <div className="rounded border border-border bg-surface p-3">
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-sm font-medium uppercase tracking-wide text-zinc-400">
          Agent Activity
        </h2>
        {phase && (
          <span className="rounded-full bg-accent/10 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-accent">
            {phase}
          </span>
        )}
      </div>
      {activeList.length === 0 ? (
        <p className="text-xs text-zinc-500">
          No agents active. Start a run to see agent activity here.
        </p>
      ) : (
        <div className="space-y-2">
          {activeList.map((a) => (
            <div key={a.name} className="flex items-center gap-2">
              <span
                className={`font-medium tabular-nums text-xs ${getAgentColor(a.name)}`}
              >
                {a.name}
              </span>
              <span
                className={`inline-block h-1.5 w-1.5 rounded-full ${
                  a.status === "Active" ? "bg-green-400" : "bg-zinc-600"
                }`}
                title={a.status}
              />
              <span className="text-[10px] text-zinc-500">
                {formatRelativeTime(a.lastSeen)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function TicketProgress() {
  const { tickets } = useUIStore();

  const doneCount = tickets.filter((t) => t.status === "done").length;
  const total = tickets.length;
  const pct = total > 0 ? Math.round((doneCount / total) * 100) : 0;

  if (total === 0) return null;

  return (
    <div className="rounded border border-border bg-surface p-3">
      <h2 className="mb-2 text-sm font-medium uppercase tracking-wide text-zinc-400">
        Project Progress
      </h2>
      <div className="flex items-center gap-3">
        <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-zinc-800">
          <div
            className="h-full rounded-full bg-accent transition-all duration-500"
            style={{ width: `${pct}%` }}
          />
        </div>
        <span className="text-xs tabular-nums text-zinc-300">
          {doneCount}/{total} ({pct}%)
        </span>
      </div>
    </div>
  );
}

export default function DashboardPage() {
  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-semibold">Dashboard</h1>
        <p className="mt-1 text-sm text-zinc-400">
          Pick or create a project, start the agent run, and rewind by clicking any checkpoint.
        </p>
      </div>
      <ProjectPicker />
      <TicketProgress />
      <CrashBanner />
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="lg:col-span-1 flex flex-col gap-4">
          <AgentActivityPanel />
          <div>
            <h2 className="mb-2 text-sm font-medium uppercase tracking-wide text-zinc-400">
              Checkpoints
            </h2>
            <CheckpointsPanel />
          </div>
        </div>
        <div className="lg:col-span-2">
          <h2 className="mb-2 text-sm font-medium uppercase tracking-wide text-zinc-400">
            Live agent feed
          </h2>
          <AgentLog />
        </div>
      </div>
      <div>
        <h2 className="mb-2 text-sm font-medium uppercase tracking-wide text-zinc-400">
          Workspace files
        </h2>
        <FileExplorer />
      </div>
    </div>
  );
}
