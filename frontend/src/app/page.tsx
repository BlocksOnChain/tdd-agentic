import { AgentLog } from "@/components/agents/AgentLog";
import { CheckpointsPanel } from "@/components/CheckpointsPanel";
import { CrashBanner } from "@/components/CrashBanner";
import { ProjectPicker } from "@/components/ProjectPicker";

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
      <CrashBanner />
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="lg:col-span-1 flex flex-col gap-4">
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
    </div>
  );
}
