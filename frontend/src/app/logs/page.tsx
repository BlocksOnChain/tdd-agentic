import { AgentLog } from "@/components/agents/AgentLog";

export default function LogsPage() {
  return (
    <div className="flex flex-col gap-4">
      <h1 className="text-2xl font-semibold">Logs</h1>
      <p className="text-sm text-zinc-400">
        Streaming every agent event live from the LangGraph orchestration backend.
      </p>
      <AgentLog />
    </div>
  );
}
