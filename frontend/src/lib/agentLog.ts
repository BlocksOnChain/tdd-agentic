const MAX_DETAIL_CHARS = 2000;

function truncate(text: string, limit = MAX_DETAIL_CHARS): string {
  if (text.length <= limit) return text;
  return `${text.slice(0, limit)}…`;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (typeof value !== "object" || value === null) return null;
  return value as Record<string, unknown>;
}

export function formatAgentLogDetail(kind: string, payload: Record<string, unknown>): string {
  if (!payload || Object.keys(payload).length === 0) return kind;

  if (kind === "tool_result") {
    const name = typeof payload.name === "string" ? payload.name : "tool";
    const preview = typeof payload.preview === "string" ? payload.preview : "";
    return truncate(`${name}: ${preview}`);
  }

  if (kind === "research_artifacts") {
    const tools = asRecord(payload.tools);
    const toolCalls = asRecord(tools?.tool_calls);
    const sync = asRecord(payload.workspace_sync);
    const files = Array.isArray(payload.workspace_files) ? payload.workspace_files : [];
    const paths = files
      .map((item) => asRecord(item)?.path)
      .filter((path): path is string => typeof path === "string")
      .join(", ");
    const ragChunks = typeof tools?.rag_ingest_chunks === "number" ? tools.rag_ingest_chunks : 0;
    const indexed = typeof sync?.total_chunks === "number" ? sync.total_chunks : 0;
    return truncate(
      `research artifacts · tool_calls=${JSON.stringify(toolCalls ?? {})} · rag_ingest_chunks=${ragChunks} · workspace_files=${files.length} · indexed_chunks=${indexed}${paths ? ` · paths=${paths}` : ""}`,
    );
  }

  if (kind === "route" || kind === "route_fallback") {
    const target = typeof payload.next_agent === "string" ? payload.next_agent : "?";
    const rationale = typeof payload.rationale === "string" ? payload.rationale : "";
    return truncate(`→ ${target}: ${rationale}`);
  }

  if (kind === "turn_end") {
    if (Object.keys(payload).length === 0) return "turn complete";
    return truncate(JSON.stringify(payload));
  }

  if (kind === "turn_start") {
    return "turn started";
  }

  for (const key of ["preview", "rationale", "error", "message"] as const) {
    const value = payload[key];
    if (typeof value === "string" && value.trim()) return truncate(value);
  }

  if (kind === "log" && asRecord(payload.update)) {
    return truncate(JSON.stringify(payload.update));
  }

  return truncate(JSON.stringify(payload));
}
