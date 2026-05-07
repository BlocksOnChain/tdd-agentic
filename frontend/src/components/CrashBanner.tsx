"use client";

import { useState } from "react";

import { api } from "@/lib/api";
import { useUIStore } from "@/lib/store";

export function CrashBanner() {
  const { crash, setCrash, selectedProjectId } = useUIStore();
  const [retrying, setRetrying] = useState(false);

  if (!crash) return null;

  const autoRetry =
    typeof crash.willRetryInSeconds === "number" && crash.willRetryInSeconds > 0;

  const retry = async () => {
    if (!selectedProjectId) return;
    setRetrying(true);
    try {
      await api.retryAgent(selectedProjectId);
      setCrash(null);
    } finally {
      setRetrying(false);
    }
  };

  return (
    <div className="rounded border border-red-500/50 bg-red-500/10 p-4 text-sm">
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1">
          <div className="text-xs font-medium uppercase tracking-wide text-red-400">
            {autoRetry ? "Transient error — auto-retrying" : "Run crashed"}
          </div>
          <div className="mt-1 text-zinc-200 break-all">{crash.message}</div>
          {crash.attempt !== undefined && (
            <div className="mt-1 text-xs text-zinc-400">
              attempt {crash.attempt}
              {autoRetry && ` · next try in ${crash.willRetryInSeconds}s`}
            </div>
          )}
        </div>
        <div className="flex shrink-0 flex-col gap-2">
          {!autoRetry && (
            <button
              className="rounded border border-red-400 bg-red-500/20 px-3 py-1.5 text-xs text-red-300 hover:bg-red-500/30 disabled:opacity-50"
              onClick={retry}
              disabled={retrying || !selectedProjectId}
            >
              {retrying ? "Retrying…" : "Retry from last checkpoint"}
            </button>
          )}
          <button
            className="rounded border border-border px-3 py-1.5 text-xs text-zinc-300 hover:bg-muted"
            onClick={() => setCrash(null)}
          >
            Dismiss
          </button>
        </div>
      </div>
    </div>
  );
}
