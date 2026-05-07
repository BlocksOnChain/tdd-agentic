"use client";

import { useEffect, useState } from "react";

import { api, ProjectT } from "@/lib/api";
import { useUIStore } from "@/lib/store";

export function ProjectPicker() {
  const { selectedProjectId, setSelectedProject } = useUIStore();
  const [projects, setProjects] = useState<ProjectT[]>([]);
  const [showNew, setShowNew] = useState(false);
  const [name, setName] = useState("");
  const [goal, setGoal] = useState("");

  const refresh = () =>
    api
      .listProjects()
      .then((p) => {
        setProjects(p);
        if (!selectedProjectId && p.length > 0) setSelectedProject(p[0].id);
        if (selectedProjectId && !p.find((x) => x.id === selectedProjectId)) {
          setSelectedProject(p.length > 0 ? p[0].id : null);
        }
      })
      .catch(() => undefined);

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const create = async () => {
    if (!name.trim()) return;
    const p = await api.createProject({ name, goal });
    setShowNew(false);
    setName("");
    setGoal("");
    setSelectedProject(p.id);
    refresh();
  };

  const startRun = async () => {
    if (!selectedProjectId) return;
    const proj = projects.find((p) => p.id === selectedProjectId);
    await api.startAgent(selectedProjectId, proj?.goal || proj?.name || "Build the project");
  };

  const stopRun = async () => {
    if (!selectedProjectId) return;
    await api.stopAgent(selectedProjectId);
  };

  const deleteProject = async () => {
    if (!selectedProjectId) return;
    const proj = projects.find((p) => p.id === selectedProjectId);
    if (!confirm(`Delete project "${proj?.name ?? selectedProjectId}"? This is irreversible.`))
      return;
    await api.deleteProject(selectedProjectId);
    refresh();
  };

  return (
    <div className="flex flex-wrap items-end gap-2">
      <div className="flex flex-col gap-1">
        <label className="text-xs text-zinc-400">Project</label>
        <select
          value={selectedProjectId ?? ""}
          onChange={(e) => setSelectedProject(e.target.value || null)}
          className="rounded border border-border bg-muted px-3 py-2 text-sm min-w-[240px]"
        >
          {projects.length === 0 && <option value="">No projects yet</option>}
          {projects.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
        </select>
      </div>
      <button
        className="rounded border border-border bg-muted px-3 py-2 text-sm hover:bg-surface"
        onClick={() => setShowNew((v) => !v)}
      >
        New project
      </button>
      <button
        className="rounded border border-accent bg-accent/20 px-3 py-2 text-sm text-accent hover:bg-accent/30 disabled:opacity-50"
        onClick={startRun}
        disabled={!selectedProjectId}
      >
        Start agent run
      </button>
      <button
        className="rounded border border-amber-500/60 bg-amber-500/10 px-3 py-2 text-sm text-amber-300 hover:bg-amber-500/20 disabled:opacity-50"
        onClick={stopRun}
        disabled={!selectedProjectId}
      >
        Stop
      </button>
      <button
        className="rounded border border-red-500/60 bg-red-500/10 px-3 py-2 text-sm text-red-300 hover:bg-red-500/20 disabled:opacity-50"
        onClick={deleteProject}
        disabled={!selectedProjectId}
      >
        Delete project
      </button>

      {showNew && (
        <div className="w-full mt-2 rounded border border-border bg-surface p-3 flex flex-col gap-2">
          <input
            className="rounded border border-border bg-muted px-3 py-2 text-sm"
            placeholder="Project name"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          <textarea
            className="rounded border border-border bg-muted px-3 py-2 text-sm"
            placeholder="Goal / brief"
            rows={3}
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
          />
          <div className="flex gap-2">
            <button
              className="rounded border border-accent bg-accent/20 px-3 py-2 text-sm text-accent"
              onClick={create}
            >
              Create
            </button>
            <button
              className="rounded border border-border px-3 py-2 text-sm"
              onClick={() => setShowNew(false)}
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
