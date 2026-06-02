"use client";

import { useEffect, useRef, useState } from "react";

import { api } from "@/lib/api";
import { cn } from "@/lib/cn";
import { useUIStore } from "@/lib/store";

/** File explorer that shows a tree of workspace files + a read-only viewer panel. */

type Node =
  | { type: "dir"; name: string; children: Node[]; expanded: boolean }
  | { type: "file"; name: string; path: string; content: string | null; loading: boolean };

const FILE_ICONS: Record<string, string> = {
  ts: "TS",
  tsx: "TSX",
  py: "PY",
  json: "{}",
  md: "MD",
  txt: "T",
  css: "C",
  html: "HTML",
  sql: "DB",
  cfg: "CFG",
  toml: "TOML",
  env: "ENV",
  sh: "SH",
  rs: "RS",
  go: "GO",
  js: "JS",
  jsx: "JSX",
};

function getIcon(ext: string): string {
  return FILE_ICONS[ext] ?? "F";
}

function getColorForExt(ext: string): string {
  const map: Record<string, string> = {
    ts: "text-cyan-400",
    tsx: "text-cyan-400",
    py: "text-yellow-400",
    json: "text-amber-300",
    md: "text-green-400",
    css: "text-blue-400",
    html: "text-orange-400",
    sql: "text-pink-400",
    env: "text-red-400",
    toml: "text-amber-400",
    cfg: "text-amber-400",
    sh: "text-green-300",
    rs: "text-orange-500",
    go: "text-cyan-300",
    js: "text-yellow-300",
    jsx: "text-cyan-300",
  };
  return map[ext] ?? "text-zinc-400";
}

function langForExt(ext: string): string {
  const map: Record<string, string> = {
    ts: "typescript",
    tsx: "typescript",
    py: "python",
    json: "json",
    md: "markdown",
    css: "css",
    html: "html",
    sql: "sql",
    rs: "rust",
    go: "go",
    js: "javascript",
    jsx: "jsx",
  };
  return map[ext] ?? "text";
}

function buildTree(files: string[]): Node {
  const root: Node = { type: "dir", name: "/", children: [], expanded: false };
  for (const f of files) {
    const parts = f.split("/");
    let current = root;
    for (let i = 0; i < parts.length; i++) {
      const part = parts[i];
      const isLast = i === parts.length - 1;
      if (isLast) {
        const ext = part.includes(".") ? part.split(".").pop() ?? "" : "";
        const child: Node = { type: "file", name: part, path: f, content: null, loading: false };
        current.children.push(child);
      } else {
        let child = current.children.find(
          (c) => c.type === "dir" && c.name === part,
        ) as Node | undefined;
        if (!child) {
          child = { type: "dir", name: part, children: [], expanded: false };
          current.children.push(child);
        }
        current = child;
      }
    }
  }
  return root;
}

function findFileNode(node: Node, path: string): Node | null {
  if (node.type === "file" && node.path === path) return node;
  if (node.type === "dir") {
    for (const child of node.children) {
      const found = findFileNode(child, path);
      if (found) return found;
    }
  }
  return null;
}

export function FileExplorer() {
  const { selectedProjectId } = useUIStore();
  const [tree, setTree] = useState<Node | null>(null);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedDirs, setExpandedDirs] = useState<Set<string>>(new Set(["/"]));

  const fetchFiles = async () => {
    if (!selectedProjectId) {
      setTree(null);
      setSelectedPath(null);
      setSelectedNode(null);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await api.listFiles(selectedProjectId);
      const t = buildTree(res.files);
      setTree(t);
      setExpandedDirs(new Set(["/"]));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load files");
      setTree(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchFiles();
  }, [selectedProjectId]);

  // Fetch file content when a file is selected
  useEffect(() => {
    if (!selectedPath || !selectedNode || selectedNode.content !== null) return;
    if (!selectedProjectId) return;
    let cancelled = false;
    setSelectedNode((prev) => (prev ? { ...prev, loading: true } : null));
    api
      .getFileContent(selectedProjectId, selectedPath)
      .then((res) => {
        if (cancelled) return;
        if (selectedNode) {
          setSelectedNode((prev) => (prev ? { ...prev, content: res.content } : null));
        }
      })
      .catch(() => {
        /* silently fail — content stays null */
      })
      .finally(() => {
        if (cancelled) return;
        setSelectedNode((prev) => (prev ? { ...prev, loading: false } : null));
      });
    return () => {
      cancelled = true;
    };
  }, [selectedPath, selectedProjectId, selectedNode]);

  const toggleDir = (path: string) => {
    setExpandedDirs((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  };

  const selectFile = (node: Node) => {
    if (node.type !== "file") return;
    setSelectedPath(node.path);
    setSelectedNode(node);
  };

  if (!selectedProjectId) {
    return (
      <div className="rounded border border-border bg-surface p-3">
        <p className="text-xs text-zinc-500">
          Select a project to browse files.
        </p>
      </div>
    );
  }

  return (
    <div className="flex h-[500px] overflow-hidden rounded border border-border bg-surface">
      {/* File tree sidebar */}
      <div className="w-56 flex-shrink-0 border-r border-border overflow-y-auto p-2 scrollbar">
        <div className="mb-2 flex items-center justify-between">
          <h2 className="text-sm font-medium uppercase tracking-wide text-zinc-400">
            Files
          </h2>
          <button
            onClick={fetchFiles}
            disabled={loading}
            className="text-[10px] text-zinc-400 hover:text-zinc-200 disabled:opacity-50"
          >
            {loading ? "…" : "↻"}
          </button>
        </div>
        {error && (
          <p className="text-[10px] text-red-400">{error}</p>
        )}
        {!error && tree && (
          <TreeRoot node={tree} expanded={expandedDirs} onToggle={toggleDir} onSelect={selectFile} depth={0} selectedPath={selectedPath} />
        )}
        {!error && !loading && !tree && (
          <p className="text-xs text-zinc-500">No files found.</p>
        )}
      </div>

      {/* File content viewer */}
      <div className="flex min-w-0 flex-1 flex-col">
        {selectedNode ? (
          <>
            {/* Header */}
            <div className="flex items-center justify-between border-b border-border px-3 py-2">
              <div className="flex items-center gap-2 truncate">
                <span
                  className={cn(
                    "text-xs font-bold",
                    getColorForExt(
                      selectedNode.name.includes(".")
                        ? selectedNode.name.split(".").pop() ?? ""
                        : "",
                    ),
                  )}
                >
                  {getIcon(
                    selectedNode.name.includes(".")
                      ? selectedNode.name.split(".").pop() ?? ""
                      : "",
                  )}
                </span>
                <span className="text-sm text-zinc-200 truncate">
                  {selectedNode.name}
                </span>
                <span className="text-[10px] text-zinc-500">
                  {selectedPath}
                </span>
              </div>
              {selectedNode.loading && (
                <span className="text-xs text-zinc-500">Loading…</span>
              )}
            </div>

            {/* Content */}
            <div className="flex-1 overflow-auto p-3 font-mono text-xs scrollbar">
              {selectedNode.content !== null ? (
                <pre className="whitespace-pre-wrap break-words text-zinc-300 leading-relaxed">
                  {selectedNode.content}
                </pre>
              ) : selectedNode.loading ? (
                <span className="text-zinc-500">Loading content…</span>
              ) : (
                <span className="text-zinc-500">Click to load content…</span>
              )}
            </div>
          </>
        ) : (
          <div className="flex flex-1 items-center justify-center text-zinc-500">
            <p className="text-sm">Select a file to view</p>
          </div>
        )}
      </div>
    </div>
  );
}

function TreeRoot({
  node,
  expanded,
  onToggle,
  onSelect,
  depth,
  selectedPath,
}: {
  node: Node;
  expanded: Set<string>;
  onToggle: (path: string) => void;
  onSelect: (node: Node) => void;
  depth: number;
  selectedPath: string | null;
}) {
  if (node.type === "dir") {
    const path = node.name === "/" ? "/" : node.name.split("/").join("/");
    const isExpanded = expanded.has(path);
    const childDirs = node.children.filter(
      (c) => c.type === "dir",
    ) as Node[];
    const childFiles = node.children.filter(
      (c) => c.type === "file",
    ) as Array<Node & { type: "file" }>;

    return (
      <div>
        <button
          onClick={() => onToggle(path)}
          className="flex w-full items-center gap-1 rounded px-1.5 py-0.5 text-left text-xs text-zinc-300 hover:bg-muted/40"
          style={{ paddingLeft: `${depth * 12 + 6}px` }}
        >
          <span className="text-zinc-500">
            {isExpanded ? "▾" : "▸"}
          </span>
          <span className="text-amber-400">📁</span>
          <span className="truncate font-medium">{node.name}</span>
          {childDirs.length > 0 && (
            <span className="text-[10px] text-zinc-600">
              ({childFiles.length + childDirs.length})
            </span>
          )}
        </button>
        {isExpanded &&
          node.children.map((child) => (
            <TreeRoot
              key={child.name}
              node={child}
              expanded={expanded}
              onToggle={onToggle}
              onSelect={onSelect}
              depth={depth + 1}
              selectedPath={selectedPath}
            />
          ))}
      </div>
    );
  }

  // File node
  const ext = node.name.includes(".") ? node.name.split(".").pop() ?? "" : "";
  return (
    <button
      onClick={() => onSelect(node)}
      className={cn(
        "flex w-full items-center gap-1.5 rounded px-1.5 py-0.5 text-left text-xs transition",
        selectedPath === node.path
          ? "bg-accent/15 text-accent"
          : "text-zinc-300 hover:bg-muted/40",
      )}
      style={{ paddingLeft: `${depth * 12 + 24}px` }}
    >
      <span
        className={cn(
          "text-[10px] font-bold",
          getColorForExt(ext),
        )}
      >
        {getIcon(ext)}
      </span>
      <span className="truncate">{node.name}</span>
    </button>
  );
}
