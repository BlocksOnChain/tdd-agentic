"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { InterruptPanel, useHumanInputCount } from "@/components/hitl/InterruptPanel";
import { cn } from "@/lib/cn";
import { useUIStore } from "@/lib/store";

const NAV = [
  { href: "/", label: "Dashboard" },
  { href: "/human", label: "Human input" },
  { href: "/tickets", label: "Tickets" },
  { href: "/logs", label: "Logs" },
];

export function Shell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const humanInputCount = useHumanInputCount();
  const hitlPending = humanInputCount > 0;
  return (
    <div className="grid min-h-screen grid-cols-[220px_1fr]">
      <aside className="border-r border-border bg-surface p-4 flex flex-col gap-2">
        <div className="text-lg font-semibold mb-4">TDD Agentic</div>
        <nav className="flex flex-col gap-1">
          {NAV.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center justify-between rounded px-3 py-2 text-sm hover:bg-muted",
                pathname === item.href && "bg-muted text-accent"
              )}
            >
              <span>{item.label}</span>
              {item.href === "/human" && hitlPending ? (
                <span className="rounded-full bg-amber-500/20 px-2 py-0.5 text-[10px] font-semibold text-amber-300">
                  {humanInputCount}
                </span>
              ) : null}
            </Link>
          ))}
        </nav>
      </aside>
      <main className="overflow-y-auto p-6 scrollbar">
        {hitlPending && (
          <div className="mb-6 rounded-lg border border-amber-500/50 bg-amber-950/40 p-4">
            <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-amber-300">
              Human input required — the agent graph is paused until you answer below
            </div>
            <p className="mb-3 text-xs text-amber-100/80">
              Type your reply and press <span className="font-medium">Send &amp; resume</span>. Works
              from any page (Dashboard, Tickets, Logs).
            </p>
            <InterruptPanel />
          </div>
        )}
        {children}
      </main>
    </div>
  );
}
