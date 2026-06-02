"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { cn } from "@/lib/cn";
import { useUIStore } from "@/lib/store";

const NAV = [
  { href: "/", label: "Dashboard" },
  { href: "/tickets", label: "Tickets" },
  { href: "/logs", label: "Logs" },
  { href: "/hitl", label: "HITL" },
];

function HitlBadge() {
  const hitlPending = useUIStore((s) => s.interrupts.filter((i) => !i.dismissed && !i.answered).length);
  if (hitlPending <= 0) return null;
  return (
    <span className="ml-1.5 inline-flex h-5 min-w-[1.25rem] items-center justify-center rounded-full bg-amber-400 px-1.5 text-[10px] font-bold text-amber-900">
      {hitlPending}
    </span>
  );
}

export function Shell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
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
                "rounded px-3 py-2 text-sm hover:bg-muted flex items-center justify-between",
                pathname === item.href && "bg-muted text-accent"
              )}
            >
              <span>{item.label}</span>
              {item.label === "HITL" && <HitlBadge />}
            </Link>
          ))}
        </nav>
      </aside>
      <main className="overflow-y-auto p-6 scrollbar">
        {children}
      </main>
    </div>
  );
}
