import type { Metadata } from "next";

import { EventBridge } from "@/components/EventBridge";
import { Shell } from "@/components/Shell";

import "./globals.css";

export const metadata: Metadata = {
  title: "TDD Agentic Dev System",
  description: "Monitor, interrupt, and steer your autonomous TDD engineering team.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <EventBridge />
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}
