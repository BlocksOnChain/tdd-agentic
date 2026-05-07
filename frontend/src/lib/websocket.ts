"use client";

import { useEffect, useRef } from "react";

export type WSEvent = {
  type: "hello" | "agent" | "ticket" | "interrupt" | "log" | "project";
  payload: Record<string, unknown>;
  ts?: number;
  project_id?: string | null;
};

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws";

export function useEventStream(handler: (e: WSEvent) => void) {
  const ref = useRef<WebSocket | null>(null);
  useEffect(() => {
    let stopped = false;

    const connect = () => {
      if (stopped) return;
      const ws = new WebSocket(WS_URL);
      ref.current = ws;
      ws.onmessage = (msg) => {
        try {
          handler(JSON.parse(msg.data));
        } catch {
          /* ignore */
        }
      };
      ws.onclose = () => {
        if (!stopped) setTimeout(connect, 1500);
      };
      ws.onerror = () => ws.close();
    };
    connect();

    return () => {
      stopped = true;
      ref.current?.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
}
