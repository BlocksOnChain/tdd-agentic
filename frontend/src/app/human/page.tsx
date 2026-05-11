import { InterruptPanel } from "@/components/hitl/InterruptPanel";
import { TicketQuestionsPanel } from "@/components/hitl/TicketQuestionsPanel";
import { ProjectPicker } from "@/components/ProjectPicker";

export default function HumanInputPage() {
  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-semibold">Human input</h1>
        <p className="mt-1 text-sm text-zinc-400">
          Answer agent interrupts and ticket questions. Replies resume the LangGraph run for the
          selected project.
        </p>
      </div>
      <ProjectPicker />
      <section className="flex flex-col gap-3">
        <h2 className="text-sm font-medium uppercase tracking-wide text-zinc-400">
          Graph interrupts
        </h2>
        <InterruptPanel />
      </section>
      <section className="flex flex-col gap-3">
        <h2 className="text-sm font-medium uppercase tracking-wide text-zinc-400">
          Ticket questions
        </h2>
        <TicketQuestionsPanel />
      </section>
    </div>
  );
}
