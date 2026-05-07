import { ProjectPicker } from "@/components/ProjectPicker";
import { TicketBoard } from "@/components/tickets/TicketBoard";

export default function TicketsPage() {
  return (
    <div className="flex flex-col gap-4">
      <h1 className="text-2xl font-semibold">Tickets</h1>
      <ProjectPicker />
      <TicketBoard />
    </div>
  );
}
