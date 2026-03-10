// frontend/react-app/src/components/layout/Header.jsx
// Top header bar showing current page context, selected ticker,
// and a live V3 agent mode indicator.

import { useLocation } from "react-router-dom";
import useAgentStore from "../../store/useAgentStore";
import usePortfolioStore from "../../store/usePortfolioStore";
import { StatusBadge } from "../ui/StatusBadge";
import { format } from "date-fns";
import { useState, useEffect } from "react";

const PAGE_TITLES = {
  "/": "Dashboard",
  "/data-explorer": "Data Explorer",
  "/strategy-lab": "Strategy Lab",
  "/model-comparator": "Model Comparator",
  "/risk-dashboard": "Risk Dashboard",
  "/sentiment-monitor": "Sentiment Monitor",
  "/agent-monitor": "Agent Monitor",
  "/embedding-visualizer": "Embbedding Visualizer",
};

export function Header() {
  const { pathname } = useLocation();
  const agentStatus = useAgentStore((s) => s.agentStatus);
  const selectedTicker = usePortfolioStore((s) => s.selectedTicker);
  const [now, setNow] = useState(new Date());

  // Update clock every second
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  const title = PAGE_TITLES[pathname] || "Lumina V3";

  return (
    <header className="h-12 flex items-center justify-between px-5 bg-surface-800 border-b border-surface-500 flex-shrink-0">
      {/* Page title */}
      <h1 className="text-white font-semibold text-sm">{title}</h1>

      {/* Center - selected ticker */}
      <span className="font-mono text-brand-primary font-bold text-sm">
        {selectedTicker}
      </span>

      {/* Right - agent mode + clock */}
      <div className="flex items-center gap-4">
        {agentStatus.mode && (
          <StatusBadge
            status={agentStatus.mode}
            label={`Agent: ${agentStatus.mode.charAt(0).toUpperCase() + agentStatus.mode.slice(1)}`}
            pulse={
              agentStatus.mode === "live" || agentStatus.mode === "training"
            }
          />
        )}
        <span className="text-surface-300 text-xs font-mono">
          {format(now, "HH:mm:ss")}
        </span>
      </div>
    </header>
  );
}
