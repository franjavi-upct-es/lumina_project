// frontend/react-app/src/components/layout/Sidebar.jsx
// Navigation sidebar – React equivalent of the Streamlit sidebar.
// Includes global ticker search, system status indicators, and
// the V3 Safety Arbitrator live status badge.

import { useState } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { clsx } from "clsx";
import { useQuery } from "@tanstack/react-query";
import { fetchHealth } from "../../services/api";
import usePortfolioStore from "../../store/usePortfolioStore";
import useAgentStore from "../../store/useAgentStore";
import { StatusBadge } from "../ui/StatusBadge";
import { SAFETY_STATUS } from "../../constants";

// Navigation items – maps exactly to the 7 Streamlit pages + home
const NAV_ITEMS = [
  { path: "/", icon: "🏠", label: "Dashboard" },
  { path: "/data-explorer", icon: "📊", label: "Data Explorer" },
  { path: "/strategy-lab", icon: "🧪", label: "Strategy Lab" },
  { path: "/model-comparator", icon: "🤖", label: "Model Comparator" },
  { path: "/risk-dashboard", icon: "⚠️", label: "Risk Dashboard" },
  { path: "/sentiment-monitor", icon: "📰", label: "Sentiment Monitor" },
  // V3 NEW pages
  { path: "/agent-monitor", icon: "🧠", label: "Agent Monitor", badge: "V3" },
  {
    path: "/embedding-visualizer",
    icon: "🔬",
    label: "Embedding Visualizer",
    badge: "V3",
  },
];

export function Sidebar() {
  const navigate = useNavigate();
  const { selectedTicker, setSelectedTicker } = usePortfolioStore();
  const [tickerInput, setTickerInput] = useState(selectedTicker);
  const safetyStatus = useAgentStore((s) => s.safetyStatus);
  const wsStatus = useAgentStore((s) => s.wsStatus);

  // Health check every 30 seconds
  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: () => fetchHealth().then((r) => r.data),
    refetchInterval: 30_000,
    retry: 1,
  });

  const apiOnline = Boolean(health);

  function handleQuickAnalysis(e) {
    e.preventDefault();
    const t = tickerInput.toUpperCase().trim();
    if (t) {
      setSelectedTicker(t);
      navigate("/data-explorer");
    }
  }

  return (
    <aside className="flex flex-col h-full bg-surface-800 border-r border-surface-500 w-56 flex-shrink-0">
      {/* Logo */}
      <div className="px-4 py-5 border-b border-surface-500">
        <div className="text-transparent bg-clip-text bg-gradient-to-r from-brand-primary to-brand-secondary font-bold text-lg leading-tight">
          🚀 Lumina V3
        </div>
        <p className="text-surface-300 text-xs mt-0.5">
          Deep Fusion Architecture
        </p>
      </div>

      {/* Quick Analysis */}
      <div className="px-3 py-3 border-b border-surface-500">
        <form onSubmit={handleQuickAnalysis} className="flex gap-1">
          <input
            type="text"
            value={tickerInput}
            onChange={(e) => setTickerInput(e.target.value.toUpperCase())}
            placeholder="AAPL"
            className="flex-1 bg-surface-600 border border-surface-500 rounded px-2 py-1.5 text-sm text-white placeholder-surface-300 focus:outline-none focus:border-brand-primary font-mono"
            maxLength={5}
          />
          <button
            type="submit"
            className="px-2 py-1.5 bg-brand-primary/20 hover:bg-brand-primary/30 text-brand-primary rounded text-sm transition-colors"
            title="Quick Analysis"
          >
            🔍
          </button>
        </form>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-2 py-3 overflow-y-auto">
        <p className="text-surface-300 text-xs font-semibold uppercase tracking-wider px-2 mb-2">
          Navigation
        </p>
        <ul className="space-y-0.5">
          {NAV_ITEMS.map(({ path, icon, label, badge }) => (
            <li key={path}>
              <NavLink
                to={path}
                end={path === "/"}
                className={({ isActive }) =>
                  clsx(
                    "flex items-center gap-2 px-2 py-2 rounded-lg text-sm transition-colors",
                    isActive
                      ? "bg-brand-primary/20 text-white font-medium"
                      : "text-surface-300 hover:text-white hover:bg-surface-600",
                  )
                }
              >
                <span className="text-base">{icon}</span>
                <span className="flex-1 truncate">{label}</span>
                {badge && (
                  <span className="text-[9px] font-bold bg-brand-cyan/20 text-brand-cyan px-1 py-0.5 rounded">
                    {badge}
                  </span>
                )}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>

      {/* System Status */}
      <div className="px-3 py-3 border-t border-surface-500 space-y-2">
        <p className="text-surface-300 text-xs font-semibold uppercase tracking-wider">
          System Status
        </p>

        <div className="flex items-center justify-between">
          <span className="text-surface-300 text-xs">API</span>
          <StatusBadge status={apiOnline ? "online" : "offline"} />
        </div>

        <div className="flex items-center justify-between">
          <span className="text-surface-300 text-xs">Agent WS</span>
          <StatusBadge status={wsStatus} />
        </div>

        <div className="flex items-center justify-between">
          <span className="text-surface-300 text-xs">Safety</span>
          <StatusBadge
            status={safetyStatus}
            pulse={safetyStatus === SAFETY_STATUS.CRITICAL}
          />
        </div>
      </div>
    </aside>
  );
}
