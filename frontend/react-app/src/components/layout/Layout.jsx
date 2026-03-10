// frontend/react-app/src/components/layout/Layout.jsx
// Root layout wrapper that renders the Sidebar, Header, and main content area.
// Also mounts the WebSocket listener for agent live decisions here so the
// connection persists across page navigations (unlike Streamlit's per-page model).

import { Outlet } from "react-router-dom";
import { Sidebar } from "./Sidebar";
import { Header } from "./Header";
import { useAgentWebSocket } from "../../hooks/useAgent";
import useAgentStore from "../../store/useAgentStore";
import { AlertBanner } from "../ui/AlertBanner";
import { SAFETY_STATUS } from "../../constants";

/** Mounts once and maintains the WebSocket connection for the entire session. */
function LiveAgentConnector() {
  useAgentWebSocket();
  return null;
}

export function Layout() {
  const safetyStatus = useAgentStore((s) => s.safetyStatus);
  const activeBreakers = useAgentStore((s) => s.activeBreakers);

  return (
    <div className="flex h-screen bg-surface-900 overflow-hidden">
      {/* Persistent WebSocket listener */}
      <LiveAgentConnector />

      {/* Sidebar */}
      <Sidebar />

      {/* Main area */}
      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
        <Header />

        {/* Circuit breaker alert banner */}
        {safetyStatus === SAFETY_STATUS.CRITICAL && (
          <AlertBanner
            variant="danger"
            title="🚨 Circuit Breaker Active"
            message={activeBreakers.map((b) => b.type).join(" · ")}
            className="mx-4 mt-3"
          />
        )}
        {safetyStatus === SAFETY_STATUS.GUARDED && (
          <AlertBanner
            variant="warning"
            title="⚠️ Elevated Uncertainty"
            message="Safety Arbitrator is in guarded mode. Agent sizing reduced."
            className="mx-4 mt-3"
          />
        )}

        {/* Page content */}
        <main className="flex-1 overflow-y-auto p-5">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
