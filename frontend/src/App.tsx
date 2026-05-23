// frontend/src/App.tsx
//
// Top-level application shell. Routes are driven by the URL hash so we
// don't need react-router for a 4-page app.

import { useEffect, useState } from "react";
import { AppShell } from "./components/common/AppShell";
import { ErrorBoundary } from "./components/common/ErrorBoundary";
import { ArenaPage } from "./pages/ArenaPage";
import { Backtest } from "./pages/Backtest";
import { Dashboard } from "./pages/Dashboard";
import { Settings } from "./pages/Settings";

type View = "dashboard" | "backtest" | "arena" | "settings";

function hashToView(hash: string): View {
  const stripped = hash.replace(/^#\/?/, "");
  if (stripped === "backtest") return "backtest";
  if (stripped === "arena") return "arena";
  if (stripped === "settings") return "settings";
  return "dashboard";
}

export default function App() {
  const [view, setView] = useState<View>(hashToView(window.location.hash));

  useEffect(() => {
    const onHashChange = () => setView(hashToView(window.location.hash));
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  return (
    <ErrorBoundary>
      <AppShell active={view} arenaBadge={2}>
        {view === "dashboard" && <Dashboard />}
        {view === "backtest" && <Backtest />}
        {view === "arena" && <ArenaPage />}
        {view === "settings" && <Settings />}
      </AppShell>
    </ErrorBoundary>
  );
}
