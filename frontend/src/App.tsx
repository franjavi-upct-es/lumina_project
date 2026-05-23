// frontend/src/App.tsx
//
// Top-level application shell.
//
// Routing
// -------
// We deliberately avoid react-router-dom because (a) it would be a new
// dependency for a 3-page app, and (b) the page set is very stable.
// Instead, a tiny in-house view switcher reads the URL hash (e.g.
// "#/backtest") and renders the corresponding page. Forward / back
// buttons work because we use the hash, and deep linking works because
// the hash is part of the URL.

import { useEffect, useState } from "react";
import { ErrorBoundary } from "./components/common/ErrorBoundary";
import { ArenaPage } from "./pages/ArenaPage";
import { Backtest } from "./pages/Backtest";
import { Dashboard } from "./pages/Dashboard";
import { Settings } from "./pages/Settings";

type View = "dashboard" | "backtest" | "arena" | "settings";

function hashToView(hash: string): View {
  // Strip the leading "#/" or "#"; default to dashboard.
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
      <NavBar active={view} />
      {view === "dashboard" && <Dashboard />}
      {view === "backtest" && <Backtest />}
      {view === "arena" && <ArenaPage />}
      {view === "settings" && <Settings />}
    </ErrorBoundary>
  );
}

function NavBar({ active }: { active: View }) {
  const linkStyle = (target: View): React.CSSProperties => ({
    padding: "8px 16px",
    textDecoration: "none",
    color: active === target ? "#0969da" : "#24292f",
    fontWeight: active === target ? 600 : 400,
    borderBottom: active === target ? "2px solid #0969da" : "2px solid transparent",
  });

  return (
    <nav
      style={{
        display: "flex",
        alignItems: "center",
        padding: "0 16px",
        borderBottom: "1px solid #d0d7de",
        background: "#f6f8fa",
      }}
    >
      <a href="#/dashboard" style={linkStyle("dashboard")}>Dashboard</a>
      <a href="#/backtest" style={linkStyle("backtest")}>Backtest</a>
      <a href="#/arena" style={linkStyle("arena")}>Arena</a>
      <a href="#/settings" style={linkStyle("settings")}>Settings</a>
    </nav>
  );
}
