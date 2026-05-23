// frontend/src/pages/Settings.tsx
//
// User-configurable settings.
//
// Persistence
// -----------
// Settings are stored in window.localStorage under the key
// "lumina:settings" as a JSON blob. They are read on mount and written
// on every change. A page reload picks them up automatically. The API
// client doesn't re-read settings on each call — so changes to API
// base URL or API key require a page reload to take effect, which we
// communicate via an inline hint.
//
// Theme
// -----
// Two themes: "light" (default) and "dark". The application currently
// renders most colours inline, so the theme just toggles a CSS class
// on the document body which a future stylesheet can hook into. The
// hooks are in place so the wiring is forward-compatible even though
// the visual change today is minimal.

import { useEffect, useState } from "react";

const STORAGE_KEY = "lumina:settings";

type Theme = "light" | "dark";

interface SettingsState {
  apiBase: string;
  apiKey: string;
  wsBase: string;
  theme: Theme;
}

const DEFAULT_SETTINGS: SettingsState = {
  apiBase: "http://localhost:8000",
  wsBase: "ws://localhost:8000",
  apiKey: "",
  theme: "light",
};

function loadSettings(): SettingsState {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_SETTINGS;
    const parsed = JSON.parse(raw) as Partial<SettingsState>;
    return { ...DEFAULT_SETTINGS, ...parsed };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

function saveSettings(s: SettingsState): void {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
}

function applyTheme(theme: Theme): void {
  // We toggle a body class. Stylesheet hooks (.theme-dark) can be
  // added later without changing this code.
  document.body.classList.remove("theme-light", "theme-dark");
  document.body.classList.add(`theme-${theme}`);
}

export function Settings() {
  const [settings, setSettings] = useState<SettingsState>(loadSettings);
  const [savedHint, setSavedHint] = useState<string | null>(null);

  useEffect(() => {
    applyTheme(settings.theme);
  }, [settings.theme]);

  const update = <K extends keyof SettingsState>(field: K, value: SettingsState[K]) => {
    const next = { ...settings, [field]: value };
    setSettings(next);
    saveSettings(next);
    setSavedHint("Saved");
    // Clear the saved hint after a short delay so it doesn't linger.
    window.setTimeout(() => setSavedHint(null), 1500);
  };

  return (
    <div style={{ padding: 16, maxWidth: 720, margin: "0 auto" }}>
      <h1 style={{ marginTop: 0 }}>Settings</h1>

      <section style={panelStyle}>
        <h2 style={{ marginTop: 0, fontSize: 16 }}>Backend connection</h2>
        <div style={{ display: "grid", gridTemplateColumns: "140px 1fr", gap: 12, alignItems: "center" }}>
          <label htmlFor="s-api">REST API base</label>
          <input
            id="s-api"
            type="text"
            value={settings.apiBase}
            onChange={(e) => update("apiBase", e.target.value)}
            placeholder="http://localhost:8000"
          />
          <label htmlFor="s-ws">WebSocket base</label>
          <input
            id="s-ws"
            type="text"
            value={settings.wsBase}
            onChange={(e) => update("wsBase", e.target.value)}
            placeholder="ws://localhost:8000"
          />
          <label htmlFor="s-key">API key</label>
          <input
            id="s-key"
            type="password"
            value={settings.apiKey}
            onChange={(e) => update("apiKey", e.target.value)}
            placeholder="(empty for local dev)"
          />
        </div>
        <small style={{ display: "block", marginTop: 8, color: "#8c959f", fontSize: 12 }}>
          Reload the page for changes to take effect.
        </small>
      </section>

      <section style={panelStyle}>
        <h2 style={{ marginTop: 0, fontSize: 16 }}>Appearance</h2>
        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          <label htmlFor="s-theme">Theme</label>
          <select
            id="s-theme"
            value={settings.theme}
            onChange={(e) => update("theme", e.target.value as Theme)}
            style={{ padding: "4px 8px" }}
          >
            <option value="light">Light</option>
            <option value="dark">Dark</option>
          </select>
        </div>
      </section>

      {savedHint && (
        <div
          style={{
            position: "fixed",
            bottom: 16,
            right: 16,
            padding: "8px 16px",
            background: "#1a7f37",
            color: "#ffffff",
            borderRadius: 4,
            fontSize: 13,
            boxShadow: "0 2px 6px rgba(0,0,0,0.2)",
          }}
        >
          {savedHint}
        </div>
      )}
    </div>
  );
}

const panelStyle: React.CSSProperties = {
  border: "1px solid #d0d7de",
  borderRadius: 6,
  padding: 16,
  marginBottom: 16,
  background: "#ffffff",
};
