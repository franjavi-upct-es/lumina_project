// frontend/src/pages/Settings.tsx
//
// User-configurable settings. The backend connection (REST + WebSocket
// URLs) is intentionally hidden — the deployment ships with sane
// defaults that operators should not need to touch. Only the API key,
// appearance preferences and risk policy live here.
//
// Persistence
// -----------
// Settings live in window.localStorage under the key "lumina:settings".
// They are read on mount and written on every change. The API client
// doesn't re-read settings on each call, so changes to the API key
// require a page reload to take effect — surfaced via an inline hint.

import { useEffect, useState } from "react";

const STORAGE_KEY = "lumina:settings";

type Theme = "dark" | "dim" | "light" | "system";
type Density = "comfortable" | "compact" | "dense";
type NumberFormat = "us" | "eu" | "iso";

interface SettingsState {
  apiKey: string;
  theme: Theme;
  density: Density;
  numberFormat: NumberFormat;
  // Risk policy (purely client-side display knobs for now).
  maxGrossExposure: string;
  maxSinglePosition: string;
  stopLoss: string;
  gateThreshold: string;
  autoFlattenSigma: string;
}

const DEFAULT_SETTINGS: SettingsState = {
  apiKey: "",
  theme: "dark",
  density: "dense",
  numberFormat: "us",
  maxGrossExposure: "150",
  maxSinglePosition: "30",
  stopLoss: "-3.5",
  gateThreshold: "0.31",
  autoFlattenSigma: "0.50",
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
  document.body.classList.remove("theme-light", "theme-dim", "theme-dark");
  if (theme !== "system") document.body.classList.add(`theme-${theme}`);
}

export function Settings() {
  const [settings, setSettings] = useState<SettingsState>(loadSettings);
  const [savedHint, setSavedHint] = useState<string | null>(null);
  const [revealKey, setRevealKey] = useState(false);

  useEffect(() => {
    applyTheme(settings.theme);
  }, [settings.theme]);

  const update = <K extends keyof SettingsState>(field: K, value: SettingsState[K]) => {
    const next = { ...settings, [field]: value };
    setSettings(next);
    saveSettings(next);
    setSavedHint("Saved");
    window.setTimeout(() => setSavedHint(null), 1200);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16, maxWidth: 1100 }}>
      <header>
        <h1 className="lx-page-title">Configure Lumina</h1>
        <div className="lx-page-subtitle">Lumina · system preferences</div>
      </header>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <section className="lx-panel" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <header>
            <div className="lx-label">API Access</div>
            <div style={{ fontSize: 11, color: "var(--text-dim)", marginTop: 4 }}>
              Backend URLs are configured by the deployment and do not need to be changed.
            </div>
          </header>
          <div className="lx-field">
            <label>API Key</label>
            <div style={{ display: "flex", gap: 6 }}>
              <input
                type={revealKey ? "text" : "password"}
                value={settings.apiKey}
                onChange={(e) => update("apiKey", e.target.value)}
                placeholder="•••• •••• •••• ••••"
                autoComplete="off"
                style={{ flex: 1, fontFamily: "var(--font-mono)" }}
              />
              <button
                className="lx-btn ghost"
                type="button"
                onClick={() => setRevealKey((r) => !r)}
              >
                {revealKey ? "HIDE" : "SHOW"}
              </button>
            </div>
            <small style={{ color: "var(--text-dim)", fontSize: 11 }}>
              Used as the <code className="lx-mono">x-api-key</code> header for every backend call.
              Reload the page after changing.
            </small>
          </div>
        </section>

        <section className="lx-panel" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <header>
            <div className="lx-label">Appearance</div>
          </header>

          <div className="lx-field">
            <label>Theme</label>
            <SegmentedControl
              value={settings.theme}
              onChange={(v) => update("theme", v as Theme)}
              options={[
                { value: "dark",   label: "Dark" },
                { value: "dim",    label: "Dim" },
                { value: "light",  label: "Light" },
                { value: "system", label: "System" },
              ]}
            />
          </div>

          <div className="lx-field">
            <label>Density</label>
            <SegmentedControl
              value={settings.density}
              onChange={(v) => update("density", v as Density)}
              options={[
                { value: "comfortable", label: "Comfortable" },
                { value: "compact",     label: "Compact" },
                { value: "dense",       label: "Dense" },
              ]}
            />
          </div>

          <div className="lx-field">
            <label>Number Format</label>
            <SegmentedControl
              value={settings.numberFormat}
              onChange={(v) => update("numberFormat", v as NumberFormat)}
              options={[
                { value: "us",  label: "1,234.56" },
                { value: "eu",  label: "1.234,56" },
                { value: "iso", label: "1 234.56" },
              ]}
            />
          </div>
        </section>
      </div>

      <section className="lx-panel" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <header style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
          <div className="lx-label">Risk Policy</div>
          <span className="lx-mono lx-dim" style={{ fontSize: 11 }}>
            Limits enforced by the gate · changes take effect on next tick
          </span>
        </header>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(5, minmax(0, 1fr))", gap: 12 }}>
          <RiskField
            label="Max gross exposure"
            suffix="%"
            value={settings.maxGrossExposure}
            onChange={(v) => update("maxGrossExposure", v)}
          />
          <RiskField
            label="Max single position"
            suffix="%"
            value={settings.maxSinglePosition}
            onChange={(v) => update("maxSinglePosition", v)}
          />
          <RiskField
            label="Stop-loss"
            suffix="%"
            value={settings.stopLoss}
            onChange={(v) => update("stopLoss", v)}
          />
          <RiskField
            label="Gate threshold τ"
            value={settings.gateThreshold}
            onChange={(v) => update("gateThreshold", v)}
          />
          <RiskField
            label="Auto-flatten on σ"
            value={settings.autoFlattenSigma}
            onChange={(v) => update("autoFlattenSigma", v)}
          />
        </div>
      </section>

      {savedHint && (
        <div
          style={{
            position: "fixed",
            bottom: 16,
            right: 16,
            padding: "8px 14px",
            background: "var(--green-soft)",
            color: "var(--green)",
            border: "1px solid rgba(34,197,94,0.4)",
            borderRadius: 6,
            fontSize: 12,
            fontFamily: "var(--font-mono)",
            boxShadow: "0 8px 20px rgba(0,0,0,0.4)",
          }}
        >
          ✓ {savedHint}
        </div>
      )}
    </div>
  );
}

function SegmentedControl<T extends string>({
  value,
  onChange,
  options,
}: {
  value: T;
  onChange: (v: T) => void;
  options: Array<{ value: T; label: string }>;
}) {
  return (
    <div style={{ display: "inline-flex", gap: 4 }}>
      {options.map((opt) => (
        <button
          key={opt.value}
          type="button"
          className={`lx-btn ${value === opt.value ? "active" : "ghost"}`}
          onClick={() => onChange(opt.value)}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

function RiskField({
  label,
  value,
  suffix,
  onChange,
}: {
  label: string;
  value: string;
  suffix?: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="lx-field">
      <label>{label}</label>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          background: "var(--bg-input)",
          border: "1px solid var(--border)",
          borderRadius: 6,
        }}
      >
        <input
          value={value}
          onChange={(e) => onChange(e.target.value)}
          style={{
            border: 0,
            background: "transparent",
            flex: 1,
            padding: "8px 10px",
            fontFamily: "var(--font-mono)",
          }}
        />
        {suffix && (
          <span
            className="lx-mono"
            style={{ paddingRight: 10, color: "var(--text-dim)", fontSize: 11 }}
          >
            {suffix}
          </span>
        )}
      </div>
    </div>
  );
}
