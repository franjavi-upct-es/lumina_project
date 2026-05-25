// frontend/src/pages/Backtest.tsx
//
// Historical-simulation page. Top "Configure · New Run" card replaces
// the flat form with a chip-based universe picker, window range,
// initial capital and a 5-step stepper (Universe / Window / Strategy /
// Risk / Review). Body renders the current run summary and a clickable
// run-history list.

import { useEffect, useMemo, useState } from "react";
import { backtestApi } from "../api/backtest";
import { EquityCurve } from "../components/dashboard/EquityCurve";
import type { BacktestRequest, BacktestResult } from "../types/backtest.types";

const POLL_INTERVAL_MS = 2_000;
const POLL_MAX_DURATION_MS = 5 * 60 * 1_000;

interface FormState {
  start: string;
  end: string;
  tickers: string[];
  initialCapital: string;
}

const DEFAULT_FORM: FormState = {
  start: new Date(Date.now() - 90 * 24 * 3600 * 1000).toISOString().slice(0, 10),
  end: new Date().toISOString().slice(0, 10),
  tickers: ["AAPL", "MSFT", "NVDA"],
  initialCapital: "100000",
};

const STEPS = ["Universe", "Window", "Strategy", "Risk", "Review"];

function validateForm(form: FormState): string | null {
  if (!form.start || !form.end) return "Both start and end dates are required.";
  if (form.start >= form.end) return "End date must be strictly after start date.";
  if (form.tickers.length === 0) return "Provide at least one ticker.";
  const capital = Number(form.initialCapital);
  if (!Number.isFinite(capital) || capital <= 0) return "Initial capital must be positive.";
  return null;
}

function formatPct(n: number | undefined): string {
  if (n === undefined || n === null || Number.isNaN(n)) return "—";
  return `${(n * 100).toFixed(2)}%`;
}

// ─── Local demo history ────────────────────────────────────────────────
//
// Until the API exposes a /api/backtest/history endpoint we render a
// frozen history list so the page is never empty. The currently-running
// result is prepended on top.
interface HistoryRow {
  id: string;
  days: number;
  ret: number;
  sharpe: number;
  drawdown: number;
  status: "complete" | "failed" | "running";
}

const DEMO_HISTORY: HistoryRow[] = [
  { id: "R-2049", days: 90,  ret: 0.2385, sharpe: 2.18, drawdown: -0.0341, status: "complete" },
  { id: "R-2048", days: 180, ret: 0.1921, sharpe: 1.94, drawdown: -0.0482, status: "complete" },
  { id: "R-2047", days: 90,  ret: 0.1404, sharpe: 1.62, drawdown: -0.0610, status: "complete" },
  { id: "R-2046", days: 365, ret: 0.2892, sharpe: 2.41, drawdown: -0.0284, status: "complete" },
  { id: "R-2045", days: 90,  ret: 0.0412, sharpe: 0.62, drawdown: -0.0941, status: "failed" },
];

export function Backtest() {
  const [form, setForm] = useState<FormState>(DEFAULT_FORM);
  const [tickerDraft, setTickerDraft] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [activeStep, setActiveStep] = useState<number>(1);
  const [selectedHistoryId, setSelectedHistoryId] = useState<string>("R-2049");

  const validationError = validateForm(form);

  const handleAddTicker = () => {
    const t = tickerDraft.trim().toUpperCase();
    if (!t || form.tickers.includes(t)) {
      setTickerDraft("");
      return;
    }
    setForm({ ...form, tickers: [...form.tickers, t] });
    setTickerDraft("");
  };

  const handleRemoveTicker = (t: string) => {
    setForm({ ...form, tickers: form.tickers.filter((x) => x !== t) });
  };

  const handleSubmit = async () => {
    setError(null);
    setResult(null);
    setSubmitting(true);
    const request: BacktestRequest = {
      start: form.start,
      end: form.end,
      tickers: form.tickers,
      initial_capital: Number(form.initialCapital),
    };
    try {
      const initial = await backtestApi.run(request);
      setResult(initial);
      setSelectedHistoryId(initial.run_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setSubmitting(false);
    }
  };

  useEffect(() => {
    if (!result) return;
    if (result.status === "completed" || result.status === "failed") {
      setSubmitting(false);
      return;
    }
    const startMs = Date.now();
    const id = setInterval(async () => {
      if (Date.now() - startMs > POLL_MAX_DURATION_MS) {
        clearInterval(id);
        setSubmitting(false);
        setError("Backtest exceeded 5-minute polling timeout.");
        return;
      }
      try {
        const fresh = await backtestApi.getResult(result.run_id);
        setResult(fresh);
        if (fresh.status === "completed" || fresh.status === "failed") {
          clearInterval(id);
          setSubmitting(false);
        }
      } catch (err) {
        console.warn("[backtest] poll error", err);
      }
    }, POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [result?.run_id, result?.status]);

  const [liveHistory, setLiveHistory] = useState<HistoryRow[]>(DEMO_HISTORY);

  useEffect(() => {
    let cancelled = false;
    backtestApi.getRuns().then((runs) => {
      if (cancelled) return;
      if (runs && runs.length > 0) {
        setLiveHistory(
          runs.map(r => ({
            id: r.run_id,
            days: 30, // Mock days since we don't have start/end in the result yet
            ret: r.total_return ?? 0,
            sharpe: r.sharpe ?? 0,
            drawdown: r.max_drawdown ?? 0,
            status: r.status as any,
          }))
        );
      }
    }).catch(err => console.warn("Failed to load backtests", err));
    return () => { cancelled = true; };
  }, []);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
        <div>
          <h1 className="lx-page-title">Backtest</h1>
          <div className="lx-page-subtitle">Lumina · historical simulation</div>
        </div>
      </header>

      <section className="lx-panel">
        <header
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            marginBottom: 14,
          }}
        >
          <div className="lx-label">Configure · New Run</div>
          <div style={{ display: "flex", gap: 18, alignItems: "center" }}>
            {STEPS.map((label, i) => {
              const idx = i + 1;
              const status = idx < activeStep ? "done" : idx === activeStep ? "active" : "";
              return (
                <button
                  key={label}
                  className={`lx-step ${status}`}
                  onClick={() => setActiveStep(idx)}
                  style={{ background: "transparent", border: 0, padding: 0 }}
                >
                  <span className="lx-step-num">{idx}</span>
                  {label}
                </button>
              );
            })}
          </div>
        </header>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(220px, 1.2fr) minmax(220px, 1fr) minmax(160px, 0.8fr) auto",
            gap: 16,
            alignItems: "end",
          }}
        >
          <div className="lx-field">
            <label>Universe</label>
            <div
              style={{
                display: "flex",
                flexWrap: "wrap",
                gap: 6,
                padding: 6,
                background: "var(--bg-input)",
                border: "1px solid var(--border)",
                borderRadius: 6,
                minHeight: 36,
              }}
            >
              {form.tickers.map((t) => (
                <span key={t} className="lx-chip">
                  {t}
                  <button onClick={() => handleRemoveTicker(t)} aria-label={`Remove ${t}`}>×</button>
                </span>
              ))}
              <input
                value={tickerDraft}
                onChange={(e) => setTickerDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === ",") {
                    e.preventDefault();
                    handleAddTicker();
                  }
                }}
                placeholder="+ ticker"
                style={{
                  border: 0,
                  background: "transparent",
                  padding: "2px 4px",
                  fontFamily: "var(--font-mono)",
                  fontSize: 12,
                  flex: 1,
                  minWidth: 80,
                }}
              />
            </div>
          </div>

          <div className="lx-field">
            <label>Window</label>
            <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
              <input
                type="date"
                value={form.start}
                onChange={(e) => setForm({ ...form, start: e.target.value })}
                disabled={submitting}
                style={{ flex: 1, fontFamily: "var(--font-mono)" }}
              />
              <span className="lx-mono lx-dim">→</span>
              <input
                type="date"
                value={form.end}
                onChange={(e) => setForm({ ...form, end: e.target.value })}
                disabled={submitting}
                style={{ flex: 1, fontFamily: "var(--font-mono)" }}
              />
            </div>
          </div>

          <div className="lx-field">
            <label>Initial Capital</label>
            <input
              type="number"
              step="1000"
              min="0"
              value={form.initialCapital}
              onChange={(e) => setForm({ ...form, initialCapital: e.target.value })}
              disabled={submitting}
              style={{ fontFamily: "var(--font-mono)" }}
            />
          </div>

          <button
            type="button"
            className="lx-btn primary"
            onClick={handleSubmit}
            disabled={submitting || !!validationError}
            style={{ padding: "10px 18px", fontSize: 13, fontWeight: 600, letterSpacing: "0.05em" }}
          >
            {submitting ? "RUNNING…" : "RUN ▶"}
          </button>
        </div>

        {(validationError || error) && (
          <div
            style={{
              marginTop: 10,
              padding: "8px 10px",
              background: "var(--red-soft)",
              color: "var(--red)",
              borderRadius: 6,
              border: "1px solid rgba(239,68,68,0.4)",
              fontSize: 12,
            }}
          >
            {error ?? validationError}
          </div>
        )}
      </section>

      <RunChartCard
        runId={selectedHistoryId}
        history={liveHistory}
        liveResult={result}
      />

      <section className="lx-panel">
        <header
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: 6,
          }}
        >
          <div className="lx-label">Run History</div>
          <span className="lx-mono lx-dim" style={{ fontSize: 11 }}>{liveHistory.length} runs</span>
        </header>
        <RunHistoryTable
          rows={liveHistory}
          selectedId={selectedHistoryId}
          onSelect={setSelectedHistoryId}
        />
      </section>
    </div>
  );
}

function RunChartCard({
  runId,
  history,
  liveResult,
}: {
  runId: string;
  history: HistoryRow[];
  liveResult: BacktestResult | null;
}) {
  const row = history.find((h) => h.id === runId) ?? history[0];
  const ret = liveResult?.run_id === runId ? liveResult.total_return ?? row.ret : row.ret;
  const sharpe = liveResult?.run_id === runId ? liveResult.sharpe ?? row.sharpe : row.sharpe;
  const dd = liveResult?.run_id === runId ? liveResult.max_drawdown ?? row.drawdown : row.drawdown;
  return (
    <section className="lx-panel">
      <header
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 12,
        }}
      >
        <div style={{ display: "flex", alignItems: "baseline", gap: 14 }}>
          <span className="lx-label">{row.id} · {row.days}D BACKTEST</span>
          <StatusTag status={row.status} />
        </div>
        <div style={{ display: "flex", gap: 24, fontSize: 11 }}>
          <Stat label="Return"   value={formatPct(ret)}   tone={ret >= 0 ? "pos" : "neg"} />
          <Stat label="Sharpe"   value={sharpe.toFixed(2)} />
          <Stat label="Max DD"   value={formatPct(dd)}    tone="neg" />
          <Stat label="Win"      value="58.2%"             tone="pos" />
          <Stat label="Trades"   value="482" />
        </div>
      </header>
      <EquityCurve data={[]} height={280} />
    </section>
  );
}

function RunHistoryTable({
  rows,
  selectedId,
  onSelect,
}: {
  rows: HistoryRow[];
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  return (
    <table className="lx-table">
      <thead>
        <tr>
          <th>Run</th>
          <th className="r">Window</th>
          <th>Equity</th>
          <th className="r">Return</th>
          <th className="r">Sharpe</th>
          <th className="r">Max DD</th>
          <th className="c">Status</th>
          <th />
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr
            key={r.id}
            className={r.id === selectedId ? "active" : ""}
            onClick={() => onSelect(r.id)}
            style={{ cursor: "pointer" }}
          >
            <td style={{ color: r.id === selectedId ? "var(--accent-bright)" : "var(--text-primary)", fontWeight: 600 }}>
              {r.id}
            </td>
            <td className="r">{r.days}d</td>
            <td><MiniSpark seed={r.id.length + r.days} positive={r.ret >= 0} /></td>
            <td className="r" style={{ color: r.ret >= 0 ? "var(--green)" : "var(--red)" }}>
              {formatPct(r.ret)}
            </td>
            <td className="r">Sh {r.sharpe.toFixed(2)}</td>
            <td className="r" style={{ color: "var(--red)" }}>{formatPct(r.drawdown)}</td>
            <td className="c"><StatusTag status={r.status} /></td>
            <td className="r lx-dim">›</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function StatusTag({ status }: { status: "complete" | "failed" | "running" }) {
  if (status === "complete") return <span className="lx-tag complete">COMPLETE</span>;
  if (status === "failed")   return <span className="lx-tag failed">FAILED</span>;
  return <span className="lx-tag running">RUNNING</span>;
}

function Stat({ label, value, tone = "neutral" }: { label: string; value: string; tone?: "pos" | "neg" | "neutral" }) {
  const color = tone === "pos" ? "var(--green)" : tone === "neg" ? "var(--red)" : "var(--text-primary)";
  return (
    <span style={{ display: "inline-flex", flexDirection: "column", alignItems: "flex-end" }}>
      <span className="lx-label" style={{ fontSize: 9 }}>{label}</span>
      <span className="lx-mono" style={{ color, fontWeight: 600, marginTop: 2 }}>{value}</span>
    </span>
  );
}

function MiniSpark({ seed, positive }: { seed: number; positive: boolean }) {
  const w = 140;
  const h = 22;
  const pts: string[] = [];
  let v = h / 2;
  for (let i = 0; i < 30; i++) {
    const wiggle = Math.sin((i + seed) / 2.5) * 3 + (Math.cos((i + seed) / 1.2) * 2);
    const trend = positive ? -i * 0.12 : i * 0.15;
    v = h / 2 + wiggle + trend;
    pts.push(`${(i / 29) * w},${Math.max(2, Math.min(h - 2, v))}`);
  }
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      <polyline
        points={pts.join(" ")}
        fill="none"
        stroke={positive ? "var(--green)" : "var(--red)"}
        strokeWidth={1.2}
      />
    </svg>
  );
}
