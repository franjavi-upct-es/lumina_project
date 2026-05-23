// frontend/src/pages/Backtest.tsx
//
// Backtest form + result viewer.
//
// Flow
// ----
// 1. The operator fills the form (date range, tickers, initial capital).
// 2. On submit we POST /api/backtest/run; the response is a run_id and
//    a pending status. We disable the form and remember the run_id.
// 3. We then poll GET /api/backtest/results/{run_id} every 2 s until
//    status leaves the "pending" / "running" set. Polling is bounded
//    to 5 minutes; longer-running backtests should switch to the
//    history list (not yet implemented).
// 4. Final result is rendered with Sharpe, max drawdown and total
//    return; the operator can submit another run.
//
// Why ticker entry as comma-separated string?
// -------------------------------------------
// A proper multi-select on top of the universe constants would mean
// either fetching the universe from a new endpoint or duplicating the
// TARGET_TICKERS list in TypeScript. The comma-separated input keeps
// the contract simple and the universe authoritative in Python.

import { useEffect, useState } from "react";
import { backtestApi } from "../api/backtest";
import type { BacktestRequest, BacktestResult } from "../types/backtest.types";

const POLL_INTERVAL_MS = 2_000;
const POLL_MAX_DURATION_MS = 5 * 60 * 1_000;

interface FormState {
  start: string;
  end: string;
  tickersRaw: string;
  initialCapital: string;
}

const DEFAULT_FORM: FormState = {
  start: new Date(Date.now() - 90 * 24 * 3600 * 1000).toISOString().slice(0, 10),
  end: new Date().toISOString().slice(0, 10),
  tickersRaw: "AAPL, MSFT, NVDA",
  initialCapital: "100000",
};

function parseTickers(raw: string): string[] {
  return raw
    .split(",")
    .map((s) => s.trim().toUpperCase())
    .filter((s) => s.length > 0);
}

function validateForm(form: FormState): string | null {
  if (!form.start || !form.end) return "Both start and end dates are required.";
  if (form.start >= form.end) return "End date must be strictly after start date.";
  const tickers = parseTickers(form.tickersRaw);
  if (tickers.length === 0) return "Provide at least one ticker.";
  const capital = Number(form.initialCapital);
  if (!Number.isFinite(capital) || capital <= 0) return "Initial capital must be a positive number.";
  return null;
}

function formatNumber(n: number | undefined, digits = 3): string {
  if (n === undefined || n === null || Number.isNaN(n)) return "—";
  return n.toFixed(digits);
}

function formatPct(n: number | undefined): string {
  if (n === undefined || n === null || Number.isNaN(n)) return "—";
  return `${(n * 100).toFixed(2)}%`;
}

export function Backtest() {
  const [form, setForm] = useState<FormState>(DEFAULT_FORM);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<BacktestResult | null>(null);

  const validationError = validateForm(form);

  const handleSubmit = async () => {
    setError(null);
    setResult(null);
    setSubmitting(true);
    const request: BacktestRequest = {
      start: form.start,
      end: form.end,
      tickers: parseTickers(form.tickersRaw),
      initial_capital: Number(form.initialCapital),
    };
    try {
      const initial = await backtestApi.run(request);
      setResult(initial);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setSubmitting(false);
    }
  };

  // Polling effect: runs while we have a non-terminal result. The
  // bound on duration is defensive — if the backend never closes the
  // run, the dashboard should not poll forever.
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
        setError("Backtest exceeded 5-minute polling timeout; check Backtest history later.");
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
        // Transient errors are tolerated — the next tick will retry.
        console.warn("[backtest] poll error", err);
      }
    }, POLL_INTERVAL_MS);

    return () => clearInterval(id);
  }, [result?.run_id, result?.status]);

  return (
    <div style={{ padding: 16, maxWidth: 720, margin: "0 auto" }}>
      <h1 style={{ marginTop: 0 }}>Backtest</h1>

      <section style={panelStyle}>
        <h2 style={{ marginTop: 0, fontSize: 16 }}>New run</h2>
        <div style={{ display: "grid", gridTemplateColumns: "120px 1fr", gap: 12, alignItems: "center" }}>
          <label htmlFor="bt-start">Start date</label>
          <input
            id="bt-start"
            type="date"
            value={form.start}
            onChange={(e) => setForm({ ...form, start: e.target.value })}
            disabled={submitting}
          />
          <label htmlFor="bt-end">End date</label>
          <input
            id="bt-end"
            type="date"
            value={form.end}
            onChange={(e) => setForm({ ...form, end: e.target.value })}
            disabled={submitting}
          />
          <label htmlFor="bt-tickers">Tickers</label>
          <input
            id="bt-tickers"
            type="text"
            value={form.tickersRaw}
            onChange={(e) => setForm({ ...form, tickersRaw: e.target.value })}
            placeholder="AAPL, MSFT, NVDA"
            disabled={submitting}
          />
          <label htmlFor="bt-capital">Initial capital</label>
          <input
            id="bt-capital"
            type="number"
            step="1000"
            min="0"
            value={form.initialCapital}
            onChange={(e) => setForm({ ...form, initialCapital: e.target.value })}
            disabled={submitting}
          />
        </div>

        <div style={{ marginTop: 16, display: "flex", gap: 12, alignItems: "center" }}>
          <button
            type="button"
            onClick={handleSubmit}
            disabled={submitting || !!validationError}
            style={{
              padding: "8px 16px",
              background: submitting || validationError ? "#8c959f" : "#0969da",
              color: "#ffffff",
              border: "none",
              borderRadius: 4,
              cursor: submitting || validationError ? "not-allowed" : "pointer",
            }}
          >
            {submitting ? "Running…" : "Run backtest"}
          </button>
          {validationError && <small style={{ color: "#cf222e" }}>{validationError}</small>}
        </div>

        {error && (
          <div style={{ marginTop: 12, padding: 8, background: "#ffeef0", color: "#cf222e", borderRadius: 4 }}>
            {error}
          </div>
        )}
      </section>

      {result && (
        <section style={panelStyle}>
          <h2 style={{ marginTop: 0, fontSize: 16 }}>Run {result.run_id}</h2>
          <p style={{ margin: "4px 0", fontSize: 13 }}>
            <strong>Status:</strong>{" "}
            <span style={{ color: statusColor(result.status), fontWeight: 600 }}>{result.status}</span>
          </p>
          {result.status === "completed" && (
            <table style={{ width: "100%", marginTop: 12, fontSize: 13 }}>
              <tbody>
                <tr>
                  <td style={{ color: "#586069", padding: "4px 0" }}>Sharpe ratio</td>
                  <td style={{ textAlign: "right", fontFamily: "monospace", fontWeight: 600 }}>
                    {formatNumber(result.sharpe)}
                  </td>
                </tr>
                <tr>
                  <td style={{ color: "#586069", padding: "4px 0" }}>Max drawdown</td>
                  <td style={{ textAlign: "right", fontFamily: "monospace", fontWeight: 600 }}>
                    {formatPct(result.max_drawdown)}
                  </td>
                </tr>
                <tr>
                  <td style={{ color: "#586069", padding: "4px 0" }}>Total return</td>
                  <td style={{ textAlign: "right", fontFamily: "monospace", fontWeight: 600 }}>
                    {formatPct(result.total_return)}
                  </td>
                </tr>
              </tbody>
            </table>
          )}
        </section>
      )}
    </div>
  );
}

function statusColor(status: BacktestResult["status"]): string {
  switch (status) {
    case "completed": return "#1a7f37";
    case "failed":    return "#cf222e";
    default:          return "#9a6700";
  }
}

const panelStyle: React.CSSProperties = {
  border: "1px solid #d0d7de",
  borderRadius: 6,
  padding: 16,
  marginBottom: 16,
  background: "#ffffff",
};
