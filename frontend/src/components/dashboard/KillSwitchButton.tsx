// frontend/src/components/dashboard/KillSwitchButton.tsx
//
// Manual liquidation control. Two-step confirmation: ARM, then type
// LIQUIDATE in a confirmation box. Styled for the dark operator console.

import { useState } from "react";
import { useRiskGate } from "../../hooks/useRiskGate";

export function KillSwitchButton() {
  const { state, setState } = useRiskGate();
  const [confirmStep, setConfirmStep] = useState(0);
  const [typed, setTyped] = useState("");

  const handleArm = () => setConfirmStep(1);
  const handleConfirm = () => {
    if (typed === "LIQUIDATE") {
      setState("LIQUIDATE_ALL", "Manual kill switch");
      setConfirmStep(0);
      setTyped("");
    }
  };
  const handleCancel = () => {
    setConfirmStep(0);
    setTyped("");
  };

  return (
    <section
      className="lx-panel"
      style={{
        borderColor: "rgba(239, 68, 68, 0.35)",
        background: "linear-gradient(180deg, rgba(239,68,68,0.06), var(--bg-panel))",
      }}
    >
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <div className="lx-label" style={{ color: "var(--red)" }}>Kill Switch</div>
          <div style={{ fontSize: 12, color: "var(--text-secondary)", marginTop: 2 }}>
            Emergency portfolio liquidation
          </div>
        </div>
        <span
          className="lx-pill"
          style={{
            color: state === "NORMAL" ? "var(--green)" : "var(--red)",
            borderColor: state === "NORMAL" ? "rgba(34,197,94,0.4)" : "rgba(239,68,68,0.5)",
          }}
        >
          <span className="lx-dot" style={{ background: state === "NORMAL" ? "var(--green)" : "var(--red)" }} />
          {state}
        </span>
      </header>

      <div style={{ marginTop: 14 }}>
        {confirmStep === 0 && (
          <button className="lx-btn danger" type="button" onClick={handleArm}>
            ⏻ ARM Kill Switch
          </button>
        )}
        {confirmStep === 1 && (
          <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 8 }}>
            <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>
              Type <code className="lx-mono" style={{ color: "var(--red)" }}>LIQUIDATE</code> to confirm:
            </span>
            <input
              value={typed}
              onChange={(e) => setTyped(e.target.value)}
              autoFocus
              style={{ width: 180, fontFamily: "var(--font-mono)" }}
            />
            <button className="lx-btn danger" disabled={typed !== "LIQUIDATE"} onClick={handleConfirm}>
              CONFIRM
            </button>
            <button className="lx-btn ghost" onClick={handleCancel}>Cancel</button>
          </div>
        )}
      </div>
    </section>
  );
}
