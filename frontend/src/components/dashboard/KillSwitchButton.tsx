// frontend/src/components/dashboard/KillSwitchButton.tsx
import { useState } from "react";
import { useRiskGate } from "../../hooks/useRiskGate";

export function KillSwitchButton() {
  const { state, setState } = useRiskGate();
  const [confirmStep, setConfirmStep] = useState(0);
  const [typed, setTyped] = useState("");

  const handleClick = () => {
    if (confirmStep === 0) setConfirmStep(1);
    else if (confirmStep === 1 && typed === "LIQUIDATE") {
      setState("LIQUIDATE_ALL", "Manual kill switch");
      setConfirmStep(0); setTyped("");
    }
  };

  return (
    <div style={{ border: "2px solid red", padding: 16 }}>
      <h3>Kill Switch — State: {state}</h3>
      {confirmStep === 0 && (
        <button onClick={handleClick} style={{ background: "red", color: "white" }}>
          ARM Kill Switch
        </button>
      )}
      {confirmStep === 1 && (
        <>
          <p>Type LIQUIDATE to confirm:</p>
          <input value={typed} onChange={(e) => setTyped(e.target.value)} />
          <button onClick={handleClick} disabled={typed !== "LIQUIDATE"}>
            CONFIRM LIQUIDATE
          </button>
          <button onClick={() => { setConfirmStep(0); setTyped(""); }}>Cancel</button>
        </>
      )}
    </div>
  );
}
