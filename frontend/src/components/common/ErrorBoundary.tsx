// frontend/src/components/common/ErrorBoundary.tsx
import { Component, type ReactNode } from "react";

interface State { err: Error | null; }

export class ErrorBoundary extends Component<{ children: ReactNode }, State> {
  state = { err: null as Error | null };
  static getDerivedStateFromError(err: Error) { return { err }; }
  componentDidCatch(err: Error) { console.error("ErrorBoundary:", err); }
  render() {
    if (this.state.err) {
      return (
        <div style={{ padding: 24 }}>
          <div className="lx-panel" style={{ borderColor: "var(--red)" }}>
            <h2 style={{ marginTop: 0, color: "var(--red)" }}>Something went wrong</h2>
            <p style={{ color: "var(--text-secondary)" }}>{this.state.err.message}</p>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
