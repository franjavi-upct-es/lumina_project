// frontend/src/components/common/ErrorBoundary.tsx
import { Component, type ReactNode } from "react";

interface State { err: Error | null; }

export class ErrorBoundary extends Component<{ children: ReactNode }, State> {
  state = { err: null as Error | null };
  static getDerivedStateFromError(err: Error) { return { err }; }
  componentDidCatch(err: Error) { console.error("ErrorBoundary:", err); }
  render() {
    if (this.state.err) return <div>Something went wrong: {this.state.err.message}</div>;
    return this.props.children;
  }
}
