// frontend/react-app/src/App.jsx
// Root application component.
// Configures React Query, React Router, and wraps the Layout.

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Layout } from "./components/layout/Layout";
import HomePage from "./pages/HomePage";
import DataExplorer from "./pages/DataExplorer";
import StrategyLab from "./pages/StrategyLab";
import ModelComparator from "./pages/ModelComparator";
import RiskDashboard from "./pages/RiskDashboard";
import SentimentMonitor from "./pages/SentimentMonitor";
import AgentMonitor from "./pages/AgentMonitor";
import EmbeddingVisualizer from "./pages/EmbeddingVisualizer";

// Global React Query client with conservative defaults for a trading dashboard
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Treat data as fresh for 60 seconds before background refetching
      staleTime: 60_000,
      // Keep data in cache for 5 minutes after the component unmounts
      gcTime: 300_000,
      // Retry failed requests twice before showing an error
      retry: 2,
      retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 10_000),
    },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          {/* All pages share the same Layout wrapper (sidebar + header) */}
          <Route path="/" element={<Layout />}>
            <Route index element={<HomePage />} />
            <Route path="data-explorer" element={<DataExplorer />} />
            <Route path="strategy-lab" element={<StrategyLab />} />
            <Route path="model-comparator" element={<ModelComparator />} />
            <Route path="risk-dashboard" element={<RiskDashboard />} />
            <Route path="sentiment-monitor" element={<SentimentMonitor />} />
            {/* V3 NEW pages */}
            <Route path="agent-monitor" element={<AgentMonitor />} />
            <Route
              path="embedding-visualizer"
              element={<EmbeddingVisualizer />}
            />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
