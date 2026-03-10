// frontend/react-app/tailwind.config.js
// Tailwind CSS configuration for Lumina V3 dashboard.
// Extends the default palette with a custom dark trading theme aligned
// with the color constants from the Streamlit components (charts.py COLORS).

/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Primary brand palette (mirrors Streamlit COLORS constant)
        brand: {
          primary: "#1f77b4", // Blue  – main accent
          secondary: "#ff7f0e", // Orange – secondary accent
          success: "#2ca02c", // Green  – positive / profit
          danger: "#d62728", // Red    – negative / loss / alert
          warning: "#ffbb00", // Yellow – caution / VaR thresholds
          info: "#17a2b8", // Teal   – informational
          purple: "#9467bd", // Purple – structural encoder
          pink: "#e377c2", // Pink   – semantic encoder
          gray: "#7f7f7f", // Gray   – neutral / zero lines
          cyan: "#17becf", // Cyan   – temporal encoder
        },
        // Dark trading dashboard background palette
        surface: {
          900: "#0d0d0f", // Deepest background (main body)
          800: "#141418", // Sidebar background
          700: "#1a1a20", // Card background
          600: "#22222a", // Input / panel background
          500: "#2a2a35", // Border / divider
          400: "#383845", // Hover state
          300: "#505060", // Muted text / icons
        },
        // Safety system status colors (V3 Safety Arbitrator)
        safety: {
          active: "#2ca02c", // System fully operational
          guarded: "#ffbb00", // Elevated uncertainty – caution
          critical: "#d62728", // Circuit breaker triggered
          offline: "#7f7f7f", // Agent not running
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
      },
      // Custom animations for live data pulsing
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "fade-in": "fadeIn 0.3s ease-in-out",
        "slide-in": "slideIn 0.25s ease-out",
      },
      keyframes: {
        fadeIn: { "0%": { opacity: "0" }, "100%": { opacity: "1" } },
        slideIn: {
          "0%": { transform: "translateX(-10px)", opacity: "0" },
          "100%": { transform: "translateX(0)", opacity: "1" },
        },
      },
    },
  },
  plugins: [],
};
