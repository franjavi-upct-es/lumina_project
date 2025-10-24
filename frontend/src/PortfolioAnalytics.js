import React, { useState, useEffect } from "react";
import "./PortfolioAnalytics.css";

const currencyFormatter = new Intl.NumberFormat("es-ES", {
    style: "currency",
    currency: "EUR",
});

const PortfolioAnalytics = () => {
    const [analytics, setAnalytics] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchAnalytics = async () => {
            try {
                const response = await fetch("http://127.0.0.1:5000/api/portfolio/analytics");
                if (!response.ok) {
                    throw new Error("No se pudo cargar el análisis del portafolio.");
                }
                const data = await response.json();
                setAnalytics(data);
            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        fetchAnalytics();
        // Actualizar cada 30 segundos
        const interval = setInterval(fetchAnalytics, 30000);
        return () => clearInterval(interval);
    }, []);

    if (loading) return <div className="portfolio-analytics">Cargando análisis...</div>;
    if (error) return <div className="portfolio-analytics error">Error: {error}</div>;
    if (!analytics) return null;

    const getSharpeColor = (sharpe) => {
        if (sharpe > 1) return "good";
        if (sharpe > 0) return "neutral";
        return "poor";
    };

    const getDiversificationColor = (score) => {
        if (score > 70) return "good";
        if (score > 40) return "neutral";
        return "poor";
    };

    return (
        <div className="portfolio-analytics">
            <h4>Análisis de Cartera</h4>

            <div className="analytics-summary">
                <div className="analytics-item">
                    <span className="label">Valor Total:</span>
                    <span className="value">{currencyFormatter.format(analytics.total_value)}</span>
                </div>

                <div className="analytics-item">
                    <span className="label">Invertido:</span>
                    <span className="value">{currencyFormatter.format(analytics.invested_value)}</span>
                </div>

                <div className="analytics-item">
                    <span className="label">Efectivo:</span>
                    <span className="value">{analytics.cash_percent.toFixed(1)}%</span>
                </div>
            </div>

            <div className="analytics-metrics">
                <div className={`metric-box ${getSharpeColor(analytics.sharpe_ratio)}`}>
                    <strong>Ratio de Sharpe</strong>
                    <span className="metric-value">{analytics.sharpe_ratio.toFixed(2)}</span>
                    <p className="metric-desc">
                        {analytics.sharpe_ratio > 1 ? "Excelente retorno ajustado al riesgo" :
                            analytics.sharpe_ratio > 0 ? "Retorno aceptable" :
                                "Retorno por debajo del riesgo asumido"}
                    </p>
                </div>

                <div className={`metric-box ${getDiversificationColor(analytics.diversification_score)}`}>
                    <strong>Diversificación</strong>
                    <span className="metric-value">{analytics.diversification_score.toFixed(0)}%</span>
                    <p className="metric-desc">
                        {analytics.positions_count} posiciones
                        {analytics.diversification_score > 70 ? " - Bien diversificado" :
                            analytics.diversification_score > 40 ? " - Diversificación moderada" :
                                " - Poco diversificado"}
                    </p>
                </div>
            </div>

            {Object.keys(analytics.returns).length > 0 && (
                <div className="returns-section">
                    <strong>Retornos Anuales:</strong>
                    <ul className="returns-list">
                        {Object.entries(analytics.returns).map(([ticker, returnPct]) => (
                            <li key={ticker} className={returnPct >= 0 ? "positive" : "negative"}>
                                <span>{ticker}</span>
                                <span>{returnPct >= 0 ? "+" : ""}{returnPct.toFixed(2)}%</span>
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
};

export default PortfolioAnalytics;
