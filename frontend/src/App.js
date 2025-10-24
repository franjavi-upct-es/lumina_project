import React, { useState, useEffect } from 'react';
import './App.css';
import StockChart from './StockChart.js';
import Portafolio from './Portafolio.js';
import PortfolioAnalytics from './PortfolioAnalytics.js';
import PanelOperar from './PanelOperar.js';
import LSTMPredictor from './LSTMPredictor.js';
import NewsPanel from './NewsPanel.js';

// --- COMPONENTE DE TOOLTIP ÉTICO ---
const EthicalTooltip = ({ text }) => {
    const [visible, setVisible] = useState(false);
    return (
        <span className="tooltip-container">
            <span className="tooltip-trigger" onMouseEnter={() => setVisible(true)}
                onMouseLeave={() => setVisible(false)}>(i)</span>
            {visible && <div className="tooltip-content">{text}</div>}
        </span>
    );
};

// --- COMPONENTE DE SEÑAL TÉCNICA (SMA) ---
const SignalIndicator = ({ event, state }) => {
    let signalClass = 'signal-box';
    let signalText = 'Evaluando...';
    let explanation = '';

    if (state === 'GOLDEN') {
        signalClass += ' signal-golden';
        signalText = 'Estado: Alcista (Cruce Dorado)';
        explanation = 'La media móvil rápida (50 días) está POR ENCIMA de la lenta (200 días). Históricamente, esto se considera una señal alcista a largo plazo.';
    } else if (state === 'DEATH') {
        signalClass += ' signal-death';
        signalText = 'Estado: Bajista (Cruce de la Muerte)';
        explanation = 'La media móvil rápida (50 días) está POR DEBAJO de la lenta (200 días). Históricamente, esto se considera una señal bajista a largo plazo.';
    }

    if (event === 'GOLDEN_CROSS') {
        signalClass += ' signal-alert';
        signalText = '¡ALERTA: Cruce Dorado Detectado!';
        explanation = '¡Hoy, la media de 50 días ha cruzado POR ENCIMA de la de 200 días! Esta es una señal técnica alcista significativa.';
    } else if (event === 'DEATH_CROSS') {
        signalClass += ' signal-alert';
        signalText = '¡ALERTA: Cruce de la Muerte Detectado!';
        explanation = '¡Hoy, la media de 50 días ha cruzado POR DEBAJO de la de 200 días! Esta es una señal técnica bajista significativa.';
    }

    return (
        <div className={signalClass}>
            <strong>Tendencia (SMA)</strong>
            <p>{signalText}</p>
            <EthicalTooltip
                text={`${explanation} Recuerda: Esto es solo un indicador técnico basado en datos pasados y no garantiza resultados futuros.`}
            />
        </div>
    );
};

// --- COMPONENTE DE SENTIMIENTO (NLP) ---
const SentimentIndicator = ({ score, count }) => {
    let sentimentClass = 'signal-box';
    let sentimentEmoji = '😐';
    let sentimentText = 'Neutral';
    let explanation = `Análisis de ${count} titulares de noticias recientes. Un puntaje cercano a 0 indica un tono neutral.`;

    if (score > 0.05) {
        sentimentClass += ' signal-positive';
        sentimentEmoji = '😊';
        sentimentText = 'Positivo';
        explanation = `Análisis de ${count} titulares. El tono general de las noticias recientes es positivo (Puntaje: ${score.toFixed(2)}).`;
    } else if (score < -0.05) {
        sentimentClass += ' signal-negative';
        sentimentEmoji = '😟';
        sentimentText = 'Negativo';
        explanation = `Análisis de ${count} titulares. El tono general de las noticias recientes es negativo (Puntaje: ${score.toFixed(2)}).`;
    }

    if (count === 0) {
        sentimentText = 'Sin noticias';
        explanation = 'No se encontraron noticias recientes para analizar.';
    }

    return (
        <div className={sentimentClass}>
            <strong>Sentimiento (NLP)</strong>
            <p>{sentimentEmoji} {sentimentText}</p>
            <EthicalTooltip
                text={`${explanation} Esto mide el *tono* de los titulares, no su *veracidad* ni el impacto real en el mercado. Es solo un indicador del "ruido" mediático.`}
            />
        </div>
    );
};

const RsiIndicator = ({ rsi }) => {
    let rsiClass = 'signal-box';
    let rsiText = 'Neutral';
    let explanation = `El RSI es un indicador de impulso que mide la velocidad y el cambio de los movimientos de precios. Un valor entre 30 y 70 se considera neutral.`;

    if (rsi > 70) {
        rsiClass += ' signal-overbought'; // Sobrecompra
        rsiText = 'Sobrecompra';
        explanation = `Un RSI superior a 70 sugiere que la acción puede estar sobrevalorada o "sobrecomprada" y podría estar preparada para una corrección a la baja.`;
    } else if (rsi < 30) {
        rsiClass += ' signal-oversold'; // Sobreventa
        rsiText = 'Sobreventa';
        explanation = `Un RSI inferior a 30 sugiere que la acción puede estar infravalorada o "sobrevendida" y podría estar preparada para un rebote al alza.`;
    }

    return (
        <div className={rsiClass}>
            <strong>Impulso (RSI)</strong>
            <p>{rsi.toFixed(2)} - {rsiText}</p>
            <EthicalTooltip
                text={`${explanation} El RSI puede permanecer en niveles de sobrecompra o sobreventa durante largos períodos en tendencias fuertes. No es una señal de venta/compra por sí sola.`}
            />
        </div>
    );
};

const MacdIndicator = ({ macdData }) => {
    if (!macdData) {
        return null;
    }
    const { histogram, macd, signal } = macdData;
    let macdClass = 'signal-box';
    let macdText = 'Neutral';
    let explanation = `El histograma (diferencia entre MACD y Señal) es casi cero, indicando poco impulso.`;

    if (histogram > 0) {
        macdClass += ' signal-positive';
        macdText = 'Impulso Alcista';
        explanation = 'La línea MACD está por encima de su línea de Señal, y el histograma es positivo. Esto sugiere que el impulso alcista está aumentando.';
    } else if (histogram < 0) {
        macdClass += ' signal-negative';
        macdText = 'Impulso Bajista';
        explanation = 'La línea MACD está por debajo de su línea de Señal, y el histograma es negativo. Esto sugiere que el impulso bajista está aumentando.';
    }

    return (
        <div className={macdClass}>
            <strong>Impulso (MACD)</strong>
            <p>{macdText}</p>
            <EthicalTooltip
                text={`${explanation} El MACD es útil para confirmar la fuerza de una tendencia, pero puede generar señales falsas en mercados laterales.`}
            />
        </div>
    );
};

const BollingerBandsIndicator = ({ bbData, currentPrice }) => {
    if (!bbData || !currentPrice) {
        return null;
    }
    const { upper, middle, lower, percent } = bbData;
    let bbClass = 'signal-box';
    let bbText = 'En el medio';
    let explanation = `El precio está cerca de la banda media, indicando volatilidad normal.`;

    if (currentPrice > upper) {
        bbClass += ' signal-overbought';
        bbText = 'Por encima del límite superior';
        explanation = `El precio está por encima de la banda superior, lo que puede indicar sobrecompra y una posible corrección a la baja.`;
    } else if (currentPrice < lower) {
        bbClass += ' signal-oversold';
        bbText = 'Por debajo del límite inferior';
        explanation = `El precio está por debajo de la banda inferior, lo que puede indicar sobreventa y una posible rebote al alza.`;
    } else if (percent > 0.7) {
        bbClass += ' signal-positive';
        bbText = 'Cerca del límite superior';
        explanation = `El precio está en la parte superior del rango de las bandas (${(percent * 100).toFixed(1)}%), mostrando fuerza alcista.`;
    } else if (percent < 0.3) {
        bbClass += ' signal-negative';
        bbText = 'Cerca del límite inferior';
        explanation = `El precio está en la parte inferior del rango de las bandas (${(percent * 100).toFixed(1)}%), mostrando debilidad.`;
    }

    return (
        <div className={bbClass}>
            <strong>Volatilidad (Bollinger)</strong>
            <p>{bbText}</p>
            <EthicalTooltip
                text={`${explanation} Las Bandas de Bollinger miden la volatilidad del mercado. Cuando el precio toca las bandas, puede indicar condiciones extremas.`}
            />
        </div>
    );
};

const EmaIndicator = ({ emaData }) => {
    if (!emaData) {
        return null;
    }
    const { short: emaShort, long: emaLong } = emaData;
    let emaClass = 'signal-box';
    let emaText = 'Neutral';
    let explanation = `Las EMAs están muy cerca, sin tendencia clara.`;

    const difference = emaShort - emaLong;
    const percentDiff = (difference / emaLong) * 100;

    if (percentDiff > 1) {
        emaClass += ' signal-positive';
        emaText = 'Tendencia Alcista';
        explanation = `La EMA rápida (12) está ${percentDiff.toFixed(2)}% por encima de la EMA lenta (26), indicando una tendencia alcista fuerte.`;
    } else if (percentDiff < -1) {
        emaClass += ' signal-negative';
        emaText = 'Tendencia Bajista';
        explanation = `La EMA rápida (12) está ${Math.abs(percentDiff).toFixed(2)}% por debajo de la EMA lenta (26), indicando una tendencia bajista fuerte.`;
    }

    return (
        <div className={emaClass}>
            <strong>Tendencia (EMA)</strong>
            <p>{emaText}</p>
            <EthicalTooltip
                text={`${explanation} Las EMAs dan más peso a los precios recientes. El cruce entre la EMA rápida y lenta puede señalar cambios de tendencia.`}
            />
        </div>
    );
};

const StochasticIndicator = ({ stochData }) => {
    if (!stochData) {
        return null;
    }
    const { k, d } = stochData;
    let stochClass = 'signal-box';
    let stochText = 'Neutral';
    let explanation = `El oscilador está en zona neutral (${k.toFixed(1)}).`;

    if (k > 80) {
        stochClass += ' signal-overbought';
        stochText = 'Sobrecompra';
        explanation = `El estocástico está en ${k.toFixed(1)}, por encima de 80, indicando condiciones de sobrecompra.`;
    } else if (k < 20) {
        stochClass += ' signal-oversold';
        stochText = 'Sobreventa';
        explanation = `El estocástico está en ${k.toFixed(1)}, por debajo de 20, indicando condiciones de sobreventa.`;
    } else if (k > d) {
        stochClass += ' signal-positive';
        stochText = 'Señal Alcista';
        explanation = `La línea %K (${k.toFixed(1)}) está por encima de %D (${d.toFixed(1)}), sugiriendo impulso alcista.`;
    } else if (k < d) {
        stochClass += ' signal-negative';
        stochText = 'Señal Bajista';
        explanation = `La línea %K (${k.toFixed(1)}) está por debajo de %D (${d.toFixed(1)}), sugiriendo impulso bajista.`;
    }

    return (
        <div className={stochClass}>
            <strong>Momentum (Estocástico)</strong>
            <p>{stochText}</p>
            <EthicalTooltip
                text={`${explanation} El estocástico compara el precio de cierre actual con su rango de precios durante un período, ideal para identificar puntos de giro.`}
            />
        </div>
    );
};

function App() {
    // --- ESTADOS DE DATOS ---
    const [companyName, setCompanyName] = useState(null);
    const [stockData, setStockData] = useState([]);
    const [ticker, setTicker] = useState('AAPL');
    const [inputValue, setInputValue] = useState('AAPL');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [signalEvent, setSignalEvent] = useState('NONE');
    const [currentState, setCurrentState] = useState('HOLD');
    const [sentimentScore, setSentimentScore] = useState(0);
    const [sentimentNewsCount, setSentimentNewsCount] = useState(0);
    const [latestRsi, setLatestRsi] = useState(50); // Empezamos en 50 (neutral)
    const [latestMacd, setLatestMacd] = useState(null);
    const [latestBb, setLatestBb] = useState(null);
    const [latestEma, setLatestEma] = useState(null);
    const [latestStoch, setLatestStoch] = useState(null);
    const [darkMode, setDarkMode] = useState(() => {
        // Cargar preferencia de localStorage
        const saved = localStorage.getItem('darkMode');
        return saved ? JSON.parse(saved) : false;
    });

    // --- ESTADO DEL PORTAFOLIO ---
    const [portafolio, setPortafolio] = useState({
        efectivo: 0, // Se cargará desde la API
        posiciones: {},
    });

    // Efecto para aplicar el modo oscuro
    useEffect(() => {
        if (darkMode) {
            document.body.classList.add('dark-mode');
        } else {
            document.body.classList.remove('dark-mode');
        }
        localStorage.setItem('darkMode', JSON.stringify(darkMode));
    }, [darkMode]);

    const toggleDarkMode = () => {
        setDarkMode(!darkMode);
    };

    // Este useEffect se ejecuta cuando el ticker CAMBIA
    useEffect(() => {
        const fetchStockData = async () => {
            setLoading(true);
            setError(null);
            setCompanyName(null);
            setStockData([]);
            setSignalEvent('NONE');
            setCurrentState('HOLD');
            setSentimentScore(0);
            setSentimentNewsCount(0);
            setLatestRsi(50);
            setLatestMacd(null);
            setLatestBb(null);
            setLatestEma(null);
            setLatestStoch(null);

            try {
                const response = await fetch(`http://127.0.0.1:5000/api/stock/${ticker}`);
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || `Error: El ticker "${ticker}" no fue encontrado.`);
                }
                const data = await response.json();
                setCompanyName(data.companyName);
                setStockData(data.history);
                setSignalEvent(data.signal_event);
                setCurrentState(data.current_state);
                setSentimentScore(data.sentiment_score);
                setSentimentNewsCount(data.sentiment_news_count);
                setLatestRsi(data.latest_rsi);
                setLatestMacd(data.latest_macd);
                setLatestBb(data.latest_bb);
                setLatestEma(data.latest_ema);
                setLatestStoch(data.latest_stoch);
            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false); // Paramos de cargar SOLO los datos de la acción
            }
        };
        if (ticker) {
            fetchStockData();
        }
    }, [ticker]);

    useEffect(() => {
        const fetchPortfolioData = async () => {
            try {
                const response = await fetch(`http://127.0.0.1:5000/api/portfolio`);
                if (!response.ok) {
                    throw new Error("No se pudo cargar el portafolio.");
                }
                const data = await response.json();
                setPortafolio(data); // Establecemos el portafolio desde la DB
            } catch (err) {
                setError(err.message); // Mostramos error si la API de portafolio falla
            }
        };

        fetchPortfolioData();
    }, []); // El array vacío `[]` significa "ejecutar solo al montar"

    const handleSearch = (e) => {
        e.preventDefault();
        setTicker(inputValue.toUpperCase());
    };

    // --- LÓGICA DE OPERACIONES (COMPRAR/VENDER) ---
    const getCurrentPrice = () => {
        if (stockData.length === 0) return 0;
        return stockData.at(-1).Close;
    };

    const handleComprar = async (cantidad) => {
        const precio = getCurrentPrice();
        if (precio === 0) {
            alert("Error: No se ha podido obtener el precio actual.");
            return;
        }

        try {
            const response = await fetch(`http://127.0.0.1:5000/api/trade`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    ticker: ticker,
                    cantidad: cantidad,
                    tipo: 'BUY',
                }),
            });

            const data = await response.json();

            if (!response.ok) {
                // Si el backend devuelve un error (ej: fondos insuficientes), lo mostramos
                throw new Error(data.error || 'Error al procesar la compra.');
            }

            // Si la compra es exitosa, el backend nos devuelve el portafolio actualizado
            setPortafolio(data);
            alert(`${cantidad} acciones de ${ticker} compradas!`);

        } catch (err) {
            alert(`Error al comprar: ${err.message}`);
        }
    };

    const handleVender = async (cantidad) => {
        try {
            const response = await fetch(`http://127.0.0.1:5000/api/trade`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    ticker: ticker,
                    cantidad: cantidad,
                    tipo: 'SELL',
                }),
            });

            const data = await response.json();

            if (!response.ok) {
                // Si el backend devuelve un error (ej: no tienes acciones)
                throw new Error(data.error || 'Error al procesar la venta.');
            }

            // Si la venta es exitosa, actualizamos el portafolio
            setPortafolio(data);
            alert(`${cantidad} acciones de ${ticker} vendidas!`);

        } catch (err) {
            alert(`Error al vender: ${err.message}`);
        }
    };

    return (
        <div className="App">
            <header className="App-header">
                <div className="header-content">
                    <div>
                        <h1>Lumina 📈</h1>
                        <p>Tu herramienta de análisis con datos.</p>
                    </div>
                    <button className="dark-mode-toggle" onClick={toggleDarkMode} aria-label="Toggle dark mode">
                        {darkMode ? '☀️' : '🌙'}
                    </button>
                </div>
            </header>
            <div className="main-layout">
                <div className="analysis-column">
                    <div className="stock-viewer">
                        <h2>Visor de Acciones</h2>
                        <form onSubmit={handleSearch} className="search-form">
                            <input type="text" value={inputValue} onChange={(e) => setInputValue(e.target.value)}
                                placeholder="Buscar ticker (ej: MSFT, GOOGL)" className="search-input" />
                            <button type="submit" className="search-button">Buscar</button>
                        </form>

                        {loading && <p>Cargando datos...</p>}
                        {error && <p className="error-message">Error: {error}</p>}

                        {!loading && !error && companyName && (
                            <div className="chart-container">
                                <h3>{companyName} ({ticker})</h3>
                                <div className="indicators-grid-8">
                                    <SignalIndicator event={signalEvent} state={currentState} />
                                    <RsiIndicator rsi={latestRsi} />
                                    <SentimentIndicator score={sentimentScore} count={sentimentNewsCount} />
                                    <MacdIndicator macdData={latestMacd} />
                                    <BollingerBandsIndicator bbData={latestBb} currentPrice={getCurrentPrice()} />
                                    <EmaIndicator emaData={latestEma} />
                                    <StochasticIndicator stochData={latestStoch} />
                                </div>

                                {/* Panel de Noticias */}
                                <div className="news-lstm-container">
                                    <NewsPanel ticker={ticker} />
                                </div>

                                {/* Predictor LSTM */}
                                <div className="news-lstm-container">
                                    <LSTMPredictor ticker={ticker} />
                                </div>

                                <PanelOperar
                                    ticker={ticker}
                                    currentPrice={getCurrentPrice()}
                                    onComprar={handleComprar}
                                    onVender={handleVender}
                                />
                                {stockData.length > 0 ? (
                                    <StockChart chartData={stockData} companyName={companyName} />
                                ) : (
                                    <p>No se encontraron datos históricos para mostrar.</p>
                                )}
                            </div>
                        )}
                    </div>
                </div>
                <div className="portfolio-column">
                    {/* El componente Portafolio ahora recibe los datos de la DB */}
                    <Portafolio portafolio={portafolio} />
                    <PortfolioAnalytics />
                </div>
            </div>
        </div>
    );
}

export default App;