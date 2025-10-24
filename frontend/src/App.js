import React, { useState, useEffect } from 'react';
import './App.css';
import StockChart from './StockChart.js';
import Portafolio from "./Portafolio";
import PanelOperar from "./PanelOperar";

// Este es nuestro botón (i)
const EthicalTooltip = ({ text }) => {
  const [visible, setVisible] = useState(false);
  return (
    <span className='tooltip-container'>
      <span
        className='tooltip-trigger'
        onMouseEnter={() => setVisible(true)}
        onMouseLeave={() => setVisible(false)}
      >
        (i)
      </span>
      {visible && <div className='tooltip-content'>{text}</div>}
    </span>
  );
};

const SignalIndicator = ({ event, state }) => {
  let signalClass = 'signal-box';
  let signalText = 'Evaluando...';
  let explanation = '';

  if (state === 'GOLDEN') {
    signalClass += 'signal-golden';
    signalText = 'Estado: Alcista (Cruce Dorado)';
    explanation = 'La media móvil rápida (50 días) está POR ENCIMA de la lenta (200 días). Históricamente, esto se considera una señal alcista a largo plazo.';
  } else if (state === 'DEATH') {
    signalClass += 'signal-death';
    signalText = 'Estado: Bajista (Cruce de la Muerte)';
    explanation = 'La media móvil rápida (50 días) está POR DEBAJO de la lenta (200 días). Históricamente, esto se considera una señal bajista a largo plazo.';
  }

  // Si justo HOY hay un evento de cruce, lo mostramos como una alerta
  if (event === 'GOLDEN_CROSS') {
    signalClass += ' signal-alert';
    signalText = '¡ALERTA: Cruce Dorado Detectado!';
    explanation = "¡Hoy, la media de 50 días ha cruzado POR ENCIMA de la de 200 días! Esta es una señal técnica alcista significativa."
  } else if (event === 'DEATH_CROSS') {
    signalClass += ' signal-alert';
    signalText = '¡ALERTA: Cruce de la Muerte Detectado!';
    explanation = "¡Hoy, la media de 50 días ha cruzado POR DEBAJO de la de 200 días! Esta es una señal técnica bajista significativa."
  }

  return (
    <div className={signalClass}>
      <strong>Señal de Medias Móviles</strong>
      <p>{signalText}</p>
      <EthicalTooltip
        text={`${explanation} Recuerda: Esto es solo un indicador técnico basado en datos pasados y no garantiza resultados futuros.`}
      />
    </div>
  )
};

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
      <strong>Señal de Sentimiento (NLP)</strong>
      <p>{sentimentEmoji} {sentimentText}</p>
      <EthicalTooltip
        text={`${explanation} Esto mide el *tono* de los titulares, no su *veracidad* ni el impacto real en el mercado. Es solo un indicador del "ruido" mediático.`}
      />
    </div>
  );
};

// El componente principal de nuestra aplicación
function App() {
  // Usamos el "estado" de React para guardar la información
  const [companyName, setCompanyName] = useState(null); // Para el nombre de la empresa
  const [stockData, setStockData] = useState([]);      // Para los datos del gráfico
  const [ticker, setTicker] = useState('AAPL');        // El ticker que estamos buscando
  const [loading, setLoading] = useState(true);        // Para mostrar un mensaje de "Cargando..."
  const [inputValue, setInputValue] = useState('AAPL'); // Para el valor del input
  const [error, setError] = useState(null);            // Para manejar errores
  const [signalEvent, setSignalEvent] = useState('NONE'); // Para el evento de señal
  const [currentState, setCurrentState] = useState('HOLD'); // Para el estado actual de la señal
  const [sentimentScore, setSentimentScore] = useState(0); // Para el puntaje de sentimiento
  const [sentimentNewsCount, setSentimentNewsCount] = useState(0); // Para el conteo de noticias analizadas
    
    const [portafolio, setPortafolio] = useState({
        efectivo: 100000,   // Empezamos con 100,000 €
        posiciones: {},     // Un objeto: { 'AAPL': 10, 'MSFT': 5 }
    });
    
  // useEffect se ejecuta cuando el componente se "monta" (carga por primera vez)
  // o cuando una de sus dependencias (en este caso, 'ticker') cambia.
  useEffect(() => {
    // Definimos la función asíncrona para pedir los datos
    const fetchData = async () => {
      setLoading(true); // Ponemos "Cargando..."
      setError(null);   // Limpiamos errores anteriores
      setCompanyName(null); // Limpiamos datos viejos
      setStockData([]);    // Limpiamos datos viejos
      setSignalEvent('NONE'); // Limpiamos datos viejos
      setCurrentState('HOLD'); // Limpiamos datos viejos
      setSentimentScore(0); // Limpiamos datos viejos
      setSentimentNewsCount(0); // Limpiamos datos viejos

      try {
        // ¡LA CONEXIÓN! Esta es la URL de nuestro backend
        const response = await fetch(`http://127.0.0.1:5000/api/stock/${ticker}`);

        if (!response.ok) {
          // Si el backend devuelve un error (ej: 404), lo capturamos
          throw new Error(`Error: El ticker "${ticker}" no fue encontrado.`);
        }

        const data = await response.json();

        // Guardamos los datos en nuestro "estado"
        setCompanyName(data.companyName);
        setStockData(data.history);
        setSignalEvent(data.signal_event);
        setCurrentState(data.current_state);
        setSentimentScore(data.sentiment_score);
        setSentimentNewsCount(data.sentiment_news_count);

      } catch (err) {
        // Si algo falla (la API está caída o el ticker no existe)
        setError(err.message);
        setCompanyName(null); // Limpiamos datos viejos
        setStockData([]);
      } finally {
        setLoading(false); // Quitamos el "Cargando..."
      }
    };

    fetchData(); // Ejecutamos la función que acabamos de definir

  }, [ticker]); // El array [ticker] significa: "Vuelve a ejecutar esto si la variable 'ticker' cambia"

  // Función que se ejecuta al enviar el formulario (pulsar Enter o el botón)
  const handleSearch = (e) => {
    e.preventDefault(); // Evita que la página se recargue
    setTicker(inputValue.toUpperCase()); // Actualiza el 'ticker' y dispara el useEffect
  }

  // Función auxiliar para obtener el precio más reciente
  const getCurrentPrice = () => {
    if (stockData.length === 0) return 0;
    // .at(-1) es una forma moderna de coger el último elemento
    return stockData.at(-1).Close;
  };

  const handleComprar = (cantidad) => {
    const precio = getCurrentPrice();
    const costeTotal = precio * cantidad;

    if (costeTotal > portafolio.efectivo) {
      alert("No tienes suficiente efectivo para esta operación.");
      return;
    }

    // Actualizamos el estado del portafolio
    setPortafolio((prevPortafolio) => {
      // Calculamos las nuevas acciones de este ticker
      const accionesActuales = prevPortafolio.posiciones[ticker] || 0;
      const nuevasAcciones = accionesActuales + cantidad;

      return {
        efectivo: prevPortafolio.efectivo - costeTotal,
        posiciones: {
          ...prevPortafolio.posiciones, // Copiamos todas las demás acciones
          [ticker]: nuevasAcciones,     // Actualizamos la acción comprada
        },
      };
    });
  };

  const handleVender = (cantidad) => {
    const accionesActuales = portafolio.posiciones[ticker] || 0;

    if (cantidad > accionesActuales) {
      alert("No tienes suficientes acciones para vender.");
      return;
    }

    const precio = getCurrentPrice();
    const beneficioTotal = precio * cantidad;

    setPortafolio((prevPortafolio) => {
      const nuevasAcciones = accionesActuales - cantidad;
      
      // Creamos una copia de las posiciones
      const nuevasPosiciones = { ...prevPortafolio.posiciones };

      if (nuevasAcciones === 0) {
        // Si vendemos todo, eliminamos el ticker del objeto
        delete nuevasPosiciones[ticker];
      } else {
        // Si no, actualizamos la cantidad
        nuevasPosiciones[ticker] = nuevasAcciones;
      }

      return {
        efectivo: prevPortafolio.efectivo + beneficioTotal,
        posiciones: nuevasPosiciones,
      };
    });
  };
  
  // --- Renderizado (Lo que se ve en HTML) ---
  return (
    <div className="App">
      <header className="App-header">
        <h1>Lumina 📈</h1>
        <p>Tu herramienta de análisis con datos.</p>
      </header>

      <div className="main-layout">
        
        {/* Columna Izquierda: Análisis */}
        <div className="analysis-column">
          <div className="stock-viewer">
            <h2>Visor de Acciones</h2>
            <form onSubmit={handleSearch} className="search-form">
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder="Buscar ticker (ej: MSFT, GOOGL)"
                className="search-input"
              />
              <button type="submit" className="search-button">Buscar</button>
            </form>

            {loading && <p>Cargando datos...</p>}
            {error && <p className="error-message">Error: {error}</p>}
            
            {!loading && !error && companyName && (
              <div className="chart-container">
                <h3>{companyName} ({ticker})</h3>

                <div className="indicators-grid">
                  <SignalIndicator event={signalEvent} state={currentState} />
                  <SentimentIndicator score={sentimentScore} count={sentimentNewsCount} />
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

        {/* Columna Derecha: Portafolio */}
        <div className="portfolio-column">
          <Portafolio portafolio={portafolio} />
        </div>

      </div>
    </div>
  );
}

export default App;
