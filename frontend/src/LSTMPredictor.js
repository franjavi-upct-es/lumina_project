import React, { useState, useEffect } from 'react';
import './LSTMPredictor.css';

const LSTMPredictor = ({ ticker }) => {
  const [modelInfo, setModelInfo] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [daysAhead, setDaysAhead] = useState(5);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (ticker) {
      fetchModelInfo();
    }
  }, [ticker]);

  const fetchModelInfo = async () => {
    try {
      const response = await fetch(`http://localhost:5000/api/model/info/${ticker}`);
      const data = await response.json();
      setModelInfo(data);
      
      // Si el modelo existe, obtener predicción automáticamente
      if (data.exists) {
        fetchPrediction();
      }
    } catch (err) {
      console.error('Error al obtener info del modelo:', err);
    }
  };

  const handleTrain = async () => {
    setIsTraining(true);
    setError(null);
    
    try {
      const response = await fetch(`http://localhost:5000/api/train/${ticker}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ epochs: 50 })
      });
      
      const data = await response.json();
      
      if (response.ok) {
        alert(`Modelo entrenado exitosamente para ${ticker}`);
        fetchModelInfo();
        fetchPrediction();
      } else {
        setError(data.error || 'Error al entrenar el modelo');
      }
    } catch (err) {
      setError('Error de conexión al entrenar el modelo');
    } finally {
      setIsTraining(false);
    }
  };

  const fetchPrediction = async () => {
    setIsPredicting(true);
    setError(null);
    
    try {
      const response = await fetch(`http://localhost:5000/api/predict/${ticker}?days=${daysAhead}`);
      const data = await response.json();
      
      if (response.ok) {
        setPrediction(data);
      } else {
        setError(data.error || 'Error al obtener predicción');
        setPrediction(null);
      }
    } catch (err) {
      setError('Error de conexión al obtener predicción');
      setPrediction(null);
    } finally {
      setIsPredicting(false);
    }
  };

  const getConfidenceClass = (confidence) => {
    if (confidence >= 70) return 'confidence-high';
    if (confidence >= 50) return 'confidence-medium';
    return 'confidence-low';
  };

  const getTrendClass = (trend) => {
    return trend === 'alcista' ? 'trend-up' : 'trend-down';
  };

  return (
    <div className="lstm-predictor">
      <h3>🤖 Predicción LSTM</h3>
      
      {error && (
        <div className="error-message">
          ⚠️ {error}
        </div>
      )}

      <div className="model-status">
        {modelInfo ? (
          modelInfo.exists ? (
            <div className="model-exists">
              <span className="status-badge status-active">✓ Modelo Entrenado</span>
              <p className="model-date">Última actualización: {modelInfo.last_trained}</p>
            </div>
          ) : (
            <div className="model-not-found">
              <span className="status-badge status-inactive">✗ Modelo No Entrenado</span>
              <p>Entrena el modelo para comenzar a hacer predicciones</p>
            </div>
          )
        ) : (
          <p>Cargando información del modelo...</p>
        )}
      </div>

      <div className="control-panel">
        <button 
          onClick={handleTrain} 
          disabled={isTraining}
          className="btn btn-train"
        >
          {isTraining ? '🔄 Entrenando...' : '🎓 Entrenar Modelo'}
        </button>

        {modelInfo?.exists && (
          <div className="prediction-controls">
            <label>
              Días a predecir:
              <input 
                type="number" 
                min="1" 
                max="30" 
                value={daysAhead}
                onChange={(e) => setDaysAhead(e.target.value)}
                className="days-input"
              />
            </label>
            <button 
              onClick={fetchPrediction} 
              disabled={isPredicting}
              className="btn btn-predict"
            >
              {isPredicting ? '🔮 Prediciendo...' : '🔮 Predecir'}
            </button>
          </div>
        )}
      </div>

      {prediction && (
        <div className="prediction-results">
          <div className="prediction-header">
            <h4>Predicción para los próximos {prediction.days_ahead} días</h4>
          </div>

          <div className="prediction-summary">
            <div className="metric-box">
              <span className="metric-label">Precio Actual</span>
              <span className="metric-value current-price">
                ${prediction.current_price?.toFixed(2)}
              </span>
            </div>

            <div className="metric-box">
              <span className="metric-label">Precio Predicho (Día {prediction.days_ahead})</span>
              <span className="metric-value predicted-price">
                ${prediction.predictions[prediction.predictions.length - 1]?.toFixed(2)}
              </span>
            </div>

            <div className={`metric-box ${getTrendClass(prediction.trend)}`}>
              <span className="metric-label">Cambio Esperado</span>
              <span className="metric-value">
                {prediction.predicted_change_pct > 0 ? '+' : ''}
                {prediction.predicted_change_pct}%
              </span>
              <span className="trend-label">
                {prediction.trend === 'alcista' ? '📈 Tendencia Alcista' : '📉 Tendencia Bajista'}
              </span>
            </div>

            <div className={`metric-box ${getConfidenceClass(prediction.confidence)}`}>
              <span className="metric-label">Nivel de Confianza</span>
              <span className="metric-value confidence-value">
                {prediction.confidence}%
              </span>
              <div className="confidence-bar">
                <div 
                  className="confidence-fill"
                  style={{ width: `${prediction.confidence}%` }}
                ></div>
              </div>
            </div>
          </div>

          <div className="prediction-details">
            <h5>Predicciones Día a Día</h5>
            <div className="predictions-list">
              {prediction.predictions.map((price, index) => (
                <div key={index} className="prediction-day">
                  <span className="day-number">Día +{index + 1}</span>
                  <span className="day-price">${price.toFixed(2)}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="prediction-disclaimer">
            <strong>⚠️ Advertencia:</strong> Las predicciones del modelo LSTM son estimaciones 
            basadas en patrones históricos y no garantizan resultados futuros. 
            Volatilidad actual: {prediction.volatility}%
          </div>
        </div>
      )}
    </div>
  );
};

export default LSTMPredictor;
