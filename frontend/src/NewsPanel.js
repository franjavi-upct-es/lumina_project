import React, { useState, useEffect } from 'react';
import './NewsPanel.css';

const NewsPanel = ({ ticker }) => {
  const [news, setNews] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [sentimentScore, setSentimentScore] = useState(0);
  const [totalArticles, setTotalArticles] = useState(0);

  useEffect(() => {
    if (ticker) {
      fetchNews();
    }
  }, [ticker]);

  const fetchNews = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`http://localhost:5000/api/news/${ticker}?days=7`);
      const data = await response.json();

      if (response.ok) {
        setNews(data.articles || []);
        setSentimentScore(data.sentiment_score || 0);
        setTotalArticles(data.total_articles || 0);
      } else {
        setError(data.error || 'Error al cargar noticias');
        setNews([]);
      }
    } catch (err) {
      setError('Error de conexión al cargar noticias');
      setNews([]);
    } finally {
      setLoading(false);
    }
  };

  const getSentimentLabel = (score) => {
    if (score >= 0.05) return { label: 'Positivo', class: 'positive' };
    if (score <= -0.05) return { label: 'Negativo', class: 'negative' };
    return { label: 'Neutral', class: 'neutral' };
  };

  const getArticleSentimentClass = (score) => {
    if (score > 0.05) return 'article-positive';
    if (score < -0.05) return 'article-negative';
    return 'article-neutral';
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'Fecha no disponible';
    const date = new Date(dateString);
    return date.toLocaleDateString('es-ES', { 
      day: '2-digit', 
      month: 'short', 
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const sentiment = getSentimentLabel(sentimentScore);

  return (
    <div className="news-panel">
      <div className="news-header">
        <h3>📰 Noticias & Sentimiento</h3>
        <button onClick={fetchNews} disabled={loading} className="refresh-btn">
          {loading ? '🔄' : '↻'} Actualizar
        </button>
      </div>

      {error ? (
        <div className="news-error">
          <p>⚠️ {error}</p>
          <p className="error-hint">
            {error.includes('NEWS_API_KEY') 
              ? 'Configura tu NEWS_API_KEY en el archivo .env del backend'
              : 'Usando fuente alternativa de noticias (yfinance)'}
          </p>
        </div>
      ) : (
        <>
          <div className={`sentiment-summary ${sentiment.class}`}>
            <div className="sentiment-header">
              <span className="sentiment-label">Sentimiento General</span>
              <span className="sentiment-badge">{sentiment.label}</span>
            </div>
            <div className="sentiment-score">
              <div className="score-value">{(sentimentScore * 100).toFixed(1)}%</div>
              <div className="score-bar">
                <div 
                  className="score-fill"
                  style={{ 
                    width: `${Math.abs(sentimentScore) * 100}%`,
                    backgroundColor: sentimentScore > 0 ? '#4caf50' : sentimentScore < 0 ? '#f44336' : '#9e9e9e'
                  }}
                ></div>
              </div>
            </div>
            <p className="articles-count">Basado en {totalArticles} artículos</p>
          </div>

          {loading ? (
            <div className="loading">
              <div className="spinner"></div>
              <p>Cargando noticias...</p>
            </div>
          ) : news.length > 0 ? (
            <div className="news-list">
              {news.map((article, index) => (
                <div key={index} className={`news-article ${getArticleSentimentClass(article.sentiment)}`}>
                  <div className="article-header">
                    <span className="article-source">{article.source || 'Fuente desconocida'}</span>
                    <span className="article-date">{formatDate(article.publishedAt)}</span>
                  </div>
                  <h4 className="article-title">
                    <a href={article.url} target="_blank" rel="noopener noreferrer">
                      {article.title}
                    </a>
                  </h4>
                  {article.description && (
                    <p className="article-description">{article.description}</p>
                  )}
                  <div className="article-footer">
                    <span className={`article-sentiment ${getArticleSentimentClass(article.sentiment)}`}>
                      Sentimiento: {(article.sentiment * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="no-news">
              <p>📭 No hay noticias disponibles en este momento</p>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default NewsPanel;
