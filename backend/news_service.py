"""
Servicio de análisis de sentimiento usando NewsAPI
"""
from newsapi import NewsApiClient
import os
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class NewsService:
    def __init__(self):
        self.api_key = os.getenv('NEWS_API_KEY')
        self.newsapi = NewsApiClient(api_key=self.api_key) if self.api_key else None
        self.sia = SentimentIntensityAnalyzer()
    
    def get_stock_news(self, ticker, company_name=None, days_back=7):
        """
        Obtiene noticias relacionadas con un ticker específico
        
        Args:
            ticker: Símbolo del ticker (ej: 'AAPL')
            company_name: Nombre de la compañía para búsqueda más precisa
            days_back: Días hacia atrás para buscar noticias
            
        Returns:
            Lista de noticias con análisis de sentimiento
        """
        if not self.newsapi:
            return {
                'error': 'NEWS_API_KEY no configurada',
                'articles': [],
                'sentiment_score': 0
            }
        
        try:
            # Calcular fecha de inicio
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            # Buscar noticias
            query = f'{ticker} OR {company_name}' if company_name else ticker
            
            articles = self.newsapi.get_everything(
                q=query,
                language='en',
                sort_by='publishedAt',
                from_param=from_date,
                page_size=20
            )
            
            if articles['status'] != 'ok':
                return {
                    'error': 'Error al obtener noticias',
                    'articles': [],
                    'sentiment_score': 0
                }
            
            # Analizar sentimiento de cada artículo
            analyzed_articles = []
            total_sentiment = 0
            
            for article in articles['articles'][:10]:  # Limitar a 10 artículos
                title = article.get('title', '')
                description = article.get('description', '')
                
                # Analizar sentimiento del título y descripción
                text = f"{title}. {description}"
                sentiment = self.sia.polarity_scores(text)
                
                analyzed_articles.append({
                    'title': title,
                    'description': description,
                    'url': article.get('url'),
                    'publishedAt': article.get('publishedAt'),
                    'source': article.get('source', {}).get('name'),
                    'sentiment': sentiment['compound']
                })
                
                total_sentiment += sentiment['compound']
            
            # Calcular sentimiento promedio
            avg_sentiment = total_sentiment / len(analyzed_articles) if analyzed_articles else 0
            
            return {
                'articles': analyzed_articles,
                'sentiment_score': round(avg_sentiment, 4),
                'total_articles': len(analyzed_articles)
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'articles': [],
                'sentiment_score': 0
            }
    
    def get_sentiment_label(self, score):
        """Convierte el score numérico en etiqueta"""
        if score >= 0.05:
            return 'Positivo'
        elif score <= -0.05:
            return 'Negativo'
        else:
            return 'Neutral'
