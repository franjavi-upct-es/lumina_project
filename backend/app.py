import yfinance as yf
import pandas as pd
import pandas_ta as ta
from flask import Flask, jsonify, request
from flask_cors import CORS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
from dotenv import load_dotenv

from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

# Importar servicios nuevos
from news_service import NewsService
from lstm_service import LSTMPredictor

# Cargar variables de entorno
load_dotenv()

# 1. Configuración de la Aplicación
app = Flask(__name__)
CORS(app)

# Configuración de la Base de Datos SQLite
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'lumina.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)
sia = SentimentIntensityAnalyzer()

# Inicializar servicios nuevos
news_service = NewsService()
lstm_predictor = LSTMPredictor()

# Constantes de indicadores
SHORT_WINDOW = 50
LONG_WINDOW = 200
RSI_PERIOD = 14
BB_PERIOD = 20  # Bandas de Bollinger
EMA_SHORT = 12  # EMA rápida
EMA_LONG = 26   # EMA lenta
STOCH_PERIOD = 14  # Estocástico


# --- Modelos de la Base de Datos ---
class PortfolioEfectivo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    efectivo = db.Column(db.Float, nullable=False, default=100000.0)


class PortfolioPosition(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(10), unique=True, nullable=False)
    cantidad = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f'<Position {self.ticker}: {self.cantidad}>'


# --- Funciones de Análisis ---
def detect_cross_signal(hist):
    hist_valid = hist[hist[f'SMA_{LONG_WINDOW}'].notna()]
    if len(hist_valid) < 2: return "NONE", "HOLD"
    last_day = hist_valid.iloc[-1]
    prev_day = hist_valid.iloc[-2]
    sma_short_col = f'SMA_{SHORT_WINDOW}'
    sma_long_col = f'SMA_{LONG_WINDOW}'
    current_state = "GOLDEN" if last_day[sma_short_col] > last_day[sma_long_col] else "DEATH"
    if (prev_day[sma_short_col] < prev_day[sma_long_col]) and \
            (last_day[sma_short_col] > last_day[sma_long_col]):
        return "GOLDEN_CROSS", current_state
    if (prev_day[sma_short_col] > prev_day[sma_long_col]) and \
            (last_day[sma_short_col] < last_day[sma_long_col]):
        return "DEATH_CROSS", current_state
    return "NONE", current_state


def get_sentiment(ticker_obj, ticker_symbol):
    """
    Obtiene el sentimiento usando NewsAPI si está disponible,
    de lo contrario usa yfinance como fallback
    """
    try:
        # Intentar con NewsAPI primero
        if news_service.newsapi:
            # Obtener información de la empresa
            info = ticker_obj.info
            company_name = info.get('longName', info.get('shortName', ticker_symbol))
            
            news_data = news_service.get_stock_news(ticker_symbol, company_name, days_back=7)
            
            if 'error' not in news_data and news_data['articles']:
                return news_data['sentiment_score'], news_data['total_articles'], news_data['articles']
        
        # Fallback a yfinance si NewsAPI no está disponible
        news_list = ticker_obj.news
        if not news_list: 
            return 0, 0, []
        
        scores = []
        articles = []
        for news_item in news_list[:10]:
            title = news_item.get('title', '')
            if title:
                score = sia.polarity_scores(title)['compound']
                scores.append(score)
                articles.append({
                    'title': title,
                    'url': news_item.get('link', ''),
                    'publishedAt': news_item.get('providerPublishTime', ''),
                    'source': news_item.get('publisher', ''),
                    'sentiment': score
                })
        
        if not scores: 
            return 0, 0, []
        
        avg_score = sum(scores) / len(scores)
        return avg_score, len(scores), articles
        
    except Exception as e:
        print(f"Error en get_sentiment: {e}")
        return 0, 0, []


# --- Endpoint de Análisis ---
@app.route('/api/stock/<ticker>', methods=['GET'])
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="10y")
        if hist.empty:
            return jsonify(error=f"No se encontraron datos históricos para {ticker}"), 404

        hist[f'SMA_{SHORT_WINDOW}'] = hist['Close'].rolling(window=SHORT_WINDOW).mean()
        hist[f'SMA_{LONG_WINDOW}'] = hist['Close'].rolling(window=LONG_WINDOW).mean()
        
        # Calcular MACD
        macd = ta.macd(hist['Close'])
        hist = pd.concat([hist, macd], axis=1)
        
        # Calcular RSI
        hist[f'RSI_{RSI_PERIOD}'] = ta.rsi(hist['Close'], length=RSI_PERIOD)
        
        # Calcular Bandas de Bollinger
        bbands = ta.bbands(hist['Close'], length=BB_PERIOD)
        hist = pd.concat([hist, bbands], axis=1)
        
        # Calcular EMAs
        hist[f'EMA_{EMA_SHORT}'] = ta.ema(hist['Close'], length=EMA_SHORT)
        hist[f'EMA_{EMA_LONG}'] = ta.ema(hist['Close'], length=EMA_LONG)
        
        # Calcular Estocástico
        stoch = ta.stoch(hist['High'], hist['Low'], hist['Close'], k=STOCH_PERIOD)
        hist = pd.concat([hist, stoch], axis=1)
        
        signal_event, current_state = detect_cross_signal(hist)
        sentiment_score, news_count, news_articles = get_sentiment(stock, ticker)
        latest_rsi = hist[f'RSI_{RSI_PERIOD}'].iloc[-1]
        if pd.isna(latest_rsi): latest_rsi = 50

        # Datos de MACD
        latest_macd_data = {
            'macd': hist['MACD_12_26_9'].iloc[-1],
            'histogram': hist['MACDh_12_26_9'].iloc[-1],
            'signal': hist['MACDs_12_26_9'].iloc[-1]
        }
        for key, value in latest_macd_data.items():
            if pd.isna(value):
                latest_macd_data[key] = 0

        # Datos de Bandas de Bollinger
        # Verificar qué columnas existen realmente
        bb_columns = [col for col in hist.columns if 'BB' in col]
        print(f"Columnas BB disponibles: {bb_columns}")  # Debug
        
        latest_bb_data = {
            'upper': hist[f'BBU_{BB_PERIOD}_2.0'].iloc[-1] if f'BBU_{BB_PERIOD}_2.0' in hist.columns else 
                     hist[f'BBU_{BB_PERIOD}'].iloc[-1] if f'BBU_{BB_PERIOD}' in hist.columns else 0,
            'middle': hist[f'BBM_{BB_PERIOD}_2.0'].iloc[-1] if f'BBM_{BB_PERIOD}_2.0' in hist.columns else
                      hist[f'BBM_{BB_PERIOD}'].iloc[-1] if f'BBM_{BB_PERIOD}' in hist.columns else 0,
            'lower': hist[f'BBL_{BB_PERIOD}_2.0'].iloc[-1] if f'BBL_{BB_PERIOD}_2.0' in hist.columns else
                     hist[f'BBL_{BB_PERIOD}'].iloc[-1] if f'BBL_{BB_PERIOD}' in hist.columns else 0,
            'bandwidth': hist[f'BBB_{BB_PERIOD}_2.0'].iloc[-1] if f'BBB_{BB_PERIOD}_2.0' in hist.columns else
                         hist[f'BBB_{BB_PERIOD}'].iloc[-1] if f'BBB_{BB_PERIOD}' in hist.columns else 0,
            'percent': hist[f'BBP_{BB_PERIOD}_2.0'].iloc[-1] if f'BBP_{BB_PERIOD}_2.0' in hist.columns else
                       hist[f'BBP_{BB_PERIOD}'].iloc[-1] if f'BBP_{BB_PERIOD}' in hist.columns else 0
        }
        for key, value in latest_bb_data.items():
            if pd.isna(value):
                latest_bb_data[key] = 0

        # Datos de EMA
        latest_ema_data = {
            'short': hist[f'EMA_{EMA_SHORT}'].iloc[-1],
            'long': hist[f'EMA_{EMA_LONG}'].iloc[-1]
        }
        for key, value in latest_ema_data.items():
            if pd.isna(value):
                latest_ema_data[key] = 0

        # Datos de Estocástico
        stoch_columns = [col for col in hist.columns if 'STOCH' in col]
        print(f"Columnas STOCH disponibles: {stoch_columns}")  # Debug
        
        latest_stoch_data = {
            'k': hist[f'STOCHk_{STOCH_PERIOD}_3_3'].iloc[-1] if f'STOCHk_{STOCH_PERIOD}_3_3' in hist.columns else
                 hist['STOCHk_14_3_3'].iloc[-1] if 'STOCHk_14_3_3' in hist.columns else 50,
            'd': hist[f'STOCHd_{STOCH_PERIOD}_3_3'].iloc[-1] if f'STOCHd_{STOCH_PERIOD}_3_3' in hist.columns else
                 hist['STOCHd_14_3_3'].iloc[-1] if 'STOCHd_14_3_3' in hist.columns else 50
        }
        for key, value in latest_stoch_data.items():
            if pd.isna(value):
                latest_stoch_data[key] = 50

        hist.reset_index(inplace=True)
        hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
        columns_to_send = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', f'SMA_{SHORT_WINDOW}',
                           f'SMA_{LONG_WINDOW}']
        hist_filtered = hist[hist[f'SMA_{LONG_WINDOW}'].notna()]
        data = hist_filtered[columns_to_send].where(pd.notna(hist_filtered[columns_to_send]), None).to_dict('records')
        info = stock.info
        companyName = info.get('longName', 'Nombre no encontrado')

        return jsonify(
            companyName=companyName,
            history=data,
            signal_event=signal_event,
            current_state=current_state,
            sentiment_score=sentiment_score,
            sentiment_news_count=news_count,
            news_articles=news_articles[:5],  # Enviar solo las 5 noticias más recientes
            latest_rsi=latest_rsi,
            latest_macd=latest_macd_data,
            latest_bb=latest_bb_data,
            latest_ema=latest_ema_data,
            latest_stoch=latest_stoch_data
        ), 200
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    """
    Devuelve el estado actual del portafolio desde la DB.
    """
    try:
        # Buscamos el efectivo (asumimos que solo hay 1 fila/usuario)
        cash = PortfolioEfectivo.query.first()
        if not cash:
            # Si la DB está vacía por alguna razón, la inicializamos
            cash = PortfolioEfectivo(efectivo=100000.0)
            db.session.add(cash)
            db.session.commit()

        # Buscamos todas las posiciones
        positions = PortfolioPosition.query.all()

        # Convertimos la lista de posiciones a un diccionario
        # (Formato: {'AAPL': 10, 'MSFT': 5})
        posiciones_dict = {p.ticker: p.cantidad for p in positions}

        return jsonify(
            efectivo=cash.efectivo,
            posiciones=posiciones_dict
        ), 200

    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route('/api/portfolio/analytics', methods=['GET'])
def get_portfolio_analytics():
    """
    Calcula métricas avanzadas del portafolio: Sharpe ratio, diversificación, retorno
    """
    try:
        cash = PortfolioEfectivo.query.first()
        positions = PortfolioPosition.query.all()
        
        if not positions:
            return jsonify(
                total_value=cash.efectivo if cash else 100000,
                invested_value=0,
                cash_percent=100,
                diversification=0,
                diversification_score=0,
                sharpe_ratio=0,
                returns={},
                positions_count=0,
                message="No hay posiciones en el portafolio"
            ), 200
        
        # Calcular valor actual del portafolio
        portfolio_values = {}
        total_invested = 0
        returns_data = {}
        
        for position in positions:
            try:
                stock = yf.Ticker(position.ticker)
                current_price = stock.info.get('currentPrice')
                if not current_price:
                    hist = stock.history(period="1d")
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                    else:
                        continue
                
                position_value = current_price * position.cantidad
                portfolio_values[position.ticker] = position_value
                total_invested += position_value
                
                # Calcular retorno (necesitaríamos precio de compra, usamos estimación)
                hist_long = stock.history(period="1y")
                if not hist_long.empty:
                    year_ago_price = hist_long['Close'].iloc[0]
                    returns_data[position.ticker] = ((current_price - year_ago_price) / year_ago_price) * 100
                
            except Exception as e:
                print(f"Error calculando {position.ticker}: {e}")
                continue
        
        total_value = total_invested + (cash.efectivo if cash else 0)
        cash_percent = (cash.efectivo / total_value * 100) if total_value > 0 else 100
        
        # Índice de diversificación (Herfindahl-Hirschman Index inverso)
        if total_invested > 0:
            hhi = sum((val / total_invested) ** 2 for val in portfolio_values.values())
            diversification = (1 / hhi) if hhi > 0 else 1
        else:
            diversification = 0
        
        # Sharpe Ratio simplificado (necesitaríamos más datos históricos para uno real)
        if returns_data:
            avg_return = sum(returns_data.values()) / len(returns_data)
            # Asumimos tasa libre de riesgo del 2%
            risk_free_rate = 2.0
            # Simplificación: usamos desviación estándar de los retornos
            if len(returns_data) > 1:
                import statistics
                std_dev = statistics.stdev(returns_data.values())
                sharpe_ratio = (avg_return - risk_free_rate) / std_dev if std_dev > 0 else 0
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        return jsonify(
            total_value=total_value,
            invested_value=total_invested,
            cash_value=cash.efectivo if cash else 0,
            cash_percent=cash_percent,
            diversification=diversification,
            diversification_score=min(diversification / len(positions) * 100, 100) if positions else 0,
            sharpe_ratio=sharpe_ratio,
            returns=returns_data,
            positions_count=len(positions)
        ), 200
        
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route('/api/trade', methods=['POST'])
def execute_trade():
    """
    Ejecuta una orden de compra o venta y la guarda en la DB.
    """
    try:
        # 1. Obtenemos los datos de la petición del frontend
        data = request.json
        ticker = data.get('ticker')
        cantidad = int(data.get('cantidad'))
        tipo_orden = data.get('tipo')  # "BUY" o "SELL"

        if not ticker or not cantidad or not tipo_orden:
            return jsonify(error="Faltan datos en la petición (ticker, cantidad, tipo)"), 400

        # 2. ¡Importante! Obtenemos el precio actual desde el backend
        # NUNCA confíes en el precio que envía el frontend.
        try:
            stock = yf.Ticker(ticker)
            # .info['currentPrice'] es más rápido que .history()
            current_price = stock.info.get('currentPrice')
            if not current_price:
                # Fallback por si 'currentPrice' no está
                current_price = stock.history(period="1d")['Close'].iloc[-1]
        except Exception:
            return jsonify(error=f"No se pudo obtener el precio actual de {ticker}"), 404

        # 3. Obtenemos el estado actual del portafolio desde la DB
        cash = PortfolioEfectivo.query.first()
        position = PortfolioPosition.query.filter_by(ticker=ticker).first()

        # 4. Lógica de la operación
        if tipo_orden == "BUY":
            coste_total = current_price * cantidad
            if coste_total > cash.efectivo:
                return jsonify(error="Fondos insuficientes"), 400

            # Actualizamos el efectivo
            cash.efectivo -= coste_total

            # Actualizamos/Creamos la posición
            if position:
                position.cantidad += cantidad
            else:
                position = PortfolioPosition(ticker=ticker, cantidad=cantidad)
                db.session.add(position)

        elif tipo_orden == "SELL":
            acciones_actuales = position.cantidad if position else 0
            if cantidad > acciones_actuales:
                return jsonify(error="No tienes suficientes acciones para vender"), 400

            beneficio_total = current_price * cantidad

            # Actualizamos el efectivo
            cash.efectivo += beneficio_total

            # Actualizamos/Eliminamos la posición
            position.cantidad -= cantidad
            if position.cantidad == 0:
                db.session.delete(position)

        else:
            return jsonify(error="Tipo de orden no válido (debe ser BUY o SELL)"), 400

        # 5. Guardamos los cambios en la DB
        db.session.commit()

        # 6. Devolvemos el nuevo estado del portafolio (llamando al otro endpoint)
        return get_portfolio()

    except Exception as e:
        db.session.rollback()  # Revertimos los cambios si algo falla
        return jsonify(error=str(e)), 500


# --- Endpoints de Machine Learning ---
@app.route('/api/train/<ticker>', methods=['POST'])
def train_model(ticker):
    """
    Entrena un modelo LSTM para un ticker específico
    """
    try:
        # Obtener parámetros opcionales
        epochs = request.json.get('epochs', 50) if request.json else 50
        
        # Obtener datos históricos
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5y")  # 5 años para entrenamiento
        
        if hist.empty or len(hist) < 100:
            return jsonify(error=f"Datos insuficientes para entrenar modelo de {ticker}"), 400
        
        # Entrenar modelo
        result = lstm_predictor.train(hist, ticker, epochs=epochs)
        
        if result['success']:
            return jsonify(
                message=f"Modelo entrenado exitosamente para {ticker}",
                metrics=result
            ), 200
        else:
            return jsonify(error=result.get('error', 'Error desconocido')), 500
            
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route('/api/predict/<ticker>', methods=['GET'])
def predict_price(ticker):
    """
    Realiza predicciones de precio usando el modelo LSTM entrenado
    """
    try:
        # Obtener parámetros opcionales
        days_ahead = int(request.args.get('days', 5))
        
        # Obtener datos históricos recientes
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")  # 3 meses de datos recientes
        
        if hist.empty:
            return jsonify(error=f"No se encontraron datos para {ticker}"), 404
        
        # Realizar predicción
        prediction = lstm_predictor.predict(hist, ticker, days_ahead=days_ahead)
        
        if 'error' in prediction:
            return jsonify(error=prediction['error']), 404
        
        return jsonify(prediction), 200
        
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route('/api/model/info/<ticker>', methods=['GET'])
def get_model_info(ticker):
    """
    Obtiene información sobre el modelo entrenado para un ticker
    """
    try:
        info = lstm_predictor.get_model_info(ticker)
        return jsonify(info), 200
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route('/api/news/<ticker>', methods=['GET'])
def get_news(ticker):
    """
    Obtiene noticias recientes con análisis de sentimiento
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        company_name = info.get('longName', info.get('shortName', ticker))
        
        days_back = int(request.args.get('days', 7))
        news_data = news_service.get_stock_news(ticker, company_name, days_back=days_back)
        
        if 'error' in news_data:
            return jsonify(error=news_data['error']), 400
        
        return jsonify(news_data), 200
        
    except Exception as e:
        return jsonify(error=str(e)), 500


# Punto de entrada
if __name__ == '__main__':
    # Necesitamos `app.app_context()` para crear la DB si no existe
    with app.app_context():
        db.create_all()
        # Aseguramos que la fila de efectivo exista al arrancar
        if not PortfolioEfectivo.query.first():
            initial_cash = PortfolioEfectivo(efectivo=100000.0)
            db.session.add(initial_cash)
            db.session.commit()

    app.run(debug=True, port=5000)
