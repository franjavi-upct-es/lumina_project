import yfinance as yf
import pandas as pd
import pandas_ta as ta
from flask import Flask, jsonify, request
from flask_cors import CORS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

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

# Constantes de indicadores
SHORT_WINDOW = 50
LONG_WINDOW = 200
RSI_PERIOD = 14


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


def get_sentiment(ticker_obj):
    try:
        news_list = ticker_obj.news
        if not news_list: return 0, 0
        scores = []
        for news_item in news_list:
            title = news_item.get('title', '')
            if title:
                score = sia.polarity_scores(title)['compound']
                scores.append(score)
        if not scores: return 0, 0
        avg_score = sum(scores) / len(scores)
        return avg_score, len(scores)
    except Exception as e:
        print(f"Error en get_sentiment: {e}")
        return 0, 0


# --- Endpoint de Análisis ---
@app.route('/api/stock/<ticker>', methods=['GET'])
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5y")
        if hist.empty:
            return jsonify(error=f"No se encontraron datos históricos para {ticker}"), 404

        hist[f'SMA_{SHORT_WINDOW}'] = hist['Close'].rolling(window=SHORT_WINDOW).mean()
        hist[f'SMA_{LONG_WINDOW}'] = hist['Close'].rolling(window=LONG_WINDOW).mean()
        signal_event, current_state = detect_cross_signal(hist)
        sentiment_score, news_count = get_sentiment(stock)
        hist[f'RSI_{RSI_PERIOD}'] = ta.rsi(hist['Close'], length=RSI_PERIOD)
        latest_rsi = hist[f'RSI_{RSI_PERIOD}'].iloc[-1]
        if pd.isna(latest_rsi): latest_rsi = 50

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
            latest_rsi=latest_rsi
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
