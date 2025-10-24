import yfinance as yf
import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# 1. Configuración de la Aplicación
app = Flask(__name__)
# Habilitamos CORS para permitir que el frontend (en otro dominio)
# se comunique con este backend
CORS(app)

# Definimos los periodos de muestras móviles
SHORT_WINDOW = 50
LONG_WINDOW = 200

# Lo creamos una sola vez para que sea eficiente
sia = SentimentIntensityAnalyzer()


def detect_cross_signal(hist):
    """
    Analiza los últimos dos días para detectar un cruce de medias móviles
    """
    # Asegurarnos de que tenemos al menos 2 días de datos con SMAs válidas
    hist_valid = hist[hist[f"SMA_{LONG_WINDOW}"].notna()]
    if len(hist_valid) < 2:
        return "NONE", "HOLD"  # No hay suficientes datos para detectar el cruce

    # Tomamos los últimos dos días de datos
    last_day = hist_valid.iloc[-1]
    prev_day = hist_valid.iloc[-2]

    sma_short_col = f"SMA_{SHORT_WINDOW}"
    sma_long_col = f"SMA_{LONG_WINDOW}"

    # --- LÓGICAS DE CRUCE ---

    # Estado actual (último día)
    current_state = (
        "GOLDEN" if last_day[sma_short_col] > last_day[sma_long_col] else "DEATH"
    )

    # 1. Detección de CRUCE DORADO (Golden Cross)
    # Si ayer la corta estaba ABAJO de la larga y hoy está ARRIBA
    if (prev_day[sma_short_col] < prev_day[sma_long_col]) and (
        last_day[sma_short_col] > last_day[sma_long_col]
    ):
        return "GOLDEN_CROSS", current_state  # ¡Evento de cruce dorado!

    # 2. Detección de CRUCE DE LA MUERTE (Death Cross)
    # Si ayer la corta estaba ARRIBA de la larga y hoy está ABAJO
    if (prev_day[sma_short_col] > prev_day[sma_long_col]) and (
        last_day[sma_short_col] < last_day[sma_long_col]
    ):
        return "DEATH_CROSS", current_state  # ¡Evento de cruce de la muerte!

    # 3. Sin evento de cruce
    return "NONE", current_state

def get_sentiment(ticker_obj):
    """
    Obtiene las noticias reientes de yfinance y calcula un
    puntaje de sentimiento promedio
    """
    try:
        news_list = ticker_obj.news
        if not news_list:
            return 0, 0 # Sin noticias
        
        scores = []
        # Analizamos cada titular de noticia
        for news_item in news_list:
            title = news_item.get('title', '')
            if title:
                # sia.polarity_scores() devuelve un dic: {neg, neu, pos, compound}
                # 'compound' es el puntaje normalizado (-1 a +1)
                score = sia.polarity_scores(title)['compound']
                scores.append(score)

        if not scores:
            return 0, 0 # No se pudieron puntuar las noticias
        
        # Devolvemos el promedio y la cantidad de noticias analizadas
        avg_score = sum(scores) / len(scores)
        return avg_score, len(scores)
    
    except Exception as e:
        print(f"Error en get_sentiment: {e}")
        return 0, 0

# 2. Definición del "Endpoint"
# Esta es la "dirección" a la que llamará el frontend
# <ticker> es una variable dinámica (ej: AAPL, MSFT, etc.)
@app.route("/api/stock/<ticker>", methods=["GET"])
def get_stock_data(ticker):
    """
    Obtiene los datos históricos y el nombre de una acción
    """
    try:
        # 3. Lógica de Negocio (El "Cerebro")
        stock = yf.Ticker(ticker)

        # Obtenemos el historial de 1 año (1y)
        hist = stock.history(period="5y")

        if hist.empty:
            return jsonify(error=f"No se encontraron datos históricos para {ticker}.")

        # Usamos .rolling() de pandas para calcular la media de una ventana de días
        hist[f"SMA_{SHORT_WINDOW}"] = hist["Close"].rolling(
            window=SHORT_WINDOW).mean()
        hist[f"SMA_{LONG_WINDOW}"] = hist["Close"].rolling(
            window=LONG_WINDOW).mean()

        signal_event, current_state = detect_cross_signal(hist)

        sentiment_score, news_count = get_sentiment(stock)

        # Limpiamos los datos para enviarlos como JSON
        # 1. Reseteamos el índice para que 'Date' sea una columna
        hist.reset_index(inplace=True)
        # 2. Convertimos las fechas a string (JSON no maneja objetos de fecha)
        hist["Date"] = hist["Date"].dt.strftime("%Y-%m-%d")

        # (Los .fillna(None) convierten los "NaN" de pandas a "null" en JSON,
        # lo que la librería de gráficos entiende como "no dibujar aquí")
        columns_to_send = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            f"SMA_{SHORT_WINDOW}",
            f"SMA_{LONG_WINDOW}",
        ]

        # Nos aseguramos de que solo enviamos datos desde que la SMA_50 existe
        hist_filtered = hist[hist[f"SMA_{SHORT_WINDOW}"].notna()]

        # Seleccionamos solo las columnas que queremos enviar
        hist_final = hist_filtered[columns_to_send]

        # Reemplazamos CUALQUIER 'np.nan' restante por 'None'
        # .replace() es más robusto para esto.
        data = hist_final.replace({np.nan: None}).to_dict("records")

        # Obtenemos información básica
        info = stock.info
        company_name = info.get("longName", "Nombre no encontrado")

        # 4. Respuesta al Frontend
        # Devolvemos los datos en formato JSON
        return jsonify(
            companyName=company_name,
            history=data,
            signal_event=signal_event,
            current_state=current_state,
            sentiment_score=sentiment_score,
            sentiment_news_count=news_count
        ), 200

    except Exception as e:
        # Maneja de errores (ej: ticker no encontrado)
        return jsonify(error=str(e)), 404


# 5. Punto de Entrada para Ejecutar el Servidor
if __name__ == "__main__":
    # debug=True hace que el servidor se reinicie solo cada vez que guardas cambios
    app.run(debug=True, port=5000)

