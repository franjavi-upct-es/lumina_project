#!/usr/bin/env python3
"""
Script de prueba para las nuevas funcionalidades de Lumina
Ejecuta: python test_new_features.py
"""

import requests
import json
import time

BASE_URL = "http://localhost:5000"

def print_separator(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60 + "\n")

def test_stock_data():
    """Prueba el endpoint de datos de acciones con nuevos indicadores"""
    print_separator("1. Probando /api/stock/AAPL")
    
    response = requests.get(f"{BASE_URL}/api/stock/AAPL")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Empresa: {data['companyName']}")
        print(f"✅ Señal: {data['signal_event']} ({data['current_state']})")
        print(f"✅ RSI: {data['latest_rsi']:.2f}")
        print(f"✅ Sentimiento: {data['sentiment_score']:.2f} ({data['sentiment_news_count']} noticias)")
        
        # Nuevos indicadores
        bb = data['latest_bb']
        print(f"✅ Bollinger Bands: Upper={bb['upper']:.2f}, Middle={bb['middle']:.2f}, Lower={bb['lower']:.2f}")
        
        ema = data['latest_ema']
        print(f"✅ EMA: Short={ema['ema_short']:.2f}, Long={ema['ema_long']:.2f}")
        
        stoch = data['latest_stoch']
        print(f"✅ Estocástico: %K={stoch['stoch_k']:.2f}, %D={stoch['stoch_d']:.2f}")
        
        if 'news_articles' in data and data['news_articles']:
            print(f"\n📰 Noticias recientes:")
            for i, article in enumerate(data['news_articles'][:3], 1):
                print(f"   {i}. {article['title'][:60]}...")
                print(f"      Sentimiento: {article['sentiment']:.2f}")
    else:
        print(f"❌ Error: {response.status_code}")

def test_news():
    """Prueba el endpoint de noticias"""
    print_separator("2. Probando /api/news/AAPL")
    
    response = requests.get(f"{BASE_URL}/api/news/AAPL?days=7")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Total de artículos: {data['total_articles']}")
        print(f"✅ Sentimiento promedio: {data['sentiment_score']:.2f}")
        
        if data['articles']:
            print(f"\n📰 Primeras 3 noticias:")
            for i, article in enumerate(data['articles'][:3], 1):
                print(f"\n   {i}. {article['title']}")
                print(f"      Fuente: {article['source']}")
                print(f"      Sentimiento: {article['sentiment']:.2f}")
        else:
            print("ℹ️  No hay artículos disponibles (NewsAPI no configurada)")
    else:
        print(f"❌ Error: {response.status_code}")

def test_train_model():
    """Prueba el entrenamiento del modelo LSTM"""
    print_separator("3. Probando /api/train/AAPL")
    
    print("⏳ Entrenando modelo... (esto puede tardar 2-5 minutos)")
    
    response = requests.post(
        f"{BASE_URL}/api/train/AAPL",
        json={"epochs": 10},  # Solo 10 épocas para prueba rápida
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ {data['message']}")
        metrics = data['metrics']
        print(f"✅ Train Loss: {metrics['train_loss']:.6f}")
        print(f"✅ Test Loss: {metrics['test_loss']:.6f}")
        print(f"✅ Train MAE: {metrics['train_mae']:.6f}")
        print(f"✅ Test MAE: {metrics['test_mae']:.6f}")
        print(f"✅ Épocas: {metrics['epochs_trained']}")
    else:
        data = response.json()
        print(f"❌ Error: {data.get('error', 'Unknown error')}")

def test_predict():
    """Prueba las predicciones del modelo LSTM"""
    print_separator("4. Probando /api/predict/AAPL")
    
    response = requests.get(f"{BASE_URL}/api/predict/AAPL?days=5")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Ticker: {data['ticker']}")
        print(f"✅ Precio actual: ${data['current_price']:.2f}")
        print(f"✅ Cambio predicho: {data['predicted_change_pct']:+.2f}%")
        print(f"✅ Tendencia: {data['trend'].upper()}")
        print(f"✅ Confianza: {data['confidence']:.2f}%")
        print(f"✅ Volatilidad: {data['volatility']:.2f}%")
        
        print(f"\n📈 Predicciones para los próximos {data['days_ahead']} días:")
        for i, price in enumerate(data['predictions'], 1):
            change = ((price - data['current_price']) / data['current_price']) * 100
            print(f"   Día +{i}: ${price:.2f} ({change:+.2f}%)")
    else:
        data = response.json()
        print(f"❌ Error: {data.get('error', 'Unknown error')}")
        if 'Modelo no encontrado' in data.get('error', ''):
            print("ℹ️  Ejecuta primero la prueba de entrenamiento")

def test_model_info():
    """Prueba la información del modelo"""
    print_separator("5. Probando /api/model/info/AAPL")
    
    response = requests.get(f"{BASE_URL}/api/model/info/AAPL")
    
    if response.status_code == 200:
        data = response.json()
        if data['exists']:
            print(f"✅ Modelo encontrado")
            print(f"✅ Última vez entrenado: {data['last_trained']}")
            print(f"✅ Ruta: {data['model_path']}")
        else:
            print("❌ Modelo no existe todavía")
    else:
        print(f"❌ Error: {response.status_code}")

def test_portfolio_analytics():
    """Prueba el análisis avanzado de cartera"""
    print_separator("6. Probando /api/portfolio/analytics")
    
    response = requests.get(f"{BASE_URL}/api/portfolio/analytics")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Valor total: €{data['total_value']:.2f}")
        print(f"✅ Valor invertido: €{data['invested_value']:.2f}")
        print(f"✅ Efectivo: {data['cash_percentage']:.2f}%")
        print(f"✅ Ratio de Sharpe: {data['sharpe_ratio']:.4f}")
        print(f"✅ Diversificación: {data['diversification_score']:.4f}")
        
        if data['returns']:
            print(f"\n📊 Retornos por posición:")
            for ticker, ret in data['returns'].items():
                print(f"   {ticker}: {ret:+.2f}%")
    else:
        print(f"❌ Error: {response.status_code}")

def main():
    print("\n" + "="*60)
    print("  🚀 LUMINA - TEST DE NUEVAS FUNCIONALIDADES")
    print("="*60)
    print("\nAsegúrate de que el servidor backend esté corriendo en http://localhost:5000\n")
    
    try:
        # 1. Probar datos de acciones con nuevos indicadores
        test_stock_data()
        time.sleep(1)
        
        # 2. Probar endpoint de noticias
        test_news()
        time.sleep(1)
        
        # 3. Probar información del modelo (antes de entrenar)
        test_model_info()
        time.sleep(1)
        
        # 4. Preguntar si entrenar modelo (tarda tiempo)
        print_separator("Entrenamiento de Modelo LSTM")
        response = input("⚠️  El entrenamiento tarda 2-5 minutos. ¿Continuar? (s/n): ")
        if response.lower() == 's':
            test_train_model()
            time.sleep(1)
            
            # 5. Probar predicciones
            test_predict()
            time.sleep(1)
            
            # 6. Verificar info del modelo nuevamente
            test_model_info()
        else:
            print("⏭️  Saltando entrenamiento y predicción")
        
        time.sleep(1)
        
        # 7. Probar análisis de cartera
        test_portfolio_analytics()
        
        print_separator("✅ PRUEBAS COMPLETADAS")
        print("\nTodas las funcionalidades han sido probadas.")
        print("Visita http://localhost:3000 para ver la interfaz web.\n")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: No se puede conectar al servidor backend")
        print("   Asegúrate de que está corriendo: python backend/app.py\n")
    except Exception as e:
        print(f"\n❌ ERROR INESPERADO: {e}\n")

if __name__ == "__main__":
    main()
