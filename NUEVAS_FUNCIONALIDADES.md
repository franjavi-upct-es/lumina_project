# Guía de Nuevas Funcionalidades 🚀

## 📰 NewsAPI - Análisis de Sentimiento Robusto

### Configuración

1. **Obtén tu API Key gratuita:**
   - Visita [https://newsapi.org/register](https://newsapi.org/register)
   - Regístrate con tu email
   - Copia tu API key

2. **Configura el Backend:**
   - Abre el archivo `backend/.env`
   - Añade tu API key: `NEWS_API_KEY=tu_clave_aqui`

3. **Características:**
   - ✅ Acceso a miles de fuentes de noticias profesionales
   - ✅ Análisis de sentimiento individual por artículo
   - ✅ Score agregado de sentimiento del mercado
   - ✅ Fallback automático a yfinance si no está configurado

### Uso en el Frontend

El **Panel de Noticias** aparece automáticamente al buscar un ticker:
- Muestra las 10 noticias más recientes
- Cada noticia tiene su propio score de sentimiento
- Código de colores: Verde (positivo), Rojo (negativo), Gris (neutral)
- Clic en el título para leer el artículo completo

---

## 🤖 Predicción LSTM - Machine Learning

### Cómo Entrenar un Modelo

1. **Busca un ticker** en la aplicación (ej: AAPL, MSFT)
2. **Localiza el panel "🤖 Predicción LSTM"** debajo de los indicadores
3. **Haz clic en "🎓 Entrenar Modelo"**
   - El entrenamiento puede tardar 2-5 minutos
   - Usa 5 años de datos históricos
   - 50 épocas de entrenamiento por defecto
   - El modelo se guarda en `backend/models/`

### Realizar Predicciones

Una vez entrenado el modelo:

1. **Selecciona los días** a predecir (1-30 días)
2. **Haz clic en "🔮 Predecir"**
3. **Interpreta los resultados:**
   - **Precio Actual vs Precio Predicho:** Comparación visual
   - **Cambio Esperado:** Porcentaje de variación predicho
   - **Tendencia:** Alcista 📈 o Bajista 📉
   - **Nivel de Confianza:** 
     - 🟢 70-100%: Alta confianza
     - 🟡 50-69%: Confianza media
     - 🔴 0-49%: Baja confianza (mercado muy volátil)
   - **Predicciones Día a Día:** Lista detallada de precios proyectados

### ⚠️ Advertencias Importantes

- **NO son garantías:** Las predicciones son estimaciones basadas en patrones históricos
- **Volatilidad:** Mercados muy volátiles reducen la confianza del modelo
- **Eventos externos:** Noticias, crisis o cambios regulatorios no se predicen
- **Uso educativo:** Úsalo para aprender sobre ML, no para decisiones financieras reales

### Métricas del Modelo

Después del entrenamiento verás:
- **Train Loss:** Error en datos de entrenamiento (más bajo = mejor)
- **Test Loss:** Error en datos de validación (debe ser similar al train loss)
- **MAE (Mean Absolute Error):** Error promedio en dólares
- **Épocas Entrenadas:** Número de iteraciones completadas

### Mejores Prácticas

1. **Re-entrena periódicamente:** Actualiza el modelo cada 1-2 meses
2. **Tickers líquidos:** Mejor rendimiento en acciones con alto volumen
3. **Predicciones cortas:** 5-7 días más fiables que 30 días
4. **Compara con indicadores:** Usa RSI, MACD, etc. como validación cruzada

---

## 🎨 Modo Oscuro

- **Toggle en el header:** Botón con ☀️/🌙
- **Persistencia:** Tu preferencia se guarda en localStorage
- **Todos los componentes:** Tema oscuro completo en toda la app

---

## 📊 Análisis Avanzado de Cartera

### Métricas Disponibles

1. **Ratio de Sharpe:**
   - Mide retorno ajustado al riesgo
   - > 1.0: Bueno
   - 0.5-1.0: Neutral
   - < 0.5: Pobre

2. **Índice de Diversificación (HHI):**
   - < 0.25: Bien diversificado
   - 0.25-0.5: Diversificación moderada
   - > 0.5: Poco diversificado (riesgo concentrado)

3. **Retornos Anualizados:**
   - Ganancia/pérdida porcentual por posición
   - Actualización en tiempo real

### Actualización Automática

El panel se actualiza cada 30 segundos con:
- Valor total de la cartera
- Distribución de efectivo
- Métricas de riesgo actualizadas

---

## 🔧 Solución de Problemas

### NewsAPI no funciona

- ✅ Verifica que `NEWS_API_KEY` esté en el archivo `.env`
- ✅ Reinicia el servidor backend (`python app.py`)
- ✅ Si no tienes API key, el sistema usa yfinance automáticamente

### El modelo LSTM no entrena

- ✅ Asegúrate de que el ticker tenga al menos 2 años de datos
- ✅ Verifica que tensorflow esté instalado: `pip list | grep tensorflow`
- ✅ El entrenamiento puede tardar - ¡ten paciencia!

### Errores de importación de TensorFlow

```bash
# Reinstala TensorFlow
pip uninstall tensorflow
pip install tensorflow
```

---

## 📈 Endpoints de la API

### NewsAPI
```bash
GET http://localhost:5000/api/news/AAPL?days=7
```

### Entrenar Modelo
```bash
POST http://localhost:5000/api/train/AAPL
Content-Type: application/json

{
  "epochs": 50
}
```

### Predicción
```bash
GET http://localhost:5000/api/predict/AAPL?days=5
```

### Info del Modelo
```bash
GET http://localhost:5000/api/model/info/AAPL
```

---

## 🎓 Recursos de Aprendizaje

### LSTM y Redes Neuronales
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)

### Análisis de Sentimiento
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
- [NewsAPI Documentation](https://newsapi.org/docs)

### Análisis Técnico
- [Investopedia - Technical Analysis](https://www.investopedia.com/technical-analysis-4689657)

---

## 💡 Consejos Avanzados

1. **Combina señales:** Usa LSTM + RSI + MACD para decisiones más informadas
2. **Backtesting:** Compara predicciones pasadas con precios reales
3. **Gestión de riesgo:** Nunca inviertas basándote solo en un indicador
4. **Educación continua:** El mercado cambia, sigue aprendiendo

---

¿Preguntas? Abre un issue en el repositorio de GitHub.
