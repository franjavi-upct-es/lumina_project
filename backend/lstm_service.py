"""
Servicio de predicción de precios usando LSTM
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
import joblib
from datetime import datetime


class LSTMPredictor:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.sequence_length = 60  # Usar 60 días para predecir el siguiente
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Crear directorio de modelos si no existe
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    def prepare_data(self, df, target_col='Close'):
        """
        Prepara los datos para el entrenamiento LSTM
        
        Args:
            df: DataFrame con datos históricos
            target_col: Columna objetivo a predecir
            
        Returns:
            X_train, X_test, y_train, y_test, scaler
        """
        # Usar solo la columna de cierre
        data = df[[target_col]].values
        
        # Normalizar los datos
        scaled_data = self.scaler.fit_transform(data)
        
        # Crear secuencias
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Reshape para LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Dividir en train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, input_shape):
        """
        Construye la arquitectura del modelo LSTM
        
        Args:
            input_shape: Forma de los datos de entrada
            
        Returns:
            Modelo compilado
        """
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            
            LSTM(units=50),
            Dropout(0.2),
            
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model
    
    def train(self, df, ticker, epochs=50, batch_size=32):
        """
        Entrena el modelo LSTM
        
        Args:
            df: DataFrame con datos históricos
            ticker: Símbolo del ticker
            epochs: Número de épocas
            batch_size: Tamaño del batch
            
        Returns:
            Historial de entrenamiento y métricas
        """
        try:
            # Preparar datos
            X_train, X_test, y_train, y_test = self.prepare_data(df)
            
            # Construir modelo
            model = self.build_model(input_shape=(X_train.shape[1], 1))
            
            # Early stopping para evitar overfitting
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Entrenar
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=[early_stop],
                verbose=1
            )
            
            # Evaluar
            train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
            test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
            
            # Guardar modelo y scaler
            model_path = os.path.join(self.model_dir, f'{ticker}_lstm_model.h5')
            scaler_path = os.path.join(self.model_dir, f'{ticker}_scaler.joblib')
            
            model.save(model_path)
            joblib.dump(self.scaler, scaler_path)
            
            return {
                'success': True,
                'train_loss': float(train_loss),
                'test_loss': float(test_loss),
                'train_mae': float(train_mae),
                'test_mae': float(test_mae),
                'epochs_trained': len(history.history['loss']),
                'model_path': model_path
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def predict(self, df, ticker, days_ahead=5):
        """
        Realiza predicciones usando el modelo entrenado
        
        Args:
            df: DataFrame con datos históricos recientes
            ticker: Símbolo del ticker
            days_ahead: Días a predecir hacia adelante
            
        Returns:
            Predicciones con nivel de confianza
        """
        try:
            model_path = os.path.join(self.model_dir, f'{ticker}_lstm_model.h5')
            scaler_path = os.path.join(self.model_dir, f'{ticker}_scaler.joblib')
            
            # Verificar si existe el modelo
            if not os.path.exists(model_path):
                return {
                    'error': f'Modelo no encontrado para {ticker}. Entrena el modelo primero.',
                    'predictions': []
                }
            
            # Cargar modelo y scaler
            model = load_model(model_path)
            scaler = joblib.load(scaler_path)
            
            # Preparar datos recientes
            recent_data = df['Close'].values[-self.sequence_length:]
            scaled_data = scaler.transform(recent_data.reshape(-1, 1))
            
            # Realizar predicciones iterativas
            predictions = []
            current_sequence = scaled_data.copy()
            
            for _ in range(days_ahead):
                # Reshape para predicción
                X_pred = current_sequence.reshape(1, self.sequence_length, 1)
                
                # Predecir
                pred_scaled = model.predict(X_pred, verbose=0)
                pred_price = scaler.inverse_transform(pred_scaled)[0][0]
                
                predictions.append(float(pred_price))
                
                # Actualizar secuencia para siguiente predicción
                current_sequence = np.append(current_sequence[1:], pred_scaled)
                current_sequence = current_sequence.reshape(-1, 1)
            
            # Calcular nivel de confianza basado en volatilidad histórica
            recent_returns = df['Close'].pct_change().dropna()
            volatility = recent_returns.std()
            
            # Confianza inversa a la volatilidad (más volatilidad = menos confianza)
            confidence = max(0, min(100, 100 * (1 - volatility * 10)))
            
            # Calcular tendencia
            current_price = df['Close'].iloc[-1]
            predicted_change = ((predictions[-1] - current_price) / current_price) * 100
            
            return {
                'ticker': ticker,
                'current_price': float(current_price),
                'predictions': predictions,
                'predicted_change_pct': round(predicted_change, 2),
                'confidence': round(confidence, 2),
                'trend': 'alcista' if predicted_change > 0 else 'bajista',
                'volatility': round(volatility * 100, 2),
                'days_ahead': days_ahead
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'predictions': []
            }
    
    def get_model_info(self, ticker):
        """Obtiene información sobre el modelo entrenado"""
        model_path = os.path.join(self.model_dir, f'{ticker}_lstm_model.h5')
        
        if not os.path.exists(model_path):
            return {'exists': False}
        
        # Obtener fecha de última modificación
        mod_time = os.path.getmtime(model_path)
        mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
        
        return {
            'exists': True,
            'last_trained': mod_date,
            'model_path': model_path
        }
