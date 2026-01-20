import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import ta
import os
from pathlib import Path
from datetime import datetime
from contracts.schema import StockData, MLSignal

class MLEngine:
    # Model Metadata
    MODEL_TYPE = "LSTM Deep Learning"
    MODEL_VERSION = "v4.0.0 (LSTM)"
    LAST_TRAINED = "2026-01-19"
    PREDICTION_FREQUENCY = "Real-time"
    
    def __init__(self):
        """Load trained models on initialization"""
        self.models_loaded = False
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.label_encoder = None
        self._load_models()

        # Config
        self.SEQUENCE_LENGTH = 20
        self.CONFIDENCE_THRESHOLD = 0.60
        self.label_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
    
    def _load_models(self):
        """Load LSTM model and artifacts"""
        try:
            models_path = Path(__file__).parent / "models"
            
            model_path = models_path / "lstm_stock_model.h5"
            scaler_path = models_path / "scaler.pkl"
            feats_path = models_path / "feature_cols.pkl"
            label_enc_path = models_path / "label_encoder.pkl"
            
            if model_path.exists() and scaler_path.exists():
                self.model = load_model(model_path)
                self.scaler = joblib.load(scaler_path)
                self.feature_cols = joblib.load(feats_path)
                self.label_encoder = joblib.load(label_enc_path)
                self.models_loaded = True
                print("✅ [MLEngine] Loaded LSTM Deep Learning Model")
            else:
                print(f"⚠️ [MLEngine] Models not found at {models_path}")
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")

    def _prepare_data(self, data: StockData) -> pd.DataFrame:
        """Convert StockData object to DataFrame used by LSTM logic"""
        # Create DataFrame from lists
        df = pd.DataFrame({
            "date": data.dates,
            "open": data.opens,
            "high": data.highs,
            "low": data.lows,
            "close": data.closes,
            "volume": data.volumes
        })
        
        # Ensure numeric
        numeric_cols = ["open", "high", "low", "close", "volume"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        return df

    def _prepare_sequence(self, df: pd.DataFrame, ticker: str):
        """Prepare LSTM sequences (Copied logic from predict_lstm.py)"""
        try:
            df = df.copy()
            
            # Add ticker_id using label encoder (Handle unknown labels safely)
            try:
                df["ticker_id"] = self.label_encoder.transform([ticker])[0]
            except Exception:
                # Fallback for unknown ticker -> use index 0 or similar
                df["ticker_id"] = 0

            # Calculate daily return
            df["daily_return"] = df["close"].pct_change(fill_method=None)
            df["volume_change"] = df["volume"].pct_change(fill_method=None)

            # Technical indicators
            df["ma20"] = ta.trend.SMAIndicator(df["close"], 20).sma_indicator()
            df["ma50"] = ta.trend.SMAIndicator(df["close"], 50).sma_indicator()
            df["close_ma20_ratio"] = df["close"] / df["ma20"]

            df["volatility"] = df["daily_return"].rolling(20).std()
            df["rsi"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
            
            df["ema12"] = ta.trend.EMAIndicator(df["close"], 12).ema_indicator()
            df["ema26"] = ta.trend.EMAIndicator(df["close"], 26).ema_indicator()

            macd = ta.trend.MACD(df["close"])
            df["macd"] = macd.macd()
            df["macd_signal"] = macd.macd_signal()

            # Lag features
            for lag in [1, 2, 3, 5]:
                df[f"close_lag_{lag}"] = df["close"].shift(lag)
                df[f"return_lag_{lag}"] = df["close"].pct_change(lag)

            # Forward fill then backward fill NaN values
            df = df.ffill().bfill()
            df = df.dropna().reset_index(drop=True)

            if len(df) < self.SEQUENCE_LENGTH:
                return None

            X = df.tail(self.SEQUENCE_LENGTH)[self.feature_cols].copy()
            
            # Replace any remaining NaN or infinity
            X = X.fillna(0)
            X = X.replace([np.inf, -np.inf], 0)
            
            X = self.scaler.transform(X)
            return np.expand_dims(X, axis=0)
            
        except Exception as e:
            print(f"Error preparing sequence: {e}")
            return None

    def predict(self, data: StockData) -> MLSignal:
        """Generate prediction using LSTM model"""
        
        # 1. Fallback if models not loaded
        if not self.models_loaded:
             return self._fallback_signal(data)

        try:
            # 2. Data Preparation
            df = self._prepare_data(data)
            
            # 3. Create Features & Sequence
            X = self._prepare_sequence(df, data.symbol)
            
            if X is None:
                return self._fallback_signal(data, reason="Not enough data for LSTM sequence")

            # 4. Predict
            probs = self.model.predict(X, verbose=0)[0]
            confidence = float(np.max(probs))
            label = np.argmax(probs)
            
            # 5. Decode Signal
            action = self.label_map[label] if confidence >= self.CONFIDENCE_THRESHOLD else "HOLD"
            
            # 6. Construct Signal Object
            signal_value = 1 if action == "BUY" else (-1 if action == "SELL" else 0)
            
            conf_percent = confidence * 100
            if conf_percent >= 85: conf_level = "Very High"
            elif conf_percent >= 70: conf_level = "High"
            elif conf_percent >= 55: conf_level = "Medium"
            else: conf_level = "Low"
            
            probabilities = {
                "SELL": float(probs[0]*100),
                "HOLD": float(probs[1]*100),
                "BUY": float(probs[2]*100)
            }

            # Generate Explanation
            explanation = f"🤖 **LSTM AI Analysis for {data.symbol}**\n\n"
            explanation += f"The deep learning model predicts **{action}** with **{conf_percent:.1f}% confidence**.\n\n"
            explanation += "**Class Probabilities:**\n"
            explanation += f"• BUY: {probabilities['BUY']:.1f}%\n"
            explanation += f"• HOLD: {probabilities['HOLD']:.1f}%\n"
            explanation += f"• SELL: {probabilities['SELL']:.1f}%\n"

            return MLSignal(
                action=action,
                signal_value=signal_value,
                timestamp=datetime.now(),
                prediction_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                confidence=conf_percent,
                confidence_level=conf_level,
                reasoning=explanation,
                key_factors=["LSTM Pattern Recognition", "Deep Learning Sequence Analysis"],
                feature_importance=probabilities, # Using probabilities as "importance" for UI visualization
                prediction_frequency=self.PREDICTION_FREQUENCY,
                model_type=self.MODEL_TYPE,
                model_version=self.MODEL_VERSION,
                last_trained=self.LAST_TRAINED
            )

        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return self._fallback_signal(data, reason=f"Model error: {str(e)}")

    def _fallback_signal(self, data: StockData, reason="Heuristic Fallback") -> MLSignal:
        """Basic heuristic fallback if ML fails"""
        rsi = data.rsi[-1] if data.rsi else 50
        action = "HOLD"
        if rsi < 30: action = "BUY"
        elif rsi > 70: action = "SELL"
        
        return MLSignal(
            action=action,
            signal_value=1 if action=="BUY" else -1 if action=="SELL" else 0,
            timestamp=datetime.now(),
            confidence=50.0,
            confidence_level="Low",
            reasoning=f"⚠️ **Fallback Analysis**\n\n{reason}",
            key_factors=["RSI Heuristic"],
            feature_importance={},
            prediction_frequency="Fallback",
            model_type="Heuristic",
            model_version="1.0",
            last_trained="N/A"
        )
