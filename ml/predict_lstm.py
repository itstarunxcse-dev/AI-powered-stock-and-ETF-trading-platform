import requests
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import ta

# Config
API_URL = "http://127.0.0.1:8000/indicators"
MODEL_PATH = "ml/models/lstm_stock_model.h5"
SCALER_PATH = "ml/models/scaler.pkl"
FEATURES_PATH = "ml/models/feature_cols.pkl"

SEQUENCE_LENGTH = 20
CONFIDENCE_THRESHOLD = 0.60

# Load artifacts
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_cols = joblib.load(FEATURES_PATH)
label_encoder = joblib.load("ml/models/label_encoder.pkl")

label_map = {0: "SELL", 1: "HOLD", 2: "BUY"}

# Fetch data from API
def fetch_from_api(ticker: str) -> pd.DataFrame:
    """Fetch historical data directly using yfinance instead of API"""
    try:
        import yfinance as yf
        
        # Fetch last 100 days to have enough data after calculating indicators
        df = yf.download(
            ticker,
            period="6mo",
            interval="1d",
            auto_adjust=False,
            progress=False,
            timeout=30
        )
        
        if df.empty or len(df) < 51:
            raise ValueError("Not enough historical data")
        
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.columns = [c.lower() for c in df.columns]
        
        return df.tail(100)
    except KeyboardInterrupt:
        raise Exception("Download was cancelled")
    except Exception as e:
        raise Exception(f"Failed to fetch stock data: {e}")

# Prepare features (MATCH TRAINING)
def prepare_sequence(df: pd.DataFrame, ticker: str):
    df = df.sort_values("date").reset_index(drop=True).copy()

    # Add ticker_id using label encoder
    df["ticker_id"] = label_encoder.transform([ticker])[0]

    # Calculate daily return
    df["daily_return"] = df["close"].pct_change(fill_method=None)
    df["volume_change"] = df["volume"].pct_change(fill_method=None)

    # Technical indicators (same as training)
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

    # Lag features (same as training)
    for lag in [1, 2, 3, 5]:
        df[f"close_lag_{lag}"] = df["close"].shift(lag)
        df[f"return_lag_{lag}"] = df["close"].pct_change(lag)

    # Forward fill then backward fill NaN values
    df = df.ffill().bfill()
    df = df.dropna().reset_index(drop=True)

    if len(df) < SEQUENCE_LENGTH:
        raise ValueError(f"Not enough rows for LSTM sequence: {len(df)} < {SEQUENCE_LENGTH}")

    X = df.tail(SEQUENCE_LENGTH)[feature_cols].copy()
    
    # Replace any remaining NaN or infinity
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    X = scaler.transform(X)
    return np.expand_dims(X, axis=0)

# Predict
def predict_stock(ticker: str):
    df = fetch_from_api(ticker)
    X = prepare_sequence(df, ticker)

    probs = model.predict(X, verbose=0)[0]
    confidence = probs.max()
    label = probs.argmax()

    signal = label_map[label] if confidence >= CONFIDENCE_THRESHOLD else "HOLD"

    return {
        "ticker": ticker,
        "signal": signal,
        "confidence": round(confidence * 100, 2),
        "probabilities": {
            "SELL": round(probs[0] * 100, 2),
            "HOLD": round(probs[1] * 100, 2),
            "BUY": round(probs[2] * 100, 2),
        }
    }

# Example Run
if __name__ == "__main__":
    result = predict_stock(input("Enter stock ticker: ").upper())
    print(result)
