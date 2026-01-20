# -*- coding: utf-8 -*-
"""
🤖 ML Signal Service API
Dual-endpoint API for live predictions and historical signals
- Live Signal: For dashboard predictions
- Historical Signals: For backtesting engine
- Market Data: Serves real data via YFinance
"""

import joblib
import yfinance as yf
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
import uvicorn
import os
import sys

# Ensure parent directory is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(
    title="ML Signal Service",
    description="AI-Powered Stock Prediction API with Live & Historical Endpoints",
    version="2.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =================================================
# GLOBAL STATE & CONSTANTS
# =================================================
RF_MODEL = None
XGB_MODEL = None
FEATURES = ["Daily_Return", "Volatility", "SMA_ratio", "EMA_ratio", "MACD"]

# Load models once at startup
try:
    rf_path = os.path.join("ml", "models", "rf_model.pkl")
    xgb_path = os.path.join("ml", "models", "xgb_model.pkl")
    if os.path.exists(rf_path) and os.path.exists(xgb_path):
        RF_MODEL = joblib.load(rf_path)
        XGB_MODEL = joblib.load(xgb_path)
        print("✅ Models loaded successfully at startup")
    else:
        print("⚠️ Models not found. API will return errors for predictions.")
except Exception as e:
    print(f"⚠️ Error loading models: {e}")

# =================================================
# HELPER FUNCTIONS
# =================================================
def flatten_yf_df(df: pd.DataFrame) -> pd.DataFrame:
    """Safely flatten yfinance MultiIndex columns if present."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def calculate_rsi(prices: pd.Series, period=14) -> pd.Series:
    """Standardized RSI calculation."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate technical features - strict order enforced later."""
    df = df.copy()
    df = flatten_yf_df(df)
    
    close = df["Close"]
    
    # 1. Base Features
    df["Daily_Return"] = close.pct_change()
    df["Volatility"] = df["Daily_Return"].rolling(14).std()
    
    # 2. Moving Averages
    df["SMA20"] = close.rolling(20).mean()
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["SMA50"] = close.rolling(50).mean()
    
    df["SMA_ratio"] = close / df["SMA20"]
    df["EMA_ratio"] = close / df["EMA20"]
    
    # 3. MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    
    # 4. RSI (using helper)
    df["RSI"] = calculate_rsi(close)

    df.dropna(inplace=True)
    return df

def fetch_real_market_data(ticker: str, period: str = "1mo", interval: str = "1d"):
    """Fetch and format market data from yfinance."""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            return []
        
        df = flatten_yf_df(df)
        df.reset_index(inplace=True) # Ensure Date is a column
        
        data = []
        for _, row in df.iterrows():
            data.append({
                "ticker": ticker.upper(),
                "date": row["Date"].strftime("%Y-%m-%d"),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"])
            })
        return data
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return []

# =================================================
# SCHEMAS
# =================================================
class TickerRequest(BaseModel):
    ticker: str

# =================================================
# ENDPOINTS
# =================================================
@app.post("/api/v1/ml/signal/live")
def get_live_signal(request: TickerRequest):
    """Generates live trading signal for dashboard."""
    if not RF_MODEL or not XGB_MODEL:
        raise HTTPException(status_code=503, detail="Models not loaded")

    ticker = request.ticker.upper()

    try:
        # Fetch data (enough for feature calculation)
        df = yf.download(ticker, period="6mo", auto_adjust=True, progress=False)
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found for ticker")

        df = create_features(df)
        if len(df) < 1:
            raise HTTPException(status_code=400, detail="Not enough data for features")

        # 1. Enforce Feature Order for Prediction
        X = df[FEATURES].tail(1)
        
        # 2. Predict
        rf_pred = RF_MODEL.predict(X)[0]
        xgb_pred = XGB_MODEL.predict(X)[0]
        avg_pred = (rf_pred + xgb_pred) / 2
        expected_return_pct = float(avg_pred * 100)

        # 3. Get Current Market Context
        current_price = float(df["Close"].iloc[-1])
        rsi = float(df["RSI"].iloc[-1])
        sma50 = float(df["SMA50"].iloc[-1])
        macd = float(df["MACD"].iloc[-1])
        macd_sig = float(df["MACD_Signal"].iloc[-1])

        # 4. Technical Scoring
        technical_score = 0
        reasons = []

        # RSI Logic
        if rsi < 30:
            technical_score += 1
            reasons.append(f"RSI is Oversold ({rsi:.1f})")
        elif rsi > 70:
            technical_score -= 1
            reasons.append(f"RSI is Overbought ({rsi:.1f})")
        else:
            reasons.append(f"RSI is Neutral ({rsi:.1f})")
        
        # Trend Logic
        if current_price > sma50:
            technical_score += 0.5
            reasons.append("Price above 50-day SMA (Bullish)")
        else:
            technical_score -= 0.5
            reasons.append("Price below 50-day SMA (Bearish)")

        # MACD Logic
        if macd > macd_sig:
            technical_score += 0.5
            reasons.append("MACD Bullish Crossover")
        else:
            technical_score -= 0.5
            reasons.append("MACD Bearish Divergence")

        reasons.append(f"ML Model expects {expected_return_pct:.2f}% return")

        # 5. Signal Logic (Strict Sanity Checks)
        signal = "HOLD"
        
        # Confidence Calculation (Dynamic & Sensitive)
        # Base 60. Sensitivity 20. Max boost from return 35.
        # Example: 0.5% return -> 60 + 10 = 70% (before tech boost).
        raw_confidence = 60 + min(abs(expected_return_pct) * 20, 35)
        
        # Feature: Boost confidence if Technicals agree with ML
        if (expected_return_pct > 0 and technical_score > 0) or (expected_return_pct < 0 and technical_score < 0):
             raw_confidence += 10
             
        confidence = min(raw_confidence, 95.0)

        # Buy Logic: Positive return + Technicals OR Strong return (>0.02%)
        if expected_return_pct > 0.02 or (expected_return_pct > 0 and technical_score > 0):
            signal = "BUY"
            if technical_score >= 1.5: confidence = min(confidence + 5, 98.0)
            
        # Sell Logic: Negative return + Technicals OR Strong negative return (<-0.02%)
        # STRICTLY requires negative return to avoid selling winners.
        elif expected_return_pct < -0.02 or (expected_return_pct < 0 and technical_score < 0):
            signal = "SELL"
            if technical_score <= -1.5: confidence = min(confidence + 5, 98.0)
        
        # Dip Buy Logic: Slight negative return but STRONG technicals (Rebound play)
        elif expected_return_pct > -0.1 and technical_score >= 2.0:
             signal = "BUY"
             reasons.append("Technical Override: Strong Rebound Signals")

        # 6. Feature Contribution (Renamed from feature_importance internally)
        signal_contributors = {
            "AI Model Weight": 45.0,
            "RSI Impact": 25.0,
            "Trend (SMA)": 20.0,
            "Momentum (MACD)": 10.0
        }
        
        # Adjust dynamic weights lightly based on score
        if abs(technical_score) > 1.5:
            signal_contributors["AI Model Weight"] = 30.0
            signal_contributors["Technical Confluence"] = 40.0
            signal_contributors["Trend (SMA)"] = 30.0
            if "RSI Impact" in signal_contributors: del signal_contributors["RSI Impact"]

        return {
            "ticker": ticker,
            "signal": signal,
            "expected_return": expected_return_pct,
            "current_price": current_price,
            "confidence": float(round(confidence, 2)),
            "timestamp": datetime.now().isoformat(),
            "reasoning": "\n".join([f"• {r}" for r in reasons]),
            "key_factors": reasons,
            "feature_importance": signal_contributors 
        }

    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/ml/signal/historical")
def get_historical_signals(request: TickerRequest):
    """Generates historical signals for backtesting."""
    if not RF_MODEL or not XGB_MODEL:
        raise HTTPException(status_code=503, detail="Models not loaded")
        
    ticker = request.ticker.upper()

    try:
        # Fetch 5y data
        df = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False)
        if df.empty:
            raise HTTPException(status_code=404, detail="No historical data")

        df = create_features(df) # Handles flattening
        if df.empty:
            raise HTTPException(status_code=404, detail="Not enough data")
            
        # 1. Enforce Feature Order
        X = df[FEATURES]
        
        # 2. Batch Predict
        rf_preds = RF_MODEL.predict(X)
        xgb_preds = XGB_MODEL.predict(X)
        avg_preds = (rf_preds + xgb_preds) / 2
        
        # 3. Vectorized Signal Generation
        # Simple Logic for Backtest: Buy > 0, Sell < 0
        df["Signal"] = np.where(avg_preds > 0, 1, -1)
        
        # Reset index to correctly access Date
        df.reset_index(inplace=True)

        records = []
        for _, row in df.iterrows():
            records.append({
                "date": row["Date"].strftime("%Y-%m-%d"),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]),
                "signal": int(row["Signal"])
            })

        return {
            "ticker": ticker,
            "rows": records,
            "total_rows": len(records)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": RF_MODEL is not None,
        "version": "2.2.0"
    }

# =================================================
# MARKET DATA ENDPOINTS (Unchanged Logic)
# =================================================
@app.get("/supabase/recent/{ticker}")
def get_recent_data(ticker: str, days: int = Query(30)):
    try:
        data = fetch_real_market_data(ticker, period=f"{days+10}d")
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        filtered = [d for d in data if d["date"] >= cutoff]
        filtered.sort(key=lambda x: x["date"], reverse=True)
        return {"ticker": ticker, "data": filtered, "count": len(filtered)}
    except Exception as e:
        return {"data": [], "error": str(e)}

@app.get("/supabase/ticker/{ticker}")
def get_ticker_data(ticker: str, start_date: str = "2024-01-01", limit: int = 100):
    try:
        data = fetch_real_market_data(ticker, period="2y")
        filtered = [d for d in data if d["date"] >= start_date]
        filtered.sort(key=lambda x: x["date"], reverse=True)
        return {"ticker": ticker, "data": filtered[:limit], "count": len(filtered[:limit])}
    except Exception as e:
        return {"data": [], "error": str(e)}

@app.get("/supabase/latest")
def get_latest_market(limit: int = 10):
    TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "SPY"]
    market_data = []
    try:
        for t in TICKERS[:limit]:
            d = fetch_real_market_data(t, period="5d")
            if d: market_data.append(d[-1])
        return {"data": market_data}
    except Exception:
        return {"data": []}

@app.get("/supabase/top-performers")
def get_top_performers(top_n: int = 10):
    TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
    performers = []
    try:
        for t in TICKERS:
            d = fetch_real_market_data(t, period="1mo")
            if d and len(d) > 1:
                chg = ((d[-1]["close"] - d[0]["close"]) / d[0]["close"]) * 100
                performers.append({"ticker": t, "change_pct": round(chg, 2)})
        performers.sort(key=lambda x: x["change_pct"], reverse=True)
        return {"performers": performers[:top_n]}
    except Exception:
        return {"performers": []}

@app.get("/supabase/stats/{ticker}")
def get_ticker_stats(ticker: str, start_date: str = "2024-01-01"):
    try:
        data = fetch_real_market_data(ticker, period="1y")
        filtered = [d for d in data if d["date"] >= start_date]
        if not filtered: return {"stats": {}}
        closes = [d["close"] for d in filtered]
        return {"stats": {
            "price_high": max(closes), "price_low": min(closes),
            "price_avg": sum(closes)/len(closes)
        }}
    except Exception as e:
        return {"stats": {}, "error": str(e)}

@app.get("/supabase/rsi-search")
def search_by_rsi(min_rsi: float = 0, max_rsi: float = 30):
    TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]
    results = []
    try:
        for t in TICKERS:
            d = fetch_real_market_data(t, period="2mo")
            if len(d) > 15:
                # Calculate RSI on the fly from fetched data
                closes = pd.Series([x["close"] for x in d])
                rsi_val = calculate_rsi(closes).iloc[-1]
                if min_rsi <= rsi_val <= max_rsi:
                    d[-1]["rsi"] = rsi_val
                    results.append(d[-1])
        return {"results": results}
    except Exception as e:
        return {"results": [], "error": str(e)}

@app.get("/indicators")
def get_indicators(ticker: str):
    """Provides technical indicators for GenAI analysis."""
    try:
        data = fetch_real_market_data(ticker, period="3mo")
        if not data:
             raise HTTPException(status_code=404, detail="No data found")
        
        df = pd.DataFrame(data)
        df.set_index("date", inplace=True)
        df["Close"] = df["close"] # create_features expects 'Close'

        df = create_features(df)
        if df.empty:
             raise HTTPException(status_code=404, detail="Not enough data for indicators")

        latest = df.iloc[-1]
        
        return {
            "RSI": float(latest["RSI"]),
            "MACD": float(latest["MACD"]),
            "Close_MA20_Ratio": float(latest["SMA_ratio"])
        }
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
