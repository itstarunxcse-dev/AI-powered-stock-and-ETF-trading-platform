# -*- coding: utf-8 -*-
"""
🤖 ML Signal Service API
Dual-endpoint API for live predictions and historical signals
- Live Signal: For dashboard predictions
- Historical Signals: For backtesting engine
- Market Data: Serves real data via YFinance (acting as Supabase proxy)
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

# Ensure parent directory is in path for imports if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(
    title="ML Signal Service",
    description="AI-Powered Stock Prediction API with Live & Historical Endpoints",
    version="2.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =================================================
# LOAD MODELS
# =================================================
try:
    rf_model = joblib.load(os.path.join("ml", "models", "rf_model.pkl"))
    xgb_model = joblib.load(os.path.join("ml", "models", "xgb_model.pkl"))
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"⚠️ Warning: Models not loaded: {e}")
    rf_model = None
    xgb_model = None

FEATURES = ["Daily_Return", "Volatility", "SMA_ratio", "EMA_ratio", "MACD"]

# =================================================
# FEATURE ENGINEERING
# =================================================
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fix yfinance multi-index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"]

    df["Daily_Return"] = close.pct_change()
    df["Volatility"] = df["Daily_Return"].rolling(14).std()

    df["SMA20"] = close.rolling(20).mean()
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()

    df["SMA_ratio"] = close / df["SMA20"]
    df["EMA_ratio"] = close / df["EMA20"]

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26

    df.dropna(inplace=True)
    return df

# =================================================
# HELPER: YFINANCE DATA PROVIDER
# =================================================
def fetch_real_market_data(ticker: str, period: str = "1mo", interval: str = "1d"):
    """
    Fetch real data using yfinance to populate API responses
    This replaces Supabase if credentials are missing
    """
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            return []
        
        # MEANINGFUL FIX: Flatten MultiIndex columns (common in new yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Reset index to make Date a column
        df.reset_index(inplace=True)
        
        # Format columns to match typical API/DB response
        data = []
        for _, row in df.iterrows():
            record = {
                "ticker": ticker.upper(),
                "date": row["Date"].strftime("%Y-%m-%d"),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"])
            }
            # Calculate RSI if enough data
            data.append(record)
            
        return data  # Returns oldest to newest by default
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return []

def calculate_rsi(prices: pd.Series, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# =================================================
# REQUEST SCHEMA
# =================================================
class TickerRequest(BaseModel):
    ticker: str

# =================================================
# 1️⃣ LIVE SIGNAL API (Dashboard)
# =================================================
@app.post("/api/v1/ml/signal/live")
def get_live_signal(request: TickerRequest):
    """
    Used by Dashboard → Predict Signal button
    Returns only today's signal
    """
    # Auto-load models if missing (double check)
    global rf_model, xgb_model
    if not rf_model or not xgb_model:
         try:
            rf_model = joblib.load(os.path.join("ml", "models", "rf_model.pkl"))
            xgb_model = joblib.load(os.path.join("ml", "models", "xgb_model.pkl"))
         except:
             raise HTTPException(status_code=503, detail="Models not loaded and training failed")
        
    ticker = request.ticker.upper()

    try:
        df = yf.download(ticker, period="6mo", auto_adjust=True, progress=False)
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found for ticker")

        # 1. Base Features (Daily)
        df = create_features(df)
        if len(df) < 1:
             raise HTTPException(status_code=400, detail="Not enough data to generate features")
             
        X = df[FEATURES].tail(1)

        rf_pred = rf_model.predict(X)[0]
        xgb_pred = xgb_model.predict(X)[0]
        avg_pred = (rf_pred + xgb_pred) / 2
        
        # Calculate expected return percentage
        expected_return_pct = avg_pred * 100

        # --- FETCH LIVE PRICE (1m Interval) ---
        try:
            live_df = yf.download(ticker, period="1d", interval="1m", progress=False, auto_adjust=True)
            if not live_df.empty:
                # Handle MultiIndex if present
                if isinstance(live_df.columns, pd.MultiIndex):
                    temp_close = live_df.xs("Close", axis=1, level=0) if "Close" in live_df.columns.get_level_values(0) else live_df.iloc[:, 0]
                    # Extract scalar safely
                    val = temp_close.iloc[-1]
                    current_price = float(val.item()) if hasattr(val, 'item') else float(val)
                else:
                    val = live_df["Close"].iloc[-1]
                    current_price = float(val.item()) if hasattr(val, 'item') else float(val)
            else:
                 val = df["Close"].iloc[-1]
                 current_price = float(val.item()) if hasattr(val, 'item') else float(val)
        except Exception as e:
            print(f"⚠️ Live price fetch failed: {e}")
            # Fallback to daily close
            val = df["Close"].iloc[-1]
            current_price = float(val.item()) if hasattr(val, 'item') else float(val)

        # --- CONFIDENCE CALCULATION (MATCHING PREDICTOR.PY) ---
        # 1. Base ML Confidence (magnitude of prediction)
        # Cap at 95%
        ml_confidence = min(70 + min(abs(expected_return_pct) * 3, 25), 95.0)
        
        # 2. Technical Score
        technical_score = 0
        reasons = []

        # Helper for RSI (Simple calculation for last point)
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        rsi = rsi_series.iloc[-1] if not rsi_series.empty else 50
        
        if rsi < 40: 
            technical_score += 1
            reasons.append(f"RSI is Oversold ({rsi:.1f}), suggesting reversal")
        elif rsi > 60: 
            technical_score -= 1
            reasons.append(f"RSI is Overbought ({rsi:.1f}), possible pullback")
        else:
            reasons.append(f"RSI is Neutral ({rsi:.1f})")
        
        # SMA 50
        sma_50 = df["Close"].rolling(window=50).mean().iloc[-1] if len(df) > 50 else current_price
        if current_price > sma_50: 
            technical_score += 0.5
            reasons.append("Price is above 50-day SMA (Bullish Trend)")
        else:
            reasons.append("Price is below 50-day SMA (Bearish Trend)")
            
        # Add ML Context
        reasons.append(f"ML Algo predicts {expected_return_pct:.2f}% short-term return")
        
        # 3. Decision & Final Confidence
        # Logic: Buy if positive return OR (small negative but strong technicals)
        if expected_return_pct > 0.02 or (expected_return_pct > -0.05 and technical_score > 0):
            signal = "BUY"
            final_confidence = min(ml_confidence * (1.1 if technical_score > 0 else 1.0), 96.0)
            
        elif expected_return_pct < -0.02 or (expected_return_pct < 0.05 and technical_score < 0):
            signal = "SELL"
            final_confidence = min(ml_confidence * (1.1 if technical_score < 0 else 1.0), 96.0)
            
        else:
            signal = "HOLD"
            final_confidence = ml_confidence

        # --- RETURN & IMPORTANCE ---
        # 3. Feature Importance (Dynamic based on Technicals)
        # We can't get exact SHAP values easily here without heavy deps, 
        # so we estimate contribution based on Rule Logic vs ML Weight.
        
        feat_imp = {}
        # Base weight for ML model
        feat_imp["AI Model V3"] = 40.0
        
        # Technical contributions
        if rsi < 30 or rsi > 70: feat_imp["RSI Extremes"] = 25.0
        else: feat_imp["RSI Trend"] = 10.0
        
        if current_price > sma_50: feat_imp["Uptrend Structure"] = 20.0
        else: feat_imp["Downtrend Resistance"] = 20.0
        
        if len(reasons) > 2: feat_imp["Confluence Factors"] = 15.0
        
        # Normalize
        total_imp = sum(feat_imp.values())
        feat_imp = {k: round((v/total_imp)*100, 1) for k,v in feat_imp.items()}

        return {
            "ticker": ticker,
            "signal": signal,
            "expected_return": float(expected_return_pct), # Now returning PERCENTAGE
            "current_price": current_price,
            "confidence": float(round(final_confidence, 2)),
            "timestamp": datetime.now().isoformat(),
            "reasoning": "\n".join([f"• {r}" for r in reasons]),
            "key_factors": reasons,
            "feature_importance": feat_imp
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =================================================
# 2️⃣ HISTORICAL SIGNALS API (Backtesting)
# =================================================
@app.post("/api/v1/ml/signal/historical")
def get_historical_signals(request: TickerRequest):
    """
    Used by Backtesting Engine
    Returns 5 years OHLCV + ML signals
    """
    if not rf_model or not xgb_model:
        raise HTTPException(status_code=503, detail="Models not loaded")
        
    ticker = request.ticker.upper()

    try:
        df = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False)
        if df.empty:
            raise HTTPException(status_code=404, detail="No historical data")

        df = create_features(df)
        if df.empty:
            raise HTTPException(status_code=404, detail="Not enough data for features")

        X = df[FEATURES]
        rf_preds = rf_model.predict(X)
        xgb_preds = xgb_model.predict(X)
        avg_preds = (rf_preds + xgb_preds) / 2

        df["Signal"] = np.where(avg_preds > 0, 1, -1)

        # Convert to JSON-safe structure
        records = []
        for date, row in df.iterrows():
            records.append({
                "date": date.strftime("%Y-%m-%d"),
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

# =================================================
# HEALTH CHECK
# =================================================
@app.get("/health")
def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": rf_model is not None and xgb_model is not None,
        "version": "2.1.0"
    }

# =================================================
# DATA PIPELINE CONTROL
# =================================================
@app.post("/run-pipeline")
def run_pipeline():
    """Trigger data pipeline execution"""
    # In a real app, this would start the Airflow/Prefect job
    # Here we can perhaps trigger a background update of cached data
    return {
        "status": "pipeline_started",
        "timestamp": datetime.now().isoformat(),
        "message": "Data pipeline execution triggered (Simulation)"
    }

# =================================================
# STOCK DATA ENDPOINTS (Implementing Real Logic)
# =================================================

@app.get("/supabase/recent/{ticker}")
def get_recent_data(ticker: str, days: int = Query(30, description="Number of days")):
    """Get recent stock data for a ticker with REAL data"""
    try:
        # Fetch slightly more to ensure coverage
        data = fetch_real_market_data(ticker, period=f"{days+10}d")
        
        # Filter strictly for days requested (approx)
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        filtered_data = [d for d in data if d["date"] >= cutoff_date]
        
        # Reverse to be descending (latest first) as expected by frontend
        filtered_data.sort(key=lambda x: x["date"], reverse=True)
        
        return {
            "ticker": ticker.upper(),
            "days": days,
            "count": len(filtered_data),
            "data": filtered_data
        }
    except Exception as e:
        return {"data": [], "error": str(e)}

@app.get("/supabase/ticker/{ticker}")
def get_ticker_data(
    ticker: str,
    start_date: str = Query("2024-01-01", description="Start date (YYYY-MM-DD)"),
    limit: int = Query(100, description="Max records")
):
    """Get ticker data with date range and limit"""
    try:
        # Convert start_date to period approx or just fetch max and filter
        data = fetch_real_market_data(ticker, period="2y") # Safe default
        
        filtered_data = [d for d in data if d["date"] >= start_date]
        filtered_data.sort(key=lambda x: x["date"], reverse=True)
        
        return {
            "ticker": ticker.upper(),
            "start_date": start_date,
            "limit": limit,
            "count": min(len(filtered_data), limit),
            "data": filtered_data[:limit]
        }
    except Exception as e:
        return {"data": [], "error": str(e)}

# --- Market Overview ---
@app.get("/supabase/latest")
def get_latest_market(limit: int = Query(10, description="Number of latest records")):
    """Get latest market data for top accessible tickers"""
    # Simulate a "market scan" by fetching a basket of popular stocks
    TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "SPY"]
    
    market_data = []
    try:
        # Fetch in bulk if possible, or loop (loop is safer for yfinance reliability here)
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        for t in TICKERS[:limit]:
            # Just get 1 day
            d = fetch_real_market_data(t, period="5d") # 5d to handle weekends
            if d:
                latest = d[-1] # Newest is last in fetch_real_market_data default sort
                latest["ticker"] = t # Ensure ticker is present
                market_data.append(latest)
                
        return {
            "limit": limit,
            "data": market_data,
            "count": len(market_data)
        }
    except Exception as e:
        return {"data": [], "error": str(e)}

@app.get("/supabase/top-performers")
def get_top_performers(top_n: int = Query(10, description="Top N performers")):
    """Get top performing stocks from our basket"""
    TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "INTC"]
    
    performers = []
    try:
        for t in TICKERS:
            d = fetch_real_market_data(t, period="1mo")
            if d and len(d) > 1:
                start_price = d[0]["close"]
                end_price = d[-1]["close"]
                if start_price > 0:
                    change = ((end_price - start_price) / start_price) * 100
                    performers.append({
                        "ticker": t,
                        "change_pct": round(change, 2),
                        "start_price": start_price,
                        "end_price": end_price
                    })
        
        performers.sort(key=lambda x: x["change_pct"], reverse=True)
        
        return {
            "top_n": top_n,
            "performers": performers[:top_n]
        }
    except Exception as e:
        return {"performers": [], "error": str(e)}

# --- Analysis & Filtering ---
@app.get("/supabase/stats/{ticker}")
def get_ticker_stats(
    ticker: str,
    start_date: str = Query("2024-01-01", description="Start date")
):
    """Get statistical analysis for a ticker"""
    try:
        data = fetch_real_market_data(ticker, period="1y")
        filtered_data = [d for d in data if d["date"] >= start_date]
        
        if not filtered_data:
            return {"stats": {}}
            
        closes = [d["close"] for d in filtered_data]
        volumes = [d["volume"] for d in filtered_data]
        
        stats = {
            "ticker": ticker.upper(),
            "period_start": start_date,
            "price_high": max(closes),
            "price_low": min(closes),
            "price_avg": sum(closes) / len(closes),
            "volume_avg": sum(volumes) / len(volumes),
            "data_points": len(closes)
        }
        
        return {"stats": stats}
    except Exception as e:
        return {"stats": {}, "error": str(e)}

@app.get("/supabase/rsi-search")
def search_by_rsi(
    min_rsi: float = Query(0, description="Minimum RSI"),
    max_rsi: float = Query(30, description="Maximum RSI")
):
    """Search stocks by RSI range"""
    TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
    results = []
    
    try:
        for t in TICKERS:
            # Need about 14+ days for RSI
            d = fetch_real_market_data(t, period="2mo") 
            if len(d) > 15:
                # Convert to df for easy calc
                df = pd.DataFrame(d)
                df.set_index("date", inplace=True)
                df["rsi"] = calculate_rsi(df["close"])
                
                current_rsi = df["rsi"].iloc[-1]
                
                if min_rsi <= current_rsi <= max_rsi:
                    latest = d[-1]
                    latest["rsi"] = current_rsi
                    results.append(latest)
                    
        return {
            "min_rsi": min_rsi,
            "max_rsi": max_rsi,
            "results": results
        }
    except Exception as e:
         return {"results": [], "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
