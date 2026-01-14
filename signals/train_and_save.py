import yfinance as yf
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ======================
# CONFIG
# ======================
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
START_DATE = "2020-01-01"

FEATURES = [
    "Daily_Return",
    "Volatility",
    "SMA_ratio",
    "EMA_ratio",
    "MACD"
]

# ======================
# FEATURE ENGINEERING
# ======================
def create_features(df):

    # ---- FLATTEN MULTI-INDEX ----
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.copy()

    # ---- FORCE CLOSE TO SERIES (CRITICAL FIX) ----
    close = df.loc[:, "Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    # ---- FEATURES ----
    df["Daily_Return"] = close.pct_change()
    df["Volatility"] = df["Daily_Return"].rolling(14).std()

    df["SMA20"] = close.rolling(20).mean()
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()

    # ---- RATIOS (SAFE) ----
    df["SMA_ratio"] = close / df["SMA20"]
    df["EMA_ratio"] = close / df["EMA20"]

    # ---- MACD ----
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26

    # ---- TARGET ----
    df["Target"] = df["Daily_Return"].shift(-1)

    df.dropna(inplace=True)
    return df


# ======================
# TRAIN MODELS
# ======================
def train_models():

    all_data = []

    for ticker in TICKERS:
        print(f"Downloading {ticker}...")
        df = yf.download(
            ticker,
            start=START_DATE,
            auto_adjust=True,
            progress=False
        )

        if df.empty:
            continue

        processed = create_features(df)
        all_data.append(processed)

    data = pd.concat(all_data)

    X = data[FEATURES]
    y = data["Target"]

    # ---- RANDOM FOREST ----
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y)

    # ---- XGBOOST ----
    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb.fit(X, y)

    return rf, xgb


# ======================
# SAVE MODELS
# ======================
if __name__ == "__main__":
    rf_model, xgb_model = train_models()

    import os
    os.makedirs(os.path.join("ml", "models"), exist_ok=True)
    joblib.dump(rf_model, os.path.join("ml", "models", "rf_model.pkl"))
    joblib.dump(xgb_model, os.path.join("ml", "models", "xgb_model.pkl"))

    print("\n✅ Models trained and saved successfully")
