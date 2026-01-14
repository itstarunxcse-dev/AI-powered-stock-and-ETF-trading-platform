import pandas as pd
import os
import sys
import yfinance as yf

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import pipeline adapter
try:
    from data.pipeline_adapter import get_pipeline_data, is_pipeline_available
    from data.pipeline_config import DATA_SOURCE
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    DATA_SOURCE = 'csv'

def load_historical_data(csv_path: str = "ml_trading_signals.csv",
                        ticker: str = None,
                        use_pipeline: bool = None) -> pd.DataFrame:
    """
    Load historical trading data from multiple sources.
    Returns DataFrame with [Open, High, Low, Close, Volume, Signal] columns.
    """
    # ... (skipping unchanged code if any, but replacing whole function for safety)
    should_use_pipeline = use_pipeline if use_pipeline is not None else (DATA_SOURCE == 'pipeline')
    df = pd.DataFrame()

    # Helper for path
    def _get_hist_path(t):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_dir, "data", "historical", f"{t}.csv")

    # Helper for saving
    def _auto_cache(d, t):
        try:
            path = _get_hist_path(t)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            d.to_csv(path)
            print(f"💾 Auto-created local cache: {path}")
        except Exception as e:
            print(f"⚠️ Failed to cache CSV: {e}")

    # 1. Pipeline
    if should_use_pipeline and PIPELINE_AVAILABLE and is_pipeline_available():
        try:
            print(f"📊 Loading data from pipeline...")
            df = load_from_pipeline(ticker)
        except Exception as e:
            print(f"⚠️ Error loading from pipeline: {e}")
    
    # 2. Local Cache (Priority if pipeline skipped/failed)
    if df.empty and ticker:
        try:
            historical_path = _get_hist_path(ticker)
            if os.path.exists(historical_path):
                print(f"📂 Found local historical data for {ticker}")
                df = load_from_csv(historical_path, ticker)
        except Exception as e:
            print(f"⚠️ Error checking historical path: {e}")

    # 3. YFinance
    if df.empty and ticker:
        print(f"🌐 Fetching data from YFinance for {ticker}...")
        df = load_from_yfinance(ticker)
        if not df.empty:
            _auto_cache(df, ticker)

    # 4. Fallback CSV
    if df.empty and os.path.exists(csv_path):
        try:
            df = load_from_csv(csv_path, ticker)
        except Exception as e:
            print(f"⚠️ Error loading from fallback CSV: {e}")

    # --- VALIDATION ---
    if df.empty:
        raise ValueError(f"No data found for {ticker}")

    # Standardize columns
    rename_map = {
        'adj close': 'Close', 'adjclose': 'Close', 'close': 'Close',
        'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume',
        'date': 'Date', 'signal': 'Signal'
    }
    # Create lower-case map
    current_cols = {c.lower(): c for c in df.columns}
    
    # Rename matching
    final_rename = {}
    for k, v in rename_map.items():
        if k in current_cols:
            final_rename[current_cols[k]] = v
    df.rename(columns=final_rename, inplace=True)

    # Ensure required columns
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in required if c not in df.columns]
    
    if missing:
        # Emergency repair for 'Close' using 'Price' if exists
        if 'Close' in missing and 'Price' in df.columns:
             df.rename(columns={'Price': 'Close'}, inplace=True)
             missing.remove('Close')
    
    if missing:
        raise ValueError(f"Data for {ticker} is missing required columns: {missing}. Found: {df.columns.tolist()}")

    # Add Signal if missing
    if 'Signal' not in df.columns:
        df['Signal'] = generate_signals_from_indicators(df)

    return df

def load_from_yfinance(ticker: str) -> pd.DataFrame:
    """Load data directly from YFinance."""
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
        
        if df.empty:
            return pd.DataFrame()
            
        # FIX: Flatten MultiIndex (Price, Ticker) -> Price
        if isinstance(df.columns, pd.MultiIndex):
            # If the columns are (Price, Ticker), we want level 0
            # If they are just Price (old yfinance), accessing level 0 is fine if it creates Index
            # But get_level_values(0) returns an Index, we need to assign it back
            df.columns = df.columns.get_level_values(0)
    
        # Ensure we have a clean index or 'Date' column
        if df.index.name != 'Date' and 'Date' not in df.columns:
             df.index.name = 'Date'
             
        return df
        
    except Exception as e:
        print(f"⚠️ YFinance Download Error: {e}")
        return pd.DataFrame()


def generate_signals_from_indicators(df: pd.DataFrame) -> pd.Series:
    """
    Generate trading signals from technical indicators.
    
    Strategy:
    - Buy (1): RSI < 30 (oversold) OR (MACD > 0 and Close > SMA_50)
    - Sell (-1): RSI > 70 (overbought) OR (MACD < 0 and Close < SMA_50)
    - Hold (0): Otherwise
    """
    signals = pd.Series(0, index=df.index)
    
    rsi = df.get('RSI_14', df.get('rsi', None))
    macd = df.get('MACD', df.get('macd', None))
    close = df.get('Close', df.get('close', None))
    sma_50 = df.get('SMA_50', df.get('ma50', None))
    
    if rsi is not None and macd is not None and close is not None and sma_50 is not None:
        # Buy signals
        buy_condition = (rsi < 30) | ((macd > 0) & (close > sma_50))
        signals[buy_condition] = 1
        
        # Sell signals
        sell_condition = (rsi > 70) | ((macd < 0) & (close < sma_50))
        signals[sell_condition] = -1
    
    return signals
