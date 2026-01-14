# -*- coding: utf-8 -*-
from typing import List, Dict, Optional, Union, Literal
from pydantic import BaseModel
from datetime import datetime

# ==========================================
# 🟦 Data Engineering Squad Contract
# ==========================================

class StockData(BaseModel):
    symbol: str
    current_price: float
    price_change: float
    price_change_pct: float
    last_updated: datetime
    market_status: str  # "Open" or "Closed"
    
    # Time Series Data (JSON serializable for charts)
    dates: List[str]
    opens: List[float]
    highs: List[float]
    lows: List[float]
    closes: List[float]
    volumes: List[int]
    
    # Technical Indicators
    rsi: List[float]
    sma_20: List[float]
    sma_50: List[float]
    ema_12: List[float]
    ema_26: List[float]
    macd: List[float]
    macd_signal: List[float]
    macd_hist: List[float]

# ==========================================
# 🟩 ML Team Squad Contract
# ==========================================

class MLSignal(BaseModel):
    # 1️⃣ Core Prediction Output
    action: Literal["BUY", "HOLD", "SELL"]
    
    # 2️⃣ Numerical Signal Encoding (for programmatic use)
    signal_value: int  # 1 = BUY, 0 = HOLD, -1 = SELL
    
    # 3️⃣ Prediction Timestamp (aligned with price data)
    timestamp: datetime
    prediction_date: str  # Formatted date for display
    
    # 4️⃣ Gen-AI Explanation (Explainability Layer)
    reasoning: str  # Human-readable explanation
    
    # 5️⃣ Confidence Score
    confidence: float  # 0.0 to 100.0
    confidence_level: Literal["Low", "Medium", "High", "Very High"]  # Categorical
    
    # 6️⃣ Top Features / Indicators Used
    key_factors: List[str]  # Main reasons/indicators
    feature_importance: Dict[str, float]  # Indicator name -> importance score
    
    # 7️⃣ Prediction Frequency Metadata
    prediction_frequency: str  # e.g., "Daily", "Hourly", "Real-time"
    
    # 8️⃣ Model Metadata
    model_type: str  # e.g., "Random Forest", "LSTM", "Hybrid"
    model_version: str  # e.g., "v2.1.0"
    last_trained: str  # e.g., "2024-12-15"

# ==========================================
# 🟧 Backtesting Team Squad Contract
# ==========================================

class TradeRecord(BaseModel):
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    profit_loss: float
    profit_loss_pct: float
    duration_days: int
    trade_type: str  # "LONG" or "SHORT"

class StrategyConfig(BaseModel):
    strategy_name: str
    initial_capital: float
    commission: float  # As decimal (e.g., 0.002 for 0.2%)
    trade_on_close: bool
    position_type: str  # "Long-only", "Short-only", or "Long/Short"
    
class BacktestMetrics(BaseModel):
    # Strategy Configuration
    config: StrategyConfig
    
    # Performance Metrics
    initial_capital: float
    final_equity: float
    total_trades: int
    win_rate: float  # Percentage
    max_drawdown: float  # Percentage
    total_return: float  # Percentage
    annual_return: float  # Percentage (same as CAGR)
    sharpe_ratio: float
    avg_trade_return: float  # Percentage
    volatility: float  # Percentage (annualized)
    cagr: float  # Percentage (Compound Annual Growth Rate)
    confidence_ratio: float # Confidence level of the strategy

    # Market Metrics (Benchmark)
    market_total_return: float
    market_annual_return: float
    market_volatility: float
    market_sharpe_ratio: float
    market_max_drawdown: float
    alpha: float
    beta: float
    information_ratio: float
    
    # Entry/Exit Logic
    entry_rule: str
    exit_rule: str
    position_strategy: str
    
    # Visualization Data
    equity_curve: List[float]
    market_equity: List[float]  # Buy & Hold benchmark
    drawdown_curve: List[float]
    returns: List[float]  # Daily returns for analysis
    dates: List[str]
    volumes: List[int] # Volume data for visualization
    monthly_returns: Dict[str, float]  # "YYYY-MM": return_pct
    
    # Trade History
    trades: List[TradeRecord]
    
    # Price & Signals for Trade Visualization
    prices: List[float]
    buy_signals: List[int]  # Indices where buys occurred
    sell_signals: List[int]  # Indices where sells occurred
    
    # Data Scope Info
    data_points: int
    date_range: str

# ==========================================
# 📦 Aggregated Dashboard State
# ==========================================

class DashboardState(BaseModel):
    stock_data: Optional[StockData] = None
    ml_signal: Optional[MLSignal] = None
    backtest_metrics: Optional[BacktestMetrics] = None
    error: Optional[str] = None
