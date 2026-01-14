from pydantic import BaseModel
from typing import List, Dict

class BacktestResponse(BaseModel):
    ml_metrics: Dict
    market_metrics: Dict
    trading_metrics: Dict
    equity_curve: List[Dict]
    pnl_graph: List[Dict]
    trade_visualization: Dict
