# -*- coding: utf-8 -*-
"""
🚀 Backtesting API Service (FastAPI)
Provides backtesting endpoints using VectorBT engine
"""

from fastapi import FastAPI
from pydantic import BaseModel
from app.data_loader_vbt import load_historical_data_from_api
from backtesting.engine_vectorbt import BacktestEngineVBT

app = FastAPI(title="Backtesting Service with VectorBT")

# -------------------------------------------------
# REQUEST SCHEMA
# -------------------------------------------------
class BacktestRequest(BaseModel):
    ticker: str


# -------------------------------------------------
# RUN BACKTEST
# -------------------------------------------------
@app.post("/api/v1/backtest/run")
def run_backtest(request: BacktestRequest):
    try:
        ticker = request.ticker.upper()

        df = load_historical_data_from_api(ticker)

        engine = BacktestEngineVBT(df)

        market = engine.run_market()
        ml = engine.run_ml()

        confidence = engine.calculate_confidence(
            ml_metrics=ml["ml_metrics"],
            market_metrics=market["metrics"]
        )

        equity_curve, pnl_graph, trade_visual = engine.build_graphs(
            market, ml
        )

        return {
            "confidence_score": confidence,
            "ml_metrics": ml["ml_metrics"],
            "market_metrics": market["metrics"],
            "trading_metrics": ml["trading_metrics"],
            "equity_curve": equity_curve,
            "pnl_graph": pnl_graph,
            "trade_visualization": trade_visual
        }


    except Exception as e:
        print("❌ BACKTEST ERROR:", e)
        raise


@app.get("/")
def health_check():
    return {"status": "active", "service": "Backtesting API (VectorBT)"}
