from fastapi import FastAPI
from pydantic import BaseModel
from app.data_loader import load_historical_data
from app.engine import BacktestEngine
import uvicorn

app = FastAPI(title="Backtesting Service")

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

        df = load_historical_data(ticker)

        engine = BacktestEngine(df)

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


    except ValueError as e:
        # Expected data validation error
        print(f"⚠️ Validation Error: {e}")
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        print("❌ BACKTEST ERROR:", e)
        # Re-raise or generic 500
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)