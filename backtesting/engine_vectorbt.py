# -*- coding: utf-8 -*-
"""
🧮 VectorBT Backtesting Engine
High-performance backtesting with ML signals
Generates complete performance metrics + Market comparison
"""

import numpy as np
import pandas as pd
import vectorbt as vbt


class BacktestEngineVBT:
    INITIAL_CAPITAL = 1_000_000
    FEES = 0.002
    TRADING_DAYS = 252

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    @staticmethod
    def to_py(val):
        if hasattr(val, "item"):   # numpy scalar
            return val.item()
        return val


    # ---------------- MARKET ----------------
    def run_market(self):
        returns = self.df["Close"].pct_change().dropna()

        equity = (1 + returns).cumprod()
        equity = equity * self.INITIAL_CAPITAL

        n_years = len(returns) / self.TRADING_DAYS
        drawdown = (equity - equity.cummax()) / equity.cummax()

        start_equity = equity.iloc[0]
        end_equity = equity.iloc[-1]

        return {
            "metrics": {
                "total_return_pct": self.to_py((end_equity / start_equity - 1) * 100),
                "cagr_pct": self.to_py(((end_equity / start_equity) ** (1 / n_years) - 1) * 100),
                "volatility_pct": self.to_py(returns.std() * np.sqrt(self.TRADING_DAYS) * 100),
                "sharpe_ratio": self.to_py((returns.mean() / returns.std()) * np.sqrt(self.TRADING_DAYS)),
                "max_drawdown_pct": self.to_py(abs(drawdown.min()) * 100),
            },
            "equity": equity
        }


    # ---------------- ML STRATEGY ----------------
    def run_ml(self):
        entries = self.df["Signal"] == 1
        exits = self.df["Signal"] == -1

        pf = vbt.Portfolio.from_signals(
            close=self.df["Close"],
            entries=entries,
            exits=exits,
            init_cash=self.INITIAL_CAPITAL,
            fees=self.FEES,
            freq="1D"
        )

        stats = pf.stats()
        equity = pf.value()
        returns = pf.returns()

        n_years = len(returns.dropna()) / self.TRADING_DAYS
        trades = pf.trades.records_readable.copy()

        # ---------------- TRADE SPLITS ----------------
        winning_trades_df = trades[trades["PnL"] > 0]
        losing_trades_df = trades[trades["PnL"] < 0]

        winning_trades = len(winning_trades_df)
        losing_trades = len(losing_trades_df)

        avg_win_price = (
            (winning_trades_df["Avg Exit Price"] - winning_trades_df["Avg Entry Price"]).mean()
            if winning_trades > 0 else 0.0
        )

        avg_loss_price = (
            (losing_trades_df["Avg Exit Price"] - losing_trades_df["Avg Entry Price"]).mean()
            if losing_trades > 0 else 0.0
        )

        # ---------------- ML METRICS ----------------
        ml_metrics = {
            "total_return_pct": self.to_py(stats["Total Return [%]"]),
            "cagr_pct": self.to_py(
                ((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1) * 100
            ),
            "volatility_pct": self.to_py(
                returns.std() * np.sqrt(self.TRADING_DAYS) * 100
            ),
            "sharpe_ratio": self.to_py(stats["Sharpe Ratio"]),
            "max_drawdown_pct": self.to_py(abs(stats["Max Drawdown [%]"])),
            "total_equity_value": self.to_py(equity.iloc[-1]),
            "total_trades": int(self.to_py(stats["Total Trades"])),
            "win_rate_pct": self.to_py(stats["Win Rate [%]"]),
        }

        # ---------------- TRADING METRICS ----------------
        trading_metrics = {
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "profit_factor": self.to_py(stats["Profit Factor"]),
            "avg_win_price": float(self.to_py(avg_win_price)),
            "avg_loss_price": float(self.to_py(avg_loss_price)),
        }

        return {
            "ml_metrics": ml_metrics,
            "trading_metrics": trading_metrics,
            "equity": equity,
            "trades": trades
        }

    # ---------------- CONFIDENCE ----------------
    @staticmethod
    def calculate_confidence(ml_metrics, market_metrics):
        """
        Returns confidence score between 0 and 100
        based on ML strategy vs Market performance."""
        try:
            mkt_sharpe = market_metrics["sharpe_ratio"]
            mkt_cagr = market_metrics["cagr_pct"]
            ml_sharpe = ml_metrics["sharpe_ratio"]
            ml_cagr = ml_metrics["cagr_pct"]

            # Safety guards
            if mkt_sharpe <= 0 or mkt_cagr <= 0:
                return 0.0

            # Raw confidence (relative edge)
            raw_confidence = (ml_sharpe / mkt_sharpe) * (ml_cagr / mkt_cagr)

            # ---- NORMALIZATION ----
            # raw = 1.0  → 50
            # raw = 2.0+ → 100
            normalized = (raw_confidence / 2.0) * 100

            # Clamp between 0 and 100
            normalized = max(0.0, min(normalized, 100.0))

            return float(round(normalized, 2))

        except Exception:
             return 0.0


    # ---------------- GRAPH DATA ----------------
    def build_graphs(self, market, ml):
        # ---------------- Equity Curve ----------------
        equity_curve = [
            {
                "date": str(d),
                "market": market["equity"].iloc[i] / market["equity"].iloc[0],
                "ml": ml["equity"].iloc[i] / ml["equity"].iloc[0],
            }
            for i, d in enumerate(market["equity"].index)
        ]

        trades_df = ml["trades"]

        # ---------------- Trade PnL Scatter ----------------
        pnl_points = []
        if not trades_df.empty:
            for _, row in trades_df.iterrows():
                pnl_points.append({
                    "entry_date": str(row["Entry Timestamp"]),
                    "exit_date": str(row["Exit Timestamp"]),
                    "entry_price": float(row["Avg Entry Price"]),
                    "exit_price": float(row["Avg Exit Price"]),
                    "pnl": float(row["PnL"]),
                    "return_pct": float(row["Return"] * 100),
                    "direction": row["Direction"],
                    "is_profit": row["PnL"] > 0
                })

        # ---------------- Trade Visualization ----------------
        trade_marks = {
            "dates": self.df.index.astype(str).tolist(),
            "close": self.df["Close"].tolist(),
            "buy_dates": self.df.index[self.df["Signal"] == 1].astype(str).tolist(),
            "sell_dates": self.df.index[self.df["Signal"] == -1].astype(str).tolist()
        }

        return equity_curve, pnl_points, trade_marks
