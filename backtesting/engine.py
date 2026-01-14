import numpy as np
import pandas as pd
import vectorbt as vbt


class BacktestEngine:
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


    @staticmethod
    def run_backtest(stock_data):
        """
        Static wrapper to run the full backtest workflow.
        Fetches historical signals from the API for the given stock.
        Fallback to local generation if API is down.
        """
        import requests
        from contracts.schema import StockData
        from app.data_loader import load_historical_data
        
        # Extract ticker from stock_data object or dictionary
        ticker = stock_data.symbol if hasattr(stock_data, 'symbol') else stock_data.get('symbol')
        
        df = None
        
        # 1. Try Fetching from Signal API
        try:
            url = "http://localhost:8000/api/v1/ml/signal/historical"
            response = requests.post(url, json={"ticker": ticker}, timeout=2) # Short timeout
            
            if response.status_code == 200:
                data = response.json()
                rows = data.get("rows", [])
                if rows:
                    df = pd.DataFrame(rows)
                    df['Date'] = pd.to_datetime(df['date'])
                    df.set_index('Date', inplace=True)
                    df.rename(columns={
                        "open": "Open", "high": "High", "low": "Low", 
                        "close": "Close", "volume": "Volume", "signal": "Signal"
                    }, inplace=True)
                    print(f"✅ Backtest data fetched from API for {ticker}")
        except Exception as e:
            print(f"⚠️ Backtest API unavailable: {e}. Switching to local data.")
            
        # 2. Fallback to Local Data Loader (if API failed)
        if df is None or df.empty:
            try:
                print(f"🔄 Using local data pipeline for {ticker}...")
                # This uses the robust loader we fixed (CSV -> YFinance -> Indicators)
                df = load_historical_data(ticker=ticker)
            except Exception as e:
                raise ValueError(f"Failed to load backtest data: {e}")

        # 3. Analyze
        try:
            # Serialize and Run Engine
            engine = BacktestEngine(df)
            market_res = engine.run_market()
            ml_res = engine.run_ml()
            
            # Format Results
            confidence = BacktestEngine.calculate_confidence(ml_res['ml_metrics'], market_res['metrics'])
            
            # Prepare formatted metrics object
            class Metrics:
                pass
            
            metrics = Metrics()
            metrics.market_metrics = market_res['metrics']
            metrics.ml_metrics = ml_res['ml_metrics']
            metrics.trading_metrics = ml_res['trading_metrics']
            metrics.confidence_score = confidence
            
            # Prepare data for charts
            metrics.dates = df.index.strftime('%Y-%m-%d').tolist()
            metrics.market_equity = market_res['equity'].tolist()
            metrics.equity_curve = ml_res['equity'].tolist()
            metrics.prices = df['Close'].tolist()
            metrics.volumes = df['Volume'].tolist()
            metrics.returns = df['Close'].pct_change().fillna(0).tolist()
            
            metrics.buy_signals = np.where(df['Signal'] == 1)[0].tolist()
            metrics.sell_signals = np.where(df['Signal'] == -1)[0].tolist()
            
            metrics.trades = ml_res['trades'].to_dict('records') if not ml_res['trades'].empty else []
            
            return metrics
            
        except Exception as e:
            print(f"Backtest engine error: {e}")
            raise e
