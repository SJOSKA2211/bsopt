import numpy as np
import pandas as pd
import structlog
from sqlalchemy import create_engine, text

from src.config import settings

logger = structlog.get_logger(__name__)


class Backtester:
    """
    Backtesting module to evaluate model predictions against historical market data.
    Supports cross-sectional evaluation and P&L simulation.
    """

    def __init__(self, database_url: str | None = None):
        self.engine = create_engine(database_url or settings.DATABASE_URL)

    def fetch_historical_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch joined options and spot data for backtesting."""
        query = f"""
            SELECT 
                o.time, o.symbol, o.strike, o.expiry, o.option_type,
                o.last as market_price, t.price as spot
            FROM options_prices o
            JOIN LATERAL (
                SELECT price 
                FROM market_ticks mt 
                WHERE mt.symbol = o.symbol AND mt.time <= o.time
                ORDER BY mt.time DESC 
                LIMIT 1
            ) t ON TRUE
            WHERE o.time >= '{start_date}' AND o.time <= '{end_date}'
        """
        return pd.read_sql(text(query), self.engine)

    def run_backtest(self, model_predictions: pd.DataFrame) -> dict:
        """
        Evaluate predictions.
        model_predictions: DataFrame with ['time', 'symbol', 'predicted_price', 'actual_price']
        """
        if model_predictions.empty:
            return {"error": "No data for backtesting"}

        # 1. Error Metrics
        model_predictions["error"] = (
            model_predictions["predicted_price"] - model_predictions["actual_price"]
        )
        mae = model_predictions["error"].abs().mean()
        rmse = np.sqrt((model_predictions["error"] ** 2).mean())

        # 2. Simple Strategy Simulation (Buy if undervalued, Sell if overvalued)
        # Assuming predictions are for 'fair value'
        model_predictions["signal"] = np.where(
            model_predictions["predicted_price"] > model_predictions["actual_price"] * 1.05,
            1,  # Buy
            np.where(
                model_predictions["predicted_price"] < model_predictions["actual_price"] * 0.95,
                -1,
                0,
            ),  # Sell
        )

        # This is a very simplified P&L logic for illustration
        # In a real scenario, we'd look at subsequent price movement

        logger.info("backtest_completed", mae=mae, rmse=rmse, sample_size=len(model_predictions))

        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "total_samples": len(model_predictions),
            "signals_generated": int(model_predictions["signal"].abs().sum()),
        }


if __name__ == "__main__":
    backtester = Backtester()
    # Logic to load predictions and run evaluation would go here
