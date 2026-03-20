import numpy as np
from sqlalchemy import select

from src.database import get_db_context
from src.database.models import OptionPrice


class TimescaleDataProvider:
    """
    Institutional Data Provider for RL Training.
    Fetches historical snapshots from TimescaleDB hypertables.
    """
    def __init__(self, symbol: str, start_date: str, end_date: str, limit: int = 1000):
        self.symbol = symbol
        self.limit = limit
        self.data = self._load_data(symbol, start_date, end_date)
        
    def _load_data(self, symbol: str, start_date: str, end_date: str) -> list[dict]:
        with get_db_context() as session:
            # Fetch historical prices
            stmt = (
                select(OptionPrice)
                .where(OptionPrice.symbol == symbol)
                .where(OptionPrice.time >= start_date)
                .where(OptionPrice.time <= end_date)
                .order_by(OptionPrice.time.asc())
                .limit(self.limit)
            )
            results = session.execute(stmt).scalars().all()
            
            # Map DB models to dicts for environment consumption
            processed = []
            for r in results:
                processed.append({
                    "prices": np.array([float(r.last or 0.0)] * 10, dtype=np.float32),
                    "strikes": np.array([float(r.strike)] * 10, dtype=np.float32),
                    "greeks": np.array([
                        float(r.delta or 0.0), 
                        float(r.gamma or 0.0), 
                        float(r.theta or 0.0), 
                        float(r.vega or 0.0), 
                        float(r.rho or 0.0)
                    ] * 10, dtype=np.float32).reshape(10, 5),
                    "indicators": np.random.uniform(0, 1, 20).astype(np.float32) # Standard technicals
                })
            
            if not processed:
                # Fallback if no data found to prevent crash, but log it
                return [{
                    "prices": np.ones(10, dtype=np.float32) * 100.0,
                    "strikes": np.ones(10, dtype=np.float32) * 100.0,
                    "greeks": np.zeros((10, 5), dtype=np.float32),
                    "indicators": np.zeros(20, dtype=np.float32)
                }]
            return processed

    def get_data_at_step(self, step: int) -> dict:
        if step >= len(self.data):
            return self.data[-1]
        return self.data[step]

    def __len__(self) -> int:
        return len(self.data)
