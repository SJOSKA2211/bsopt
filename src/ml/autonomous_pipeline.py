import structlog
import pandas as pd
from typing import Dict, Any
from src.ml.scraper import MarketDataScraper
from src.shared.db import get_db_session, MarketData, Base
from src.ml.drift import calculate_psi
from src.ml.trainer import InstrumentedTrainer
from sqlalchemy import create_engine

# Initialize structured logger
logger = structlog.get_logger()

class AutonomousMLPipeline:
    """
    End-to-end autonomous ML pipeline integrating scraping, persistence,
    drift detection, and model optimization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scraper = MarketDataScraper(api_key=config["api_key"])
        self.db_url = config["db_url"]
        self.ticker = config["ticker"]
        self.study_name = config["study_name"]
        self.n_trials = config.get("n_trials", 20)
        
        # Initialize DB (create tables if they don't exist)
        engine = create_engine(self.db_url)
        Base.metadata.create_all(engine)

    def run(self):
        """
        Executes the full pipeline.
        """
        logger.info("pipeline_started", ticker=self.ticker)
        
        try:
            # 1. Scrape Data
            df = self.scraper.fetch_historical_data(self.ticker, "2023-01-01", "2023-12-31")
            logger.info("data_scraped", rows=len(df))
            
            # 2. Persist to DB
            session = get_db_session(self.db_url)
            for _, row in df.iterrows():
                market_data = MarketData(
                    ticker=self.ticker,
                    timestamp=row["timestamp"],
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"]
                )
                session.add(market_data)
            session.commit()
            logger.info("data_persisted")
            
            # 3. Drift Detection (Simplified for integration)
            # In a real scenario, you'd compare current vs historical
            # Here we just demo the call
            historical_prices = df["close"].values
            current_prices = df["close"].values * 1.05 # Mocked "current" data
            psi_score = calculate_psi(historical_prices, current_prices)
            logger.info("drift_checked", psi_score=psi_score)
            
            # 4. Autonomous Training
            trainer = InstrumentedTrainer(study_name=self.study_name)
            
            # Prepare dummy training data for demo
            X = df[["open", "high", "low", "volume"]].values
            y = (df["close"].shift(-1) > df["close"]).iloc[:-1].astype(int).values
            X = X[:-1]
            
            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 10, 50),
                    "max_depth": trial.suggest_int("max_depth", 3, 7),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2)
                }
                return trainer.train_and_evaluate(X, y, params)
            
            study = trainer.optimize(objective, n_trials=self.n_trials)
            logger.info("pipeline_completed", best_accuracy=study.best_value)
            
            return study

        except Exception as e:
            logger.critical("pipeline_failed", error=str(e))
            raise
        finally:
            session.close()

if __name__ == "__main__":
    # Example usage (would typically load from env or config file)
    config = {
        "api_key": "YOUR_POLYGON_API_KEY",
        "db_url": "postgresql://admin:password@localhost:5432/bsopt",
        "ticker": "AAPL",
        "study_name": "aapl_optimization_v1",
        "n_trials": 10
    }
    pipeline = AutonomousMLPipeline(config)
    pipeline.run()
