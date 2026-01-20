import structlog
from typing import Dict, Any
from src.ml.scraper import MarketDataScraper
from src.shared.db import get_db_session, MarketData, Base
from src.ml.drift import calculate_ks_test, calculate_psi, PerformanceDriftMonitor
from src.ml.trainer import InstrumentedTrainer
from src.shared.observability import (
    setup_logging, 
    push_metrics,
    DATA_DRIFT_SCORE,
    KS_TEST_SCORE
)
from sqlalchemy import create_engine

# Initialize structured logger
logger = structlog.get_logger()

class AutonomousMLPipeline:
    """
    End-to-end autonomous ML pipeline integrating scraping, persistence,
    drift detection, and model optimization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        setup_logging()
        self.config = config
        self.scraper = MarketDataScraper(
            api_key=config["api_key"], 
            provider=config.get("provider", "auto")
        )
        self.db_url = config["db_url"]
        self.ticker = config["ticker"]
        self.study_name = config["study_name"]
        self.n_trials = config.get("n_trials", 20)
        self.framework = config.get("framework", "xgboost")
        
        # Initialize DB (create tables if they don't exist)
        self.engine = create_engine(self.db_url)
        Base.metadata.create_all(self.engine)
        
        self.performance_monitor = PerformanceDriftMonitor(window_size=5, threshold=0.05)

    def run(self):
        """
        Executes the full pipeline.
        """
        logger.info("pipeline_started", ticker=self.ticker, framework=self.framework)
        
        session = get_db_session(self.db_url)
        try:
            # 1. Scrape Data
            # Note: Dates are hardcoded for demo, would be dynamic in production
            try:
                df = self.scraper.fetch_historical_data(self.ticker, "2023-01-01", "2023-12-31")
            except Exception as e:
                logger.warning("scrape_failed_retrying_mock", error=str(e), provider=self.scraper.provider)
                # Fallback to mock/demo mode
                self.scraper = MarketDataScraper(api_key="DEMO_KEY", provider="mock")
                df = self.scraper.fetch_historical_data(self.ticker, "2023-01-01", "2023-12-31")

            logger.info("data_scraped", rows=len(df))
            
            # 2. Persist to DB
            for _, row in df.iterrows():
                market_data = MarketData(
                    ticker=self.ticker,
                    timestamp=int(row["timestamp"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(row["volume"])
                )
                session.add(market_data)
            session.commit()
            logger.info("data_persisted")
            
            # 3. Data Drift Detection
            historical_prices = df["close"].values
            # Mocking "current" data for demo (e.g., last 100 points)
            current_prices = historical_prices[-100:] if len(historical_prices) > 100 else historical_prices
            
            statistic, p_value = calculate_ks_test(historical_prices, current_prices)
            psi_score = calculate_psi(historical_prices, current_prices)
            
            # Update Prometheus Gauges
            DATA_DRIFT_SCORE.set(psi_score)
            KS_TEST_SCORE.set(p_value)
            
            logger.info("data_drift_checked", 
                        ks_statistic=statistic, 
                        p_value=p_value,
                        psi_score=psi_score)
            
            # 4. Autonomous Training
            trainer = InstrumentedTrainer(study_name=self.study_name)
            
            # Feature engineering (simplified)
            X = df[["open", "high", "low", "volume"]].values
            # Target: 1 if next day close is higher, else 0
            y = (df["close"].shift(-1) > df["close"]).iloc[:-1].astype(int).values
            X = X[:-1]
            
            feature_names = ["open", "high", "low", "volume"]
            dataset_metadata = {
                "ticker": self.ticker,
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
                "rows": str(len(df))
            }
            
            def objective(trial):
                if self.framework == "xgboost":
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 10, 50),
                        "max_depth": trial.suggest_int("max_depth", 3, 7),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                        "framework": "xgboost"
                    }
                elif self.framework == "sklearn":
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 10, 50),
                        "max_depth": trial.suggest_int("max_depth", 3, 7),
                        "framework": "sklearn"
                    }
                else: # pytorch
                    params = {
                        "epochs": trial.suggest_int("epochs", 5, 20),
                        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                        "framework": "pytorch"
                    }
                
                return trainer.train_and_evaluate(
                    X, y, params, 
                    feature_names=feature_names, 
                    dataset_metadata=dataset_metadata
                )
            
            study = trainer.optimize(objective, n_trials=self.n_trials)
            best_accuracy = study.best_value
            
            # 5. Performance Drift Detection
            # Add to history and check for drift
            is_drifted = self.performance_monitor.detect_drift(best_accuracy)
            self.performance_monitor.add_metric(best_accuracy)
            
            logger.info("pipeline_completed", 
                        best_accuracy=best_accuracy, 
                        performance_drift=is_drifted)
            
            push_metrics(job_name="autonomous_pipeline")
            return study

        except Exception as e:
            logger.critical("pipeline_failed", error=str(e))
            push_metrics(job_name="autonomous_pipeline")
            raise
        finally:
            session.close()

if __name__ == "__main__":
    import os
    
    # Determine API Key and Provider
    # Prioritize Polygon if available, as AV was reporting 401
    av_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    poly_key = os.getenv("POLYGON_API_KEY")
    
    if poly_key and poly_key.strip() != "DEMO_KEY":
        api_key = poly_key
        provider = "polygon"
    elif av_key and av_key.strip() != "DEMO_KEY":
        api_key = av_key
        provider = "alpha_vantage"
    else:
        api_key = "DEMO_KEY"
        provider = "mock"

    # Example usage
    config = {
        "api_key": api_key,
        "provider": provider,
        "db_url": os.getenv("DATABASE_URL", "sqlite:///bsopt.db"),
        "ticker": os.getenv("TICKER", "AAPL"),
        "study_name": os.getenv("STUDY_NAME", "aapl_opt_v1"),
        "n_trials": int(os.getenv("N_TRIALS", "5")),
        "framework": os.getenv("FRAMEWORK", "xgboost")
    }
    pipeline = AutonomousMLPipeline(config)
    pipeline.run()