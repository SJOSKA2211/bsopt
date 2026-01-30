import structlog
import asyncio
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Any, List
from src.ml.scraper import MarketDataScraper
from src.shared.db import get_db_session, Base
from src.database.models import MarketTick
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
        self.n_trials = config.get("n_trials", 50) # Increased default trials
        self.framework = config.get("framework", "xgboost")
        
        # Initialize DB (create tables if they don't exist)
        self.engine = create_engine(self.db_url)
        Base.metadata.create_all(self.engine)
        
        self.performance_monitor = PerformanceDriftMonitor(window_size=10, threshold=0.03)

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates technical indicators for enhanced feature engineering.
        """
        # Ensure data is sorted by timestamp
        df = df.sort_values("timestamp")
        
        # Add Technical Indicators using pandas-ta
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.bbands(length=20, std=2, append=True)
        df.ta.sma(length=50, append=True)
        df.ta.ema(length=20, append=True)
        
        # Add returns and volatility
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(window=20).std()
        
        # Drop rows with NaN values created by indicators
        return df.dropna()

    async def run(self):
        """
        Executes the full pipeline asynchronously.
        """
        logger.info("pipeline_started", ticker=self.ticker, framework=self.framework)
        
        session = get_db_session(self.db_url)
        try:
            # 1. Scrape Data
            try:
                df = await self.scraper.fetch_historical_data(self.ticker, "2023-01-01", "2023-12-31")
            except Exception as e:
                logger.warning("scrape_failed_retrying_mock", error=str(e), provider=self.scraper.provider)
                self.scraper = MarketDataScraper(api_key="DEMO_KEY", provider="mock")
                df = await self.scraper.fetch_historical_data(self.ticker, "2023-01-01", "2023-12-31")

            logger.info("data_scraped", rows=len(df))
            
            # 2. Persist to DB using optimized bulk insert
            # We insert raw data before feature engineering to maintain audit trail
            market_data_records = df.to_dict('records')
            for record in market_data_records:
                record["ticker"] = self.ticker
                record["timestamp"] = int(record["timestamp"])
            
            if market_data_records:
                session.bulk_insert_mappings(MarketTick, market_data_records)
                session.commit()
            logger.info("data_persisted", count=len(market_data_records))
            
            # 3. Feature Engineering
            df_featured = self.generate_features(df)
            logger.info("features_generated", columns=list(df_featured.columns))

            # 4. Data Drift Detection
            historical_prices = df_featured["close"].values
            # Compare recent data against historical average
            current_prices = historical_prices[-100:] if len(historical_prices) > 100 else historical_prices
            
            statistic, p_value = calculate_ks_test(historical_prices, current_prices)
            psi_score = calculate_psi(historical_prices, current_prices)
            
            DATA_DRIFT_SCORE.set(psi_score)
            KS_TEST_SCORE.set(p_value)
            
            logger.info("data_drift_checked", 
                        ks_statistic=statistic, 
                        p_value=p_value,
                        psi_score=psi_score)
            
            # 5. Autonomous Training
            trainer = InstrumentedTrainer(study_name=self.study_name)
            
            # Define target and features
            # Target: 1 if close price increases in the next period
            df_featured["target"] = (df_featured["close"].shift(-1) > df_featured["close"]).astype(int)
            df_featured = df_featured.iloc[:-1] # Remove last row due to shift
            
            # Select features (exclude non-numeric and target)
            exclude = ["timestamp", "target", "ticker"]
            feature_names = [col for col in df_featured.columns if col not in exclude]
            X = df_featured[feature_names].values
            y = df_featured["target"].values
            
            dataset_metadata = {
                "ticker": self.ticker,
                "rows": str(len(df_featured)),
                "features": str(len(feature_names))
            }
            
            def objective(trial):
                if self.framework == "xgboost":
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                        "max_depth": trial.suggest_int("max_depth", 3, 10),
                        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
                        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                        "framework": "xgboost"
                    }
                elif self.framework == "sklearn":
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                        "max_depth": trial.suggest_int("max_depth", 5, 20),
                        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                        "framework": "sklearn"
                    }
                else: # pytorch
                    params = {
                        "epochs": trial.suggest_int("epochs", 20, 100),
                        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
                        "framework": "pytorch"
                    }
                
                return trainer.train_and_evaluate(
                    X, y, params, 
                    feature_names=feature_names, 
                    dataset_metadata=dataset_metadata
                )
            
            study = trainer.optimize(objective, n_trials=self.n_trials)
            best_accuracy = study.best_value
            
            # 6. Performance Drift Detection
            is_drifted = self.performance_monitor.detect_drift(best_accuracy)
            self.performance_monitor.add_metric(best_accuracy)
            
            logger.info("pipeline_completed", 
                        best_accuracy=best_accuracy, 
                        performance_drift=is_drifted,
                        best_params=study.best_params)
            
            push_metrics(job_name="autonomous_pipeline")
            return study

        except Exception as e:
            logger.critical("pipeline_failed", error=str(e))
            push_metrics(job_name="autonomous_pipeline")
            raise
        finally:
            session.close()

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
    asyncio.run(pipeline.run())