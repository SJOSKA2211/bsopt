import argparse
import asyncio

import structlog

from services.ml.pipeline import MLPipeline

logger = structlog.get_logger(__name__)


async def main():
    parser = argparse.ArgumentParser(description="BS-OPT ML Pipeline Orchestrator")
    parser.add_argument("--model-type", type=str, default="xgboost", help="Model framework to use")
    parser.add_argument(
        "--promote-to-production",
        action="store_true",
        help="Promote the model to production if successful",
    )
    parser.add_argument("--ticker", type=str, default="AAPL", help="Ticker symbol for training")

    args = parser.parse_args()

    config = {
        "ticker": args.ticker,
        "framework": args.model_type,
        "n_trials": 20,
        "study_name": f"{args.model_type}_retrain_{args.ticker}",
        "promote": args.promote_to_production,
    }

    logger.info("orchestrator_start", config=config)

    try:
        pipeline = MLPipeline(config)
        await pipeline.run()
        logger.info("orchestrator_complete", status="success")
    except Exception as e:
        logger.error("orchestrator_failed", error=str(e))
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
