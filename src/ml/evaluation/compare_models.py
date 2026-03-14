"""
Champion vs. Challenger Model Evaluation script.
Compares metrics of a new model (challenger) against the current production model (champion).
"""

import argparse

import mlflow
import structlog

logger = structlog.get_logger()


def compare_models(model_name, challenger_run_id, champion_stage="Production"):
    client = mlflow.tracking.MlflowClient()

    # 1. Get Champion Metrics
    try:
        champion_version = client.get_latest_versions(model_name, stages=[champion_stage])[0]
        champion_run = client.get_run(champion_version.run_id)
        champion_rmse = champion_run.data.metrics.get("rmse", float("inf"))
        logger.info("champion_metrics_loaded", model=model_name, rmse=champion_rmse)
    except Exception as e:
        logger.warning("champion_not_found", error=str(e))
        champion_rmse = float("inf")

    # 2. Get Challenger Metrics
    challenger_run = client.get_run(challenger_run_id)
    challenger_rmse = challenger_run.data.metrics.get("rmse", float("inf"))
    logger.info("challenger_metrics_loaded", rmse=challenger_rmse)

    # 3. Decision Logic (Advanced Financial Decisioning)
    challenger_score = challenger_run.data.metrics.get("composite_score", 0.0)
    champion_score = champion_run.data.metrics.get("composite_score", 0.0) if champion_run else 0.0

    # Statistical Rigor: Requires at least 2% relative improvement to avoid noise
    EPSILON = 0.02
    improvement = (challenger_score - champion_score) / max(abs(champion_score), 1e-6)

    logger.info(
        "performance_comparison",
        challenger_score=round(challenger_score, 4),
        champion_score=round(champion_score, 4),
        improvement_pct=round(improvement * 100, 2),
    )

    # Decision: Substantial improvement in composite scorecard
    if improvement > EPSILON:
        logger.info(
            "promotion_recommended",
            challenger=challenger_run_id,
            improvement=f"{improvement * 100:.2f}%",
        )
        return True
    reason = "Improvement below statistical threshold (epsilon=2%)"
    if challenger_score <= champion_score:
        reason = "Challenger performance worse than champion"
    logger.info("promotion_rejected", reason=reason)
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--challenger", required=True)
    parser.add_argument("--champion", default="Production")
    args = parser.parse_args()

    should_promote = compare_models(args.model, args.challenger, args.champion)
    if should_promote:
        # In a real pipeline, this would trigger the promote.py script
        print("PROMOTE")
    else:
        print("REJECT")
