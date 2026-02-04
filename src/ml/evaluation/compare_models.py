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
        champion_rmse = champion_run.data.metrics.get("rmse", float('inf'))
        logger.info("champion_metrics_loaded", model=model_name, rmse=champion_rmse)
    except Exception as e:
        logger.warning("champion_not_found", error=str(e))
        champion_rmse = float('inf')

    # 2. Get Challenger Metrics
    challenger_run = client.get_run(challenger_run_id)
    challenger_rmse = challenger_run.data.metrics.get("rmse", float('inf'))
    logger.info("challenger_metrics_loaded", rmse=challenger_rmse)

    # 3. Decision Logic (Advanced Financial Decisioning)
    # Don't just check RMSE, Morty! Check if the model actually makes money.
    
    challenger_sharpe = challenger_run.data.metrics.get("sharpe_ratio", 0.0)
    champion_sharpe = champion_run.data.metrics.get("sharpe_ratio", 0.0)
    
    challenger_drawdown = challenger_run.data.metrics.get("max_drawdown", -1.0)
    champion_drawdown = champion_run.data.metrics.get("max_drawdown", -1.0)

    logger.info("performance_comparison", 
                challenger_rmse=challenger_rmse, champion_rmse=champion_rmse,
                challenger_sharpe=challenger_sharpe, champion_sharpe=champion_sharpe)

    # Decision: Improved RMSE AND (Better Sharpe OR Better Drawdown)
    if challenger_rmse < champion_rmse and (challenger_sharpe > champion_sharpe or challenger_drawdown > champion_drawdown):
        logger.info("promotion_recommended", challenger=challenger_run_id, champion=champion_stage)
        return True
    else:
        reason = "Performance did not meet financial requirements"
        if challenger_rmse >= champion_rmse: reason = "RMSE not improved"
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
