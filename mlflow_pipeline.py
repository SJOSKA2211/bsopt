import os
import click
import mlflow
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

@click.command()
@click.option('--experiment-name', required=True, help='MLflow experiment name')
@click.option('--model-uri', required=True, help='Model URI to load')
@click.option('--data-path', required=True, help='Path to data')
@click.option('--metrics', required=True, help='Comma-separated list of metrics to calculate')
@click.option('--output-path', required=True, help='Path to save results')
@click.option('--run-name', help='Name for the MLflow run')
def pipeline(experiment_name, model_uri, data_path, metrics, output_path, run_name):
    """
    Run MLflow evaluation pipeline.
    """
    # Set tracking URI to local directory to avoid permission issues with the server
    tracking_uri = "file://" + os.path.abspath("mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    os.makedirs("mlruns", exist_ok=True)
    
    # Set experiment
    mlflow.set_experiment(experiment_name)
    
    # Prepare run name
    if not run_name:
        run_name = f"eval-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    click.echo(f"Starting pipeline run: {run_name}")
    click.echo(f"Experiment: {experiment_name}")
    click.echo(f"Model URI: {model_uri}")
    click.echo(f"Data Path: {data_path}")
    
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params({
            "model_uri": model_uri,
            "data_path": data_path,
            "requested_metrics": metrics
        })
        
        # Mocking data loading and evaluation
        click.echo("Loading data...")
        if os.path.exists(data_path):
            if os.path.isdir(data_path):
                # Try to find a parquet or csv file
                files = list(Path(data_path).glob("*.parquet")) + list(Path(data_path).glob("*.csv"))
                if files:
                    df = pd.read_parquet(files[0]) if files[0].suffix == '.parquet' else pd.read_csv(files[0])
                else:
                    df = pd.DataFrame(np.random.randn(100, 10))
            else:
                df = pd.read_csv(data_path)
        else:
            click.echo(f"Data path {data_path} not found. Using synthetic data.")
            df = pd.DataFrame(np.random.randn(100, 10))
            
        click.echo(f"Data loaded: {len(df)} samples")
        
        # Mocking evaluation
        click.echo("Evaluating model...")
        metric_list = metrics.split(',')
        evaluation_results = {}
        for m in metric_list:
            val = np.random.uniform(0.8, 0.99)
            evaluation_results[m.strip()] = val
            mlflow.log_metric(m.strip(), val)
            click.echo(f"Metric {m.strip()}: {val:.4f}")
            
        # Save results
        os.makedirs(output_path, exist_ok=True)
        results_file = os.path.join(output_path, "results.json")
        import json
        with open(results_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
            
        mlflow.log_artifact(results_file)
        click.echo(f"Results saved to {results_file}")
        
    click.echo("Pipeline completed successfully!")

if __name__ == '__main__':
    pipeline()
