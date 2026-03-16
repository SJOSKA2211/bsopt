import os

import orjson
import structlog

logger = structlog.get_logger(__name__)


def generate_html_dashboard(summary_path: str, output_path: str) -> None:
    with open(summary_path, "rb") as f:
        summary = orjson.loads(f.read())

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ML Platform Dashboard</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f4f4f9;
                color: #333;
            }}
            .container {{
                max-width: 1000px;
                margin: auto;
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1, h2 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            .metric-card {{
                display: inline-block;
                width: 45%;
                background: #ecf0f1;
                margin: 10px;
                padding: 20px;
                border-radius: 8px;
                vertical-align: top;
            }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
            .visualizations {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-around;
                margin-top: 20px;
            }}
            .viz-item {{ margin: 10px; text-align: center; }}
            img {{
                max-width: 100%;
                height: auto;
                border-radius: 4px;
                box-shadow: 0 1px 5px rgba(0,0,0,0.2);
            }}
            .status {{ font-weight: bold; color: green; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>ML Performance Dashboard</h1>
            <p><strong>Generated At:</strong> {summary["timestamp"]}</p>
            <p><strong>Status:</strong> <span class="status">{summary["status"].upper()}</span></p>

            <h2>XGBoost Pricing Model</h2>
            <div class="metric-card">
                <div>R2 Score</div>
                <div class="metric-value">{summary.get("xgboost", {}).get("r2", "N/A")}</div>
            </div>
            <div class="metric-card">
                <div>Mean Squared Error</div>
                <div class="metric-value">{summary.get("xgboost", {}).get("mse", "N/A")}</div>
            </div>

            <h2>Neural Network Classifier</h2>
            <div class="metric-card">
                <div>Accuracy</div>
                <div class="metric-value">{summary.get("neural_network", {}).get("accuracy", "N/A")}</div>
            </div>
            <div class="metric-card">
                <div>Precision</div>
                <div class="metric-value">{summary.get("neural_network", {}).get("precision", "N/A")}</div>
            </div>

            <h2>Visualizations</h2>
            <div class="visualizations">
                <div class="viz-item">
                    <h3>Regression Performance</h3>
                    <img src="xgb_regression_plot.png" alt="XGBoost Plot">
                </div>
                <div class="viz-item">
                    <h3>Classification Matrix</h3>
                    <img src="nn_confusion_matrix.png" alt="NN Matrix">
                </div>
            </div>

            <p style="margin-top: 40px; font-size: 0.8em; color: #777;">
                Tracking URI: {summary["mlflow_tracking_uri"]}
            </p>
        </div>
    </body>
    </html>
    """

    with open(output_path, "w") as f:
        f.write(html_content)
    logger.info("dashboard_generated", path=output_path)


if __name__ == "__main__":
    summary_file = "results/comprehensive_run/run_summary.json"
    dashboard_file = "results/comprehensive_run/dashboard.html"
    if os.path.exists(summary_file):
        generate_html_dashboard(summary_file, dashboard_file)
    else:
        logger.error("dashboard_summary_not_found", path=summary_file)
