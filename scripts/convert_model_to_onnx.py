import os

import mlflow
import mlflow.xgboost
import onnxmltools
import structlog
from onnxconverter_common.data_types import FloatTensorType

logger = structlog.get_logger(__name__)


def convert_xgb_to_onnx(model_path: str, output_path: str, num_features: int = 9):
    """
    Convert a production XGBoost model from MLflow to ONNX format.
    """
    logger.info("loading_xgb_model", path=model_path)
    model = mlflow.xgboost.load_model(model_path)

    # Define input shape
    initial_type = [("input", FloatTensorType([None, num_features]))]

    logger.info("converting_to_onnx")
    onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    onnxmltools.utils.save_model(onnx_model, output_path)
    logger.info("onnx_model_saved", path=output_path)


if __name__ == "__main__":
    # Example usage
    convert_xgb_to_onnx(
        "models:/XGBoostOptionPricer/Production", "models/latest_xgb_pricing.onnx"
    )
