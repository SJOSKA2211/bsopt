import structlog
import torch

from src.ml.cross_sectional_pipeline import CrossSectionalPricingModel

logger = structlog.get_logger(__name__)


def export_to_onnx(model_path: str, output_path: str, input_dim: int):
    """
    Exports a trained PyTorch model to ONNX format for high-speed inference.
    Includes quantization for further performance gains.
    """
    logger.info("onnx_export_start", model_path=model_path)

    # 1. Load the model
    model = CrossSectionalPricingModel(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"), weights_only=True))
    model.eval()

    # 2. Create dummy input for tracing
    dummy_input = torch.randn(1, input_dim)

    # 3. Export to ONNX
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        logger.info("onnx_export_success", output_path=output_path)
    except Exception as e:
        logger.error("onnx_export_failed", error=str(e))
        raise


def quantize_onnx_model(onnx_path: str, quantized_path: str):
    """
    Apply INT8 quantization to the ONNX model for ultra-fast CPU inference.
    """
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(onnx_path, quantized_path, weight_type=QuantType.QUInt8)
        logger.info("onnx_quantization_success", quantized_path=quantized_path)
    except ImportError:
        logger.warning("onnx_quantization_skipped_missing_dependencies")
    except Exception as e:
        logger.error("onnx_quantization_failed", error=str(e))


if __name__ == "__main__":
    # Example usage
    INPUT_DIM = 12  # Matches features list in cross_sectional_pipeline.py
    export_to_onnx("models/latest_model.pt", "models/model.onnx", INPUT_DIM)
    quantize_onnx_model("models/model.onnx", "models/model_quantized.onnx")
