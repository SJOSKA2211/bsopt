import logging

import torch
import torch.nn.utils.prune as prune

logger = logging.getLogger(__name__)


def apply_pruning(model: torch.nn.Module, amount: float = 0.3):
    """
    Apply global unstructured pruning to the model.
    """
    logger.info(f"Applying pruning (amount={amount})...")
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, "weight"))

    if parameters_to_prune:
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
        # Make pruning permanent
        for module, name in parameters_to_prune:
            prune.remove(module, name)

    logger.info("Pruning complete.")
    return model


def apply_quantization(model: torch.nn.Module):
    """
    Apply post-training static quantization (simulated/prepared).
    """
    logger.info("Applying quantization...")
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    torch.quantization.prepare(model, inplace=True)

    # Note: In a real scenario, we need a calibration pass here before conversion.
    # For demonstration, we convert directly after preparation.
    torch.quantization.convert(model, inplace=True)

    logger.info("Quantization complete.")
    return model
