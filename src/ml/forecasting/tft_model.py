from typing import Any

import lightning.pytorch as pl
import mlflow
import numpy as np
import pandas as pd
import structlog
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

logger = structlog.get_logger()


class PriceTFTModel:
    """
    Temporal Fusion Transformer (TFT) for SOTA Price Forecasting.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {
            "max_prediction_length": 5,
            "max_encoder_length": 24,
            "batch_size": 64,
            "max_epochs": 10,
        }
        self.training_dataset = None
        self.model = None
        self._quantized_model = None

    def prepare_data(self, data: pd.DataFrame) -> dict[str, Any]:
        """
        Prepares data for TFT training/validation.
        """
        data = data.copy()
        if "price" not in data.columns and "close" in data.columns:
            data["price"] = data["close"]

        if "symbol" not in data.columns or "price" not in data.columns:
            raise KeyError("Missing required columns 'symbol' or 'price'")

        if "time_idx" not in data.columns:
            data["time_idx"] = np.arange(len(data))

        max_prediction_length = self.config.get("output_chunk_length", 5)
        max_encoder_length = self.config.get("input_chunk_length", 24)

        training_cutoff = data["time_idx"].max() - max_prediction_length

        dataset = TimeSeriesDataSet(
            data[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="price",
            group_ids=["symbol"],
            min_encoder_length=max_encoder_length // 2,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=["symbol"],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=["price"],
            target_normalizer=GroupNormalizer(
                groups=["symbol"], transformation="softplus"
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        self.training_dataset = dataset

        validation = TimeSeriesDataSet.from_dataset(
            dataset, data, predict=True, stop_randomization=True
        )

        batch_size = self.config.get("batch_size", 64)
        # Optimized: Use more workers for data loading
        num_workers = min(os.cpu_count() or 4, 4)
        train_loader = dataset.to_dataloader(
            train=True, batch_size=batch_size, num_workers=num_workers
        )
        val_loader = validation.to_dataloader(
            train=False, batch_size=batch_size * 10, num_workers=num_workers
        )

        return {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "group_ids": ["symbol"],
        }

    def train(self, data: dict[str, Any]):
        """
        Train the TFT model with experiment tracking.
        """
        if not self.training_dataset:
            raise RuntimeError("Dataset not prepared. Call prepare_data first.")

        model = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            learning_rate=0.03,
            hidden_size=self.config.get("hidden_size", 16),
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=8,
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )

        trainer = self._init_trainer()

        with mlflow.start_run() as run:
            mlflow.log_params(self.config)
            trainer.fit(model, train_dataloaders=data["train_loader"])
            self.model = model

            # 🚀 ADVANCED: Post-Training Static Quantization (Simulated for CPU)
            self._quantize()

            # Log model artifact
            mlflow.pytorch.log_model(self.model, "tft_challenger")

        return run.info.run_id

    def _init_trainer(self) -> pl.Trainer:
        """Initializes the lightning trainer with optimal callbacks."""
        early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")

        return pl.Trainer(
            max_epochs=self.config.get("max_epochs", 10),
            accelerator="auto",
            devices="auto",
            enable_model_summary=True,
            gradient_clip_val=0.1,
            callbacks=[early_stop_callback, lr_monitor],
            log_every_n_steps=10,
        )

    def _quantize(self):
        """
        🚀 OPTIMIZATION: Quantize model to INT8 for faster inference.
        """
        logger.info("model_quantization_started")
        try:
            self._quantized_model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
            logger.info("model_quantization_complete")
        except Exception as e:
            logger.warning("quantization_failed", error=str(e))
            self._quantized_model = self.model

    async def train(self, data: pd.DataFrame, **kwargs):
        """
        Implement TFT model training.
        """
        # Security: Only allow specific parameters and enforce upper bounds
        allowed_params = {
            "max_epochs": (1, 100),
            "batch_size": (1, 512),
            "hidden_size": (4, 128),
            "dropout": (0.0, 0.5),
        }

        for key, (min_val, max_val) in allowed_params.items():
            if key in kwargs:
                val = kwargs[key]
                if not isinstance(val, (int, float)) or not (min_val <= val <= max_val):
                    logger.warning("invalid_param_ignoring", key=key, value=val)
                    kwargs.pop(key)

        # Update config with validated kwargs
        self.config.update(kwargs)

        processed = self.prepare_data(data)
        train_loader = processed["train_loader"]

        with mlflow.start_run():
            mlflow.log_params(self.config)

            # Initialize Model
            self.model = TemporalFusionTransformer.from_dataset(
                self.training_dataset,
                learning_rate=0.03,
                hidden_size=self.config.get("hidden_size", 16),
                attention_head_size=self.config.get("num_attention_heads", 1),
                dropout=self.config.get("dropout", 0.1),
                loss=QuantileLoss(),
            )

            trainer = self._init_trainer()
            trainer.fit(self.model, train_dataloaders=train_loader)

            return self.model

    def predict(self, data: pd.DataFrame, quantize: bool = False):
        """
        Inference with cached dynamic quantization for speed.
        """
        if not self.model:
            return None

        data = data.copy()
        if "price" not in data.columns and "close" in data.columns:
            data["price"] = data["close"]

        # 🚀 OPTIMIZATION: Cached Dynamic Quantization for CPU inference speedup
        if quantize and not torch.cuda.is_available():
            if self._quantized_model is None:
                logger.info("model_quantization_starting", type="dynamic_int8")
                self._quantized_model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
                logger.info("model_quantization_complete")
            model_to_use = self._quantized_model
        else:
            model_to_use = self.model

        return model_to_use.predict(data)

    def get_interpretability_report(self) -> dict[str, Any]:
        """
        Extracts real feature importance using TFT's built-in attention weights.
        """
        if not self.model or self.training_dataset is None:
            return {}

        # 🚀 SOTA: Fetch actual importance from the model interpretation
        # This requires passing a sample through or using the weights directly
        try:
            interpretation = self.model.interpret_output(
                self.model.predict(self.training_dataset, mode="raw", return_x=True)[0]
            )
            return {
                "encoder_variables": interpretation.get("encoder_variables", {}),
                "decoder_variables": interpretation.get("decoder_variables", {}),
                "static_variables": interpretation.get("static_variables", {}),
            }
        except Exception as e:
            logger.warning("interpretability_fetch_failed", error=str(e))
            return {
                "status": "error",
                "message": "Interpretation requires model evaluation on dataset",
            }


# Alias for backward compatibility
TFTModel = PriceTFTModel
