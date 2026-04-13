from typing import Any, Callable, Dict
import numpy as np
from src.ml.training.base import TrainingConfig, TrainingResult

class TrainingRegistry:
    _registry: Dict[str, Callable[..., Any]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(func: Callable[..., Any]):
            cls._registry[name.lower()] = func
            return func
        return decorator

    @classmethod
    def get_trainer(cls, name: str) -> Callable[..., Any]:
        trainer = cls._registry.get(name.lower())
        if not trainer:
            raise ValueError(f"No trainer registered for framework: {name}")
        return trainer

training_registry = TrainingRegistry()
