from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pandas as pd

from .base import Feature

class LogReturnFeature(Feature):
    name: str = "log_return"
    description: str = "Logarithmic return of the closing price"

    def transform(self, data: pd.DataFrame) -> pd.Series:
        if "close" not in data.columns:
            raise ValueError("Data missing 'close' column for log_return calculation")
        return cast(pd.Series, np.log(data["close"] / data["close"].shift(1)).fillna(0))

class NumbaIndicatorFeature(Feature):
    """
    High-Performance: High-performance wrapper for JIT-compiled indicators.
    """

    def __init__(self, name: str, func: Callable, **kwargs: float | int | str) -> None:
        self.name = name
        self.func = func
        self.kwargs = kwargs

    def transform(self, data: pd.DataFrame) -> pd.Series:
        if "close" not in data.columns:
            raise ValueError(f"Data missing 'close' for {self.name}")

        closes = data["close"].values.astype(np.float64)
        result = self.func(closes, **self.kwargs)

        # Handle tuple results (e.g., BBands)
        if isinstance(result, tuple):
            return pd.Series(result[1], index=data.index)  # Default to mid band
        return pd.Series(result, index=data.index)

class RSIPeature(NumbaIndicatorFeature):
    def __init__(self, length: int = 14) -> None:
        from src.ml.indicators import get_rsi

        super().__init__(f"RSI_{length}", get_rsi, length=length)

class EMAFeature(NumbaIndicatorFeature):
    def __init__(self, span: int = 20) -> None:
        from src.ml.indicators import get_ema

        super().__init__(f"EMA_{span}", get_ema, span=span)

class MACDFeature(NumbaIndicatorFeature):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        from src.ml.indicators import get_macd

        super().__init__("MACD", get_macd, fast=fast, slow=slow, signal=signal)
