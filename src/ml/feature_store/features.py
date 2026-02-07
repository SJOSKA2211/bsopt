import numpy as np
import pandas as pd

from .base import Feature


class LogReturnFeature(Feature):
    name = "log_return"
    description = "Logarithmic return of the closing price"

    def transform(self, data: pd.DataFrame) -> pd.Series:
        if "close" not in data.columns:
            raise ValueError("Data missing 'close' column for log_return calculation")
        return np.log(data["close"] / data["close"].shift(1)).fillna(0)


class SyntheticOHLCFeature(Feature):
    name = "synthetic_ohlc"
    description = "Fills missing OHLC values from price/close"

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Note: This returns a DataFrame as it modifies multiple columns.
        """
        df = data.copy()
        if "close" in df.columns and "open" not in df.columns:
            df["open"] = df["close"]
            df["high"] = df["close"]
            df["low"] = df["close"]
        elif "price" in df.columns and "open" not in df.columns:
            # Handle case where only 'price' exists
            p = df["price"]
            df["open"] = p
            df["high"] = p
            df["low"] = p
            df["close"] = p
        return df
