from typing import Dict, Any, Iterator, List
import structlog
import pandas as pd
import numpy as np
from src.ml.feature_store.store import feature_store

logger = structlog.get_logger()

class DataNormalizer:
    """
    Normalization Layer to ensure ML models receive consistent data.
    Handles synthetic bar generation via Feature Store.
    """
    @staticmethod
    def normalize_incoming(raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts disparate data sources into a unified OHLCV format.
        Now delegates to Feature Store logic for consistency if needed, 
        but for single dicts we might keep lightweight logic or wrap in DF.
        For simplicity, we keep the lightweight logic for single-item processing here
        to avoid overhead of creating 1-row DataFrames constantly, 
        but verify alignment with SyntheticOHLCFeature.
        """
        # 1. Synthetic OHLC Generation
        if 'open' not in raw_data and 'price' in raw_data:
            p = raw_data['price']
            normalized = {
                'timestamp': raw_data.get('timestamp'),
                'symbol': raw_data.get('symbol'),
                'open': p,
                'high': p,
                'low': p,
                'close': p,
                'volume': raw_data.get('volume', 0),
                'source_type': 'scraper_synthetic'
            }
            # Preserve other metadata
            for k, v in raw_data.items():
                if k not in normalized:
                    normalized[k] = v
            return normalized

        return raw_data

    @staticmethod
    def remove_outliers(data: Dict[str, Any], prev_price: float, threshold: float = 0.1) -> bool:
        """
        Simple outlier detection logic. Returns True if data is considered an outlier.
        """
        if not prev_price:
            return False
            
        current_price = data.get('close') or data.get('price')
        if not current_price:
            return False
            
        change = abs(current_price - prev_price) / prev_price
        if change > threshold:
            logger.warning("outlier_detected", symbol=data.get('symbol'), change=change)
            return True
            
        return False

class StreamingDataLoader:
    """
    Optimized data loader for large datasets using chunking and generators.
    Prevents OOM errors during training by avoiding full dataset loading.
    Delegates feature engineering to the Feature Store.
    """
    def __init__(self, file_path: str, chunk_size: int = 10000):
        self.file_path = file_path
        self.chunk_size = chunk_size

    def stream_batches(self) -> Iterator[pd.DataFrame]:
        """
        Yields normalized dataframes in chunks with features computed.
        """
        try:
            # Use Pandas chunksize for CSVs
            if self.file_path.endswith('.csv'):
                with pd.read_csv(self.file_path, chunksize=self.chunk_size) as reader:
                    for chunk in reader:
                        yield self._process_chunk(chunk)
            elif self.file_path.endswith('.parquet'):
                # Parquet doesn't support chunked reading natively in Pandas as easily as CSV
                # But PyArrow does. For simplicity, we'll assume memory mapping if possible
                # or just load it if it fits, otherwise use pyarrow directly.
                # Here we default to full load for parquet unless using pyarrow
                df = pd.read_parquet(self.file_path)
                # Simulate chunks
                for i in range(0, len(df), self.chunk_size):
                    yield self._process_chunk(df.iloc[i:i+self.chunk_size])
        except Exception as e:
            logger.error("streaming_load_failed", error=str(e), file=self.file_path)
            raise

    def _process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize and feature engineer a chunk of data using the Feature Store.
        """
        # Define standard features to compute
        required_features = ["log_return"]
        
        # Use centralized Feature Store
        # Synthetic OHLC is handled implicitly by the store's pre-processing logic for now
        # or we can explicitly ask for it if we defined it as such.
        # In our store implementation, we call SyntheticOHLCFeature transform automatically.
        
        try:
            processed_chunk = feature_store.compute_features(chunk, required_features)
            return processed_chunk
        except Exception as e:
            logger.error("chunk_processing_error", error=str(e))
            # Return original chunk or raise depending on policy. 
            # Raising is safer for training integrity.
            raise e