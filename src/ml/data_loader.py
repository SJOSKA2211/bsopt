from typing import Dict, Any
import structlog

logger = structlog.get_logger()

class DataNormalizer:
    """
    Normalization Layer to ensure ML models receive consistent data.
    Handles synthetic bar generation for sparse sources like scrapers.
    """
    @staticmethod
    def normalize_incoming(raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts disparate data sources into a unified OHLCV format.
        """
        # 1. Synthetic OHLC Generation
        # If scraper returns only 'price', fill OHLC with that price
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
