import pandas as pd
import structlog
import msgspec

from .base import Feature, FeatureStore
from .features import EMAFeature, LogReturnFeature, MACDFeature, RSIPeature

logger = structlog.get_logger()


class InMemoryFeatureStore(FeatureStore):
    def __init__(self) -> None:
        self.features: dict[str, Feature] = {}
        # Register default features
        self.register_feature(LogReturnFeature())
        self.register_feature(RSIPeature())
        self.register_feature(EMAFeature())
        self.register_feature(MACDFeature())

    def register_feature(self, feature: Feature) -> None:
        self.features[feature.name] = feature

    def get_feature(self, name: str) -> Feature:
        if name not in self.features:
            raise KeyError(f"Feature {name} not found")
        return self.features[name]

    async def compute_features(self, data: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
        """
        Computes requested features with Redis-backed caching (msgspec).
        """
        from src.shared.utils.cache import get_redis

        redis = get_redis()
        cache_key = ""
        if redis:
            # Hash includes data shape and sorted feature names for stability
            data_hash = hash(data.values.tobytes())
            cache_key = f"feature_cache:{data_hash}:{hash(tuple(sorted(feature_names)))}"
            try:
                cached = await redis.get(cache_key)
                if cached:
                    logger.info("feature_cache_hit", key=cache_key)
                    cached_data = msgspec.json.decode(cached)
                    return pd.DataFrame(cached_data)
            except Exception as e:
                logger.warning("feature_cache_read_failed", error=str(e))

        # 3. Resolve and sort features
        requested_features = []
        for name in feature_names:
            try:
                requested_features.append(self.get_feature(name))
            except KeyError:
                logger.warning("skipping_unregistered_feature", name=name)

        sorted_features = sorted(requested_features, key=lambda f: getattr(f, "priority", 0))

        # 4. Compute (Single copy, then inplace where possible)
        df = data.copy()
        for feature in sorted_features:
            try:
                logger.debug("computing_feature", name=feature.name)
                result = feature.transform(df)
                if isinstance(result, pd.Series):
                    df[feature.name] = result
                elif isinstance(result, pd.DataFrame):
                    df = result
            except Exception as e:
                logger.error("feature_computation_failed", feature=feature.name, error=str(e))
                raise

        # 5. Background cache fill
        if redis and cache_key:
            try:
                import asyncio

                asyncio.create_task(self._background_cache_fill(df, cache_key))
            except Exception:
                pass

        return df

    async def _background_cache_fill(self, df: pd.DataFrame, key: str) -> None:
        """Persistent cache population without blocking execution."""
        try:
            from src.shared.utils.cache import get_redis

            redis = get_redis()
            if redis:
                await redis.setex(key, 300, df.to_json())
                logger.info("feature_cache_populated", key=key, rows=len(df))
        except Exception as e:
            logger.error("feature_cache_failed", key=key, error=str(e))


# Global instance
feature_store = InMemoryFeatureStore()
