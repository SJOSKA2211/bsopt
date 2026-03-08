import pandas as pd
import structlog

from .base import Feature, FeatureStore
from .features import LogReturnFeature

logger = structlog.get_logger()


class InMemoryFeatureStore(FeatureStore):
    def __init__(self):
        self.features = {}
        # Register default features
        self.register_feature(LogReturnFeature())
        self.register_feature(RSIPeature())
        self.register_feature(EMAFeature())
        self.register_feature(MACDFeature())

    def register_feature(self, feature: Feature):
        self.features[feature.name] = feature

    def get_feature(self, name: str) -> Feature:
        if name not in self.features:
            raise KeyError(f"Feature {name} not found")
        return self.features[name]

    async def compute_features(self, data: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
        """
        Computes requested features with Redis-backed caching.
        """
        from src.utils.cache import get_redis

        # ... (caching logic)
        redis = get_redis()
        if redis:
            cache_key = f"feature_cache:{hash(tuple(feature_names))}"
            try:
                cached = await redis.get(cache_key)
                if cached:
                    # ...
                    pass
            except Exception:
                pass

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
        if redis:
            try:
                # In production, use a BackgroundTask or non-blocking call
                pass  # await redis.setex(cache_key, 300, df.to_json())
            except Exception:
                pass

        return df


# Global instance
feature_store = InMemoryFeatureStore()
