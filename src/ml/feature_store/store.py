import pandas as pd
import structlog

from .base import Feature, FeatureStore
from .features import LogReturnFeature, SyntheticOHLCFeature

logger = structlog.get_logger()


class InMemoryFeatureStore(FeatureStore):
    """
    In-memory implementation of the Feature Store.
    Registry is populated at startup.
    """

    def __init__(self):
        self._registry: dict[str, Feature] = {}
        self._register_defaults()

    def _register_defaults(self):
        self.register(LogReturnFeature())
        # SyntheticOHLC is a special pre-processor, handled slightly differently usually,
        # but for now we can treat it as a transformation if we are careful.
        # Ideally, we separate Preprocessors from Features, but keeping it simple for now.

    def register(self, feature: Feature):
        if feature.name in self._registry:
            logger.warning("overwriting_feature_definition", name=feature.name)
        self._registry[feature.name] = feature

    def get_feature(self, name: str) -> Feature:
        if name not in self._registry:
            raise KeyError(f"Feature '{name}' not found in registry")
        return self._registry[name]

    def compute_features(
        self, data: pd.DataFrame, feature_names: list[str]
    ) -> pd.DataFrame:
        """
        Computes requested features and appends them to the dataframe.
        """
        df = data.copy()

        # 1. Pre-processing (Implicit for now based on logic)
        # If we need synthetic OHLC, we do it first.
        # This logic should be more robust in a full implementation.
        synthetic_gen = SyntheticOHLCFeature()
        df = synthetic_gen.transform(df)

        for name in feature_names:
            if name == "synthetic_ohlc":
                continue  # Already handled

            try:
                feature = self.get_feature(name)
                # If the feature returns a Series, assign it
                result = feature.transform(df)
                if isinstance(result, pd.Series):
                    df[name] = result
                elif isinstance(result, pd.DataFrame):
                    # Merge or update
                    df = result
            except Exception as e:
                logger.error("feature_computation_failed", feature=name, error=str(e))
                # Depending on strictness, we might raise or skip
                raise e

        return df


# Global instance
feature_store = InMemoryFeatureStore()
