import pandas as pd
import structlog

from .base import Feature, FeatureStore
from .features import LogReturnFeature

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

    def compute_features(self, data: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
        """
        Computes requested features using a dependency-aware engine.
        OPTIMIZED: Minimized copying and registration-based pre-processing.
        """
        # 1. Resolve required pre-processors (features with priority < 0)
        # 2. Sort features by priority to handle dependencies
        requested_features = []
        for name in feature_names:
            try:
                requested_features.append(self.get_feature(name))
            except KeyError:
                logger.warning("skipping_unregistered_feature", name=name)
        
        # Sort by priority (default is 0, pre-processors < 0)
        sorted_features = sorted(requested_features, key=lambda f: getattr(f, "priority", 0))
        
        # Work on a single copy
        df = data.copy()
        
        for feature in sorted_features:
            try:
                logger.debug("computing_feature", name=feature.name)
                result = feature.transform(df)
                if isinstance(result, pd.Series):
                    df[feature.name] = result
                elif isinstance(result, pd.DataFrame):
                    df = result # Feature transformed the entire frame
            except Exception as e:
                logger.error("feature_computation_failed", feature=feature.name, error=str(e))
                raise
                
        return df

# Global instance
feature_store = InMemoryFeatureStore()
