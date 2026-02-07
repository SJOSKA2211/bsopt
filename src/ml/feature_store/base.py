from abc import ABC, abstractmethod

import pandas as pd
import structlog

logger = structlog.get_logger()


class Feature(ABC):
    """
    Abstract base class for a single feature definition.
    """

    name: str
    description: str
    version: str = "1.0.0"

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.Series:
        """
        Apply the transformation logic to generate the feature.
        """
        pass


class FeatureStore(ABC):
    """
    Abstract base class for the centralized Feature Store.
    """

    @abstractmethod
    def get_feature(self, name: str) -> Feature:
        """Retrieve a feature definition by name."""
        pass

    @abstractmethod
    def compute_features(
        self, data: pd.DataFrame, feature_names: list[str]
    ) -> pd.DataFrame:
        """Compute a set of features for the given dataset."""
        pass
