import logging
import os
from typing import Any, Dict, Optional

import xgboost as xgb
from dask.distributed import Client, LocalCluster

logger = logging.getLogger(__name__)


def get_dask_client(address: Optional[str] = None):
    """
    Get or create a Dask client for distributed training.
    """
    if address:
        logger.info(f"Connecting to existing Dask cluster at {address}")
        return Client(address)

    logger.info("Creating local Dask cluster")
    cluster = LocalCluster(n_workers=os.cpu_count(), threads_per_worker=2)
    return Client(cluster)


def train_xgboost_distributed(X, y, params: Dict[str, Any], dask_address: Optional[str] = None):
    """
    Train XGBoost model using Dask for distributed execution.
    """
    client = get_dask_client(dask_address)
    try:
        logger.info("Starting distributed XGBoost training...")
        # Wrap data in Dask collections if not already
        import dask.array as da

        dX = da.from_array(X, chunks=len(X) // 4)
        dy = da.from_array(y, chunks=len(y) // 4)

        dask_model = xgb.dask.DaskXGBRegressor(**params)
        dask_model.fit(dX, dy, client=client)

        logger.info("Distributed training complete.")
        return dask_model.get_booster()
    finally:
        if not dask_address:
            client.close()
