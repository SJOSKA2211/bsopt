import logging
import os
from typing import Any, Dict, Optional, Tuple

import xgboost as xgb
import xgboost.dask
import dask
from dask.distributed import Client, LocalCluster

# Configure Dask for high-performance communication
dask.config.set({
    "distributed.comm.compression": "lz4",
    "distributed.worker.memory.target": 0.6,
    "distributed.worker.memory.spill": 0.7,
    "distributed.worker.memory.pause": 0.8,
    "distributed.worker.memory.terminate": 0.95,
})

def get_dask_client(address: Optional[str] = None) -> Tuple[Client, bool]:
    """
    Get or create a Dask client for distributed training.
    Returns (client, is_local_cluster).
    """
    if address:
        logger.info(f"Connecting to existing Dask cluster at {address}")
        return Client(address), False

    logger.info("Creating local Dask cluster")
    n_workers = os.cpu_count() or 4
    threads_per_worker = settings.DASK_LOCAL_CLUSTER_THREADS_PER_WORKER
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
    return Client(cluster), True


def train_xgboost_distributed(X, y, params: Dict[str, Any], dask_address: Optional[str] = None):
    """
    Train XGBoost model using Dask for distributed execution.
    """
    client, is_local_cluster = get_dask_client(dask_address) # Get client and flag
    try:
        logger.info("Starting distributed XGBoost training...")
        # Wrap data in Dask collections if not already
        import dask.array as da

        chunk_size_fraction = settings.DASK_ARRAY_DEFAULT_CHUNKS_FRACTION
        dX = da.from_array(X, chunks=len(X) // chunk_size_fraction)
        dy = da.from_array(y, chunks=len(y) // chunk_size_fraction)

        dask_model = xgb.dask.DaskXGBRegressor(**params)
        dask_model.fit(dX, dy, client=client)

        logger.info("Distributed training complete.")
        return dask_model.get_booster()
    finally:
        if is_local_cluster: # Only close if it was a locally created cluster
            client.close()
