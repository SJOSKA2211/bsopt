import os
from typing import Any

import dask
import structlog
import xgboost as xgb
import xgboost.dask
from dask.distributed import Client, LocalCluster

from src.config import settings

logger = structlog.get_logger(__name__)

# Configure Dask for high-performance communication
dask.config.set(
    {
        "distributed.comm.compression": "lz4",
        "distributed.worker.memory.target": 0.6,
        "distributed.worker.memory.spill": 0.7,
        "distributed.worker.memory.pause": 0.8,
        "distributed.worker.memory.terminate": 0.95,
    }
)


def get_dask_client(address: str | None = None) -> tuple[Client, bool]:
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


def train_xgboost_distributed(X, y, params: dict[str, Any], dask_address: str | None = None):
    """
    Train XGBoost model using Dask for distributed execution.
    """
    client, is_local_cluster = get_dask_client(dask_address)  # Get client and flag
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
        if is_local_cluster:  # Only close if it was a locally created cluster
            client.close()


def sync_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """
    Synchronizes metrics across all workers in a distributed training group.
    Uses torch.distributed if initialized.
    """
    import torch

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return metrics

    world_size = torch.distributed.get_world_size()
    synced_metrics = {}

    for k, v in metrics.items():
        # Ensure we use the correct device for the rank
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        t = torch.tensor([v], device=device)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        synced_metrics[k] = t.item() / world_size

    return synced_metrics


def check_ray_cluster() -> dict[str, Any]:
    """
    God-Mode: Comprehensive Ray Cluster health and resource check.
    """
    import ray
    if not ray.is_initialized():
        return {"status": "not_initialized"}
    
    nodes = ray.nodes()
    resources = ray.cluster_resources()
    available = ray.available_resources()
    
    health_report = {
        "status": "healthy" if len(nodes) > 0 else "degraded",
        "node_count": len(nodes),
        "total_cpus": resources.get("CPU", 0),
        "available_cpus": available.get("CPU", 0),
        "total_gpus": resources.get("GPU", 0),
        "total_memory_gb": resources.get("memory", 0) / (1024**3),
        "object_store_gb": resources.get("object_store_memory", 0) / (1024**3),
    }
    
    logger.info("ray_cluster_health_report", **health_report)
    return health_report
