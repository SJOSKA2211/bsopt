import torch
import torch.distributed as dist
import os
import structlog
from typing import Optional

logger = structlog.get_logger(__name__)

class NCCLOrchestrator:
    """
    🚀 SINGULARITY: High-performance multi-GPU weight synchronization.
    Uses NCCL (NVIDIA Collective Communications Library) for zero-latency GPU-to-GPU sync.
    """
    @staticmethod
    def init_process_group(rank: int, world_size: int, master_addr: str = "localhost", master_port: str = "12355"):
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        
        # 🚀 SOTA: Force NCCL backend for peak GPU performance
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        
        # Set local device
        torch.cuda.set_device(rank)
        logger.info("nccl_process_group_initialized", rank=rank, world_size=world_size)

    @staticmethod
    def sync_weights(model: torch.nn.Module):
        """🚀 SOTA: Synchronize weights across all GPUs in the world."""
        if not dist.is_initialized():
            return
            
        for param in model.parameters():
            # All-reduce gradients or weights depending on the phase
            # For pure weight sync, we broadcast from rank 0
            dist.broadcast(param.data, src=0)
            
        logger.debug("weights_synchronized_via_nccl")

    @staticmethod
    def cleanup():
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("nccl_process_group_destroyed")
