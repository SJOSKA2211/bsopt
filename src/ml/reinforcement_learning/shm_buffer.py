import struct
from multiprocessing import shared_memory

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

class SharedExperienceBuffer:
    """
    🚀 SINGULARITY: High-performance Shared Memory Replay Buffer.
    Allows zero-copy experience collection and sampling across the Ray cluster.
    """
    def __init__(self, 
                 name: str = "rl_replay_buffer", 
                 capacity: int = 100000, 
                 obs_dim: int = 100, 
                 act_dim: int = 10,
                 create: bool = False):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # 🚀 Layout: [Head(8)] + [Obs(N*D)] + [Act(N*A)] + [Rew(N)] + [Next_Obs(N*D)]
        # Total size in bytes (float32 = 4 bytes)
        self.shm_size = (
            8 + 
            capacity * obs_dim * 4 + 
            capacity * act_dim * 4 + 
            capacity * 4 + 
            capacity * obs_dim * 4
        )
        
        try:
            if create:
                try:
                    existing = shared_memory.SharedMemory(name=name)
                    existing.close()
                    existing.unlink()
                except FileNotFoundError:
                    pass
                self.shm = shared_memory.SharedMemory(name=name, create=True, size=self.shm_size)
                self.shm.buf[:8] = struct.pack("q", 0) # Head index
            else:
                self.shm = shared_memory.SharedMemory(name=name)
            
            self.buf = self.shm.buf
            
            # Map buffers to NumPy arrays
            offset = 8
            self.obs = np.ndarray((capacity, obs_dim), dtype=np.float32, buffer=self.buf, offset=offset)
            offset += capacity * obs_dim * 4
            self.act = np.ndarray((capacity, act_dim), dtype=np.float32, buffer=self.buf, offset=offset)
            offset += capacity * act_dim * 4
            self.rew = np.ndarray(capacity, dtype=np.float32, buffer=self.buf, offset=offset)
            offset += capacity * 4
            self.next_obs = np.ndarray((capacity, obs_dim), dtype=np.float32, buffer=self.buf, offset=offset)
            
            logger.info("shm_replay_buffer_initialized", name=name, capacity=capacity)
        except Exception as e:
            logger.error("shm_replay_buffer_failed", error=str(e))
            raise

    def add(self, obs, act, rew, next_obs):
        """🚀 SINGULARITY: Zero-copy transition push."""
        head = struct.unpack("q", self.buf[:8])[0]
        idx = head % self.capacity
        
        self.obs[idx] = obs
        self.act[idx] = act
        self.rew[idx] = rew
        self.next_obs[idx] = next_obs
        
        self.buf[:8] = struct.pack("q", head + 1)

    def sample(self, batch_size: int):
        """🚀 SOTA: Zero-copy batch sampling."""
        head = struct.unpack("q", self.buf[:8])[0]
        max_idx = min(head, self.capacity)
        indices = np.random.choice(max_idx, batch_size, replace=False)
        
        return (
            self.obs[indices],
            self.act[indices],
            self.rew[indices],
            self.next_obs[indices]
        )

    def close(self, unlink: bool = False):
        self.shm.close()
        if unlink:
            self.shm.unlink()
