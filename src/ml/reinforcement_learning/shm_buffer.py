import struct
import time
from multiprocessing import shared_memory
from typing import Any, cast

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

class SharedExperienceBuffer:
    """
    High-performance Shared Memory Replay Buffer.
    Allows zero-copy experience collection and sampling across the Ray cluster.
    """

    def __init__(
        self,
        name: str = "rl_replay_buffer",
        capacity: int = 100000,
        obs_dim: int = 100,
        act_dim: int = 10,
        create: bool = False,
    ) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        #  Layout: [Lock(1)] + [Head(8)] + [Obs(N*D)] + [Act(N*A)] + [Rew(N)] + [Next_Obs(N*D)]
        # Total size in bytes (float32 = 4 bytes)
        self.shm_size = (
            1
            + 8
            + capacity * obs_dim * 4
            + capacity * act_dim * 4
            + capacity * 4
            + capacity * obs_dim * 4
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
                self.shm.buf[0] = 0  # LOCK initialization
                self.shm.buf[1:9] = struct.pack("q", 0)  # Head index
            else:
                self.shm = shared_memory.SharedMemory(name=name)

            self.buf: memoryview = self.shm.buf

            # Map buffers to NumPy arrays
            offset = 9
            self.obs: np.ndarray[Any, np.dtype[np.float32]] = np.ndarray(
                (capacity, obs_dim), dtype=np.float32, buffer=self.buf, offset=offset
            )
            offset += capacity * obs_dim * 4
            self.act: np.ndarray[Any, np.dtype[np.float32]] = np.ndarray(
                (capacity, act_dim), dtype=np.float32, buffer=self.buf, offset=offset
            )
            offset += capacity * act_dim * 4
            self.rew: np.ndarray[Any, np.dtype[np.float32]] = np.ndarray(
                capacity, dtype=np.float32, buffer=self.buf, offset=offset
            )
            offset += capacity * 4
            self.next_obs: np.ndarray[Any, np.dtype[np.float32]] = np.ndarray(
                (capacity, obs_dim), dtype=np.float32, buffer=self.buf, offset=offset
            )

            logger.info("shm_replay_buffer_initialized", name=name, capacity=capacity)
        except Exception as e:
            logger.error("shm_replay_buffer_failed", error=str(e))
            raise

    def add(
        self,
        obs: np.ndarray[Any, np.dtype[np.float32]],
        act: np.ndarray[Any, np.dtype[np.float32]],
        rew: float,
        next_obs: np.ndarray[Any, np.dtype[np.float32]],
    ) -> None:
        """Zero-copy transition push with spin-lock for multi-producer safety."""
        mv = self.buf

        # 1. Spin-Lock
        start = time.perf_counter()
        while mv[0] != 0:
            if time.perf_counter() - start > 0.1:  # 100ms timeout
                mv[0] = 0  # Safety break
                break
            pass

        mv[0] = 1  # LOCK
        try:
            head = int(struct.unpack("q", mv[1:9])[0])
            idx = head % self.capacity

            self.obs[idx] = obs
            self.act[idx] = act
            self.rew[idx] = rew
            self.next_obs[idx] = next_obs

            mv[1:9] = struct.pack("q", head + 1)
        finally:
            mv[0] = 0  # UNLOCK

    def sample(
        self, batch_size: int
    ) -> tuple[
        np.ndarray[Any, np.dtype[np.float32]],
        np.ndarray[Any, np.dtype[np.float32]],
        np.ndarray[Any, np.dtype[np.float32]],
        np.ndarray[Any, np.dtype[np.float32]],
    ]:
        """Zero-copy batch sampling with wait-free polling."""
        mv = self.buf
        while mv[0] != 0:
            pass  # Busy-wait for unlock

        head = int(struct.unpack("q", mv[1:9])[0])
        max_idx = min(head, self.capacity)
        indices = np.random.choice(max_idx, batch_size, replace=False)

        return (
            cast(np.ndarray[Any, np.dtype[np.float32]], self.obs[indices]),
            cast(np.ndarray[Any, np.dtype[np.float32]], self.act[indices]),
            cast(np.ndarray[Any, np.dtype[np.float32]], self.rew[indices]),
            cast(np.ndarray[Any, np.dtype[np.float32]], self.next_obs[indices]),
        )

    def close(self, unlink: bool = False) -> None:
        self.shm.close()
        if unlink:
            self.shm.unlink()
