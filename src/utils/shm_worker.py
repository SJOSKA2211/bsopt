from collections.abc import Callable
from multiprocessing import shared_memory
from typing import Any

import numpy as np


class SHMContextManager:
    """
    Context manager for handling SharedMemory lifecycles in workers.
    Automatically closes shared memory blocks on exit.
    """

    def __init__(self, *shm_names: str):
        self.shm_names = shm_names
        self.shm_objects = []

    def __enter__(self) -> list[shared_memory.SharedMemory]:
        try:
            for name in self.shm_names:
                if isinstance(name, dict):  # Handle dict for named outputs
                    for n in name.values():
                        shm = shared_memory.SharedMemory(name=n)
                        self.shm_objects.append(shm)
                else:
                    shm = shared_memory.SharedMemory(name=name)
                    self.shm_objects.append(shm)
            return self.shm_objects
        except Exception:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        for shm in self.shm_objects:
            try:
                shm.close()
            except Exception:
                pass
        self.shm_objects.clear()


def _generic_shm_worker(
    shm_name_inputs: str,
    input_shape: tuple,
    shm_name_output: str,
    kernel_func: Callable,
    input_dtype: Any = np.float64,
    output_dtype: Any = np.float64,
    *args,
):
    """
    Generic worker for typed Input -> Kernel -> Output flow via SHM.
    """
    try:
        with SHMContextManager(shm_name_inputs, shm_name_output) as shms:
            shm_in, shm_out = shms[0], shms[1]

            inputs = np.ndarray(input_shape, dtype=input_dtype, buffer=shm_in.buf)
            outputs = np.ndarray((input_shape[0],), dtype=output_dtype, buffer=shm_out.buf)

            kernel_func(inputs, outputs, *args)

        return True
    except Exception as e:
        return str(e)
