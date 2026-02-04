from multiprocessing import shared_memory
import numpy as np
from contextlib import contextmanager
from typing import Generator, List, Callable, Any, Tuple

class SHMContextManager:
    """
    Context manager for handling SharedMemory lifecycles in workers.
    Automatically closes shared memory blocks on exit.
    """
    def __init__(self, *shm_names: str):
        self.shm_names = shm_names
        self.shm_objects = []

    def __enter__(self) -> List[shared_memory.SharedMemory]:
        try:
            for name in self.shm_names:
                if isinstance(name, dict): # Handle dict for named outputs
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
            except:
                pass
        self.shm_objects.clear()

def _generic_shm_worker(
    shm_name_inputs: str, 
    input_shape: Tuple[int, int], 
    shm_name_output: str, 
    kernel_func: Callable, 
    *args
):
    """
    Generic worker for simple Input -> Kernel -> Output flow.
    """
    try:
        with SHMContextManager(shm_name_inputs, shm_name_output) as shms:
            # Assuming first is input, last is output for simple case
            # But SHMContextManager flattens the list.
            # We need to be careful about order.
            # Re-instantiating inside for clarity/safety might be better if order is complex, 
            # but we can rely on order of shm_names passed to init.
            
            shm_in = shms[0]
            shm_out = shms[1]
            
            inputs = np.ndarray(input_shape, dtype=np.float64, buffer=shm_in.buf)
            outputs = np.ndarray((input_shape[0],), dtype=np.float64, buffer=shm_out.buf)
            
            # Execute Kernel
            # We assume kernel takes (inputs, outputs, *args) or similar
            # But the existing kernels take columns. 
            # We need an adapter or the kernel needs to handle array-of-structs style.
            # Most existing kernels in quant_utils take component arrays.
            
            kernel_func(inputs, outputs, *args)
            
        return True
    except Exception as e:
        return str(e)
