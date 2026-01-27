import numpy as np
import cutlass
from cutlass.utils.datatypes import is_numpy_tensor
if cutlass.use_rmm:
    import rmm
else:
    from cuda import cudart

class PoolMemoryManager:

    def __init__(self, init_pool_size: int, max_pool_size: int) -> None:
        self.pool = rmm.mr.PoolMemoryResource(rmm.mr.CudaMemoryResource(), initial_pool_size=init_pool_size, maximum_pool_size=max_pool_size)
        self.mr = rmm.mr.TrackingResourceAdaptor(self.pool)
        rmm.mr.set_current_device_resource(self.mr)

    def pool_size(self):
        return self.pool.pool_size()

class DevicePtrWrapper:
    """
    Wrapper around a pointer to device memory to provide a uniform interface with the RMM DeviceBuffer
    (at least in terms of the interface used by the CUTLASS Python interface)
    """

    def __init__(self, dev_ptr):
        self.dev_ptr = dev_ptr

    @property
    def ptr(self):
        return self.dev_ptr

def _todevice(host_data):
    """
    Helper for transferring host data to device memory
    """
    if cutlass.use_rmm:
        return rmm.DeviceBuffer.to_device(host_data.tobytes())
    else:
        nbytes = len(host_data.tobytes())
        dev_ptr_wrapper = device_mem_alloc(nbytes)
        err, = cudart.cudaMemcpy(dev_ptr_wrapper.ptr, host_data.__array_interface__['data'][0], nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        if err != cudart.cudaError_t.cudaSuccess:
            raise Exception(f'cudaMemcpy failed with error {err}')
        return dev_ptr_wrapper

def todevice(host_data, dtype=np.float32):
    """
    Pass the host_data to device memory
    """
    if isinstance(host_data, list):
        return _todevice(np.array(host_data, dtype=dtype))
    elif is_numpy_tensor(host_data):
        return _todevice(host_data)

def device_mem_alloc(size):
    if cutlass.use_rmm:
        return rmm.DeviceBuffer(size=size)
    else:
        err, ptr = cudart.cudaMalloc(size)
        if err != cudart.cudaError_t.cudaSuccess:
            raise Exception(f'cudaMalloc failed with error {err}')
        return DevicePtrWrapper(ptr)

def align_size(size, alignment=256):
    return (size + alignment - 1) // alignment * alignment

def create_memory_pool(init_pool_size=0, max_pool_size=2 ** 34):
    if cutlass.use_rmm:
        memory_pool = PoolMemoryManager(init_pool_size=init_pool_size, max_pool_size=max_pool_size)
        return memory_pool
    else:
        return None