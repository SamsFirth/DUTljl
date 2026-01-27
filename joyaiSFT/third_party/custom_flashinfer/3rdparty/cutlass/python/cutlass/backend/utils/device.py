"""
Utility functions for interacting with the device
"""
from cuda import cuda, cudart
import cutlass
from cutlass.utils.datatypes import is_cupy_tensor, is_numpy_tensor, is_torch_tensor

def check_cuda_errors(result: list):
    """
    Checks whether `result` contains a CUDA error raises the error as an exception, if so. Otherwise,
    returns the result contained in the remaining fields of `result`.

    :param result: the results of the `cudart` method, consisting of an error code and any method results
    :type result: list

    :return: non-error-code results from the `results` parameter
    """
    err = result[0]
    if err.value:
        raise RuntimeError('CUDA error: {}'.format(cudart.cudaGetErrorName(err)))
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]

def device_cc(device: int=-1) -> int:
    """
    Returns the compute capability of the device with ID `device`.

    :param device: ID of the device to query
    :type device: int

    :return: compute capability of the queried device (e.g., 80 for SM80)
    :rtype: int
    """
    if device == -1:
        device = cutlass.device_id()
    deviceProp = check_cuda_errors(cudart.cudaGetDeviceProperties(device))
    major = str(deviceProp.major)
    minor = str(deviceProp.minor)
    return int(major + minor)

def device_sm_count(device: int=-1):
    if device == -1:
        device = cutlass.device_id()
    err, device_sm_count = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device)
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise Exception(f'Failed to retireve SM count. cuDeviceGetAttribute() failed with error: {cuda.cuGetErrorString(err)[1]}')
    return device_sm_count

def to_device_ptr(tensor) -> cuda.CUdeviceptr:
    """
    Converts a tensor to a CUdeviceptr

    :param tensor: tensor to convert
    :type tensor: np.ndarray | torch.Tensor | cp.ndarray | int

    :return: device pointer
    :rtype: cuda.CUdeviceptr
    """
    if is_numpy_tensor(tensor):
        ptr = cuda.CUdeviceptr(tensor.__array_interface__['data'][0])
    elif is_torch_tensor(tensor):
        ptr = cuda.CUdeviceptr(tensor.data_ptr())
    elif is_cupy_tensor(tensor):
        ptr = cuda.CUdeviceptr(int(tensor.data.ptr))
    elif isinstance(tensor, cuda.CUdeviceptr):
        ptr = tensor
    elif isinstance(tensor, int):
        ptr = cuda.CUdeviceptr(tensor)
    else:
        raise NotImplementedError(tensor)
    return ptr