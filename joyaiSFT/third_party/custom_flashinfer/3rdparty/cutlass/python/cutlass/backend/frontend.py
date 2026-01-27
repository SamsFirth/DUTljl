from cuda import cuda
import numpy as np
from cutlass.backend.memory_manager import device_mem_alloc, todevice
from cutlass.utils.datatypes import is_cupy_tensor, is_numpy_tensor, is_torch_tensor

class NumpyFrontend:
    """
    Frontend node for numpy
    """

    @staticmethod
    def argument(np_tensor: 'np.ndarray', is_output: 'bool') -> cuda.CUdeviceptr:
        """Convert the input numpy tensor to CUDA device pointer

        :param np_tensor: input numpy nd array
        :param is_output: whether the tensor is output

        :return: CUDA device pointer
        """
        if is_output:
            return device_mem_alloc(np_tensor.size * np_tensor.itemsize)
        else:
            return todevice(np_tensor)

class TorchFrontend:
    """
    Frontend node for torch
    """

    @staticmethod
    def argument(torch_tensor: 'torch.Tensor') -> cuda.CUdeviceptr:
        """Convert the input torch tensor to CUDA device pointer

        :param torch_tensor: input torch tensor
        :param is_output: whether the tensor is output

        :return: CUDA device pointer
        """
        if not torch_tensor.is_cuda:
            torch_tensor = torch_tensor.to('cuda')
        return cuda.CUdeviceptr(torch_tensor.data_ptr())

class CupyFrontend:
    """
    Frontend node for cupy
    """

    @staticmethod
    def argument(cupy_ndarray: 'cp.ndarray'):
        return cuda.CUdeviceptr(int(cupy_ndarray.data.ptr))

class TensorFrontend:
    """
    Universal Frontend for client-provide tensors
    """

    @staticmethod
    def argument(tensor, is_output=False):
        if is_numpy_tensor(tensor):
            return NumpyFrontend.argument(tensor, is_output)
        elif is_torch_tensor(tensor):
            return TorchFrontend.argument(tensor)
        elif is_cupy_tensor(tensor):
            return CupyFrontend.argument(tensor)
        else:
            raise NotImplementedError('Unknown Tensor Type')