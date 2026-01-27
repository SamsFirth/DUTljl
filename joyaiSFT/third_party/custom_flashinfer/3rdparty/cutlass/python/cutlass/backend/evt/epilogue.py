"""
Epilogue Visitor interface for compiling, and running visitor-based epilogue.
"""
import ctypes
from cuda import cuda
from cutlass_library import DataType
import numpy as np
from cutlass.backend.epilogue import EpilogueFunctorBase
import cutlass.backend.evt.backend
from cutlass.backend.frontend import TensorFrontend
from cutlass.utils.datatypes import is_numpy_tensor
from cutlass.backend.evt.passes.util import cc_map

class EpilogueFunctorVisitor(EpilogueFunctorBase):
    """
    Apply an epilogue functor described by the epilogue EVT

    :param cc: compute capability
    :param visitor_frontend: user-provide visitor frontend

    """

    def __init__(self, cc: int, visitor, element_compute=DataType.f32) -> None:
        self.emit_cls = getattr(cutlass.backend.evt.backend, f'Sm{cc_map[cc]}Emitter')
        self.visitor = visitor
        self.graph = visitor.dag_ir
        self.element_epilogue = element_compute
        self.element_output = self.graph.get_node_meta('D').underlying_impl.element
        epilogue_thread_type = self.visitor.epilogue_thread_type
        if cc == 90:
            self.arg_c_type = self.visitor.arg_c_type
            self.arg_d_type = self.visitor.arg_d_type
        output_names = self.visitor.return_names
        reduction_names = self.visitor.reduction_names
        if cc == 80:
            if hasattr(self.visitor, 'epilogue_stages'):
                self.epilogue_stages = self.visitor.epilogue_stages
                assert self.epilogue_stages <= 2, 'Only supports Stages <=2 in SM80 Epilogue'

        class _Arguments(ctypes.Structure):
            """
            Concepts:
            class _EpilogueArguments(ctypes.Structure):
                _fields_ = [
                    ("epilogue", _Arguments), <- this class
                    ("ptr_C", ctypes.c_void_p),
                    ("stride_C", StrideBatched_),
                    ("ptr_D", ctypes.c_void_p),
                    ("stride_D", StrideBatched_)
                ]
            """
            _fields_ = [('output_op', epilogue_thread_type)]

            def __init__(self, kwargs: dict) -> None:
                ptr_kwargs = {}
                for key in kwargs.keys():
                    is_output = key in output_names and key not in reduction_names
                    ptr_kwargs[key] = self.get_tensor_ptr(key, kwargs, is_output)
                self.output_op = epilogue_thread_type(ptr_kwargs)

            def get_tensor_ptr(self, tensor_name, kwargs, is_output=False):
                """
                Helper function for extracting device pointer
                """
                if cc == 90:
                    if tensor_name in ['C', 'D']:
                        return 0
                if tensor_name not in kwargs.keys():
                    raise ValueError(f'Tensor {tensor_name} is not provided.')
                tensor = kwargs[tensor_name]
                if isinstance(tensor, float):
                    return tensor
                buffer_or_ptr = TensorFrontend.argument(tensor, is_output)
                if is_numpy_tensor(tensor):
                    setattr(self, f'{tensor_name}_buffer', buffer_or_ptr)
                    setattr(self, f'{tensor_name}_host', tensor)
                    return int(buffer_or_ptr.ptr)
                else:
                    return int(buffer_or_ptr)

            def sync(self):
                """
                Synchronize the results from device to host
                """
                for name in output_names:
                    if hasattr(self, f'{name}_host'):
                        host_tensor = getattr(self, f'{name}_host')
                        tensor_ptr = getattr(self, f'{name}_buffer').ptr
                        err, = cuda.cuMemcpyDtoH(host_tensor, tensor_ptr, host_tensor.size * host_tensor.itemsize)
                        if err != cuda.CUresult.CUDA_SUCCESS:
                            raise RuntimeError('CUDA Error %s' % str(err))
        self.epilogue_type = _Arguments

    def emit(self, operation):
        """
        Emit the C++ code
        """
        emitter = self.emit_cls(operation, self.graph)
        return emitter.emit()

    def get_smem_size(self, tile_description):
        """
        Get the shared memory size in bytes
        """
        return self.visitor.get_smem_size(tile_description)