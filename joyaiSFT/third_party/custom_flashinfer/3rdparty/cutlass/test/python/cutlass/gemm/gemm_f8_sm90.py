"""
Low-level functionality tests for GEMM with S8 operands on SM90
"""
from functools import partial
import logging
import unittest
import cutlass
from cutlass.backend.utils.device import device_cc
from utils import LayoutCombination, add_test_gemm
cutlass.set_log_level(logging.WARNING)
cc = 90
dtype = cutlass.DataType.e4m3

@unittest.skipIf(device_cc() < cc, 'Device compute capability is insufficient for SM90 tests.')
@unittest.skipIf(cutlass.utils.datatypes.torch_type(dtype) is None, f'Version of torch installed does not contain a datatype match for {dtype}')
class GemmF8E4M3Sm90(unittest.TestCase):
    """
    Wrapper class to which tests will be added dynamically in __main__
    """
    pass
add_test_specialized = partial(add_test_gemm, cls=GemmF8E4M3Sm90, element=dtype, compilation_modes=['nvcc'])
add_test_tensorop = partial(add_test_specialized, opclass=cutlass.OpcodeClass.TensorOp)
add_test_tensorop(layouts=LayoutCombination.TNT, alignments=[16, 16, 16], element_output=cutlass.DataType.e4m3, element_accumulator=cutlass.DataType.f32, cluster_shape=[1, 1, 1], threadblock_shape=[128, 128, 128], stages=None)
add_test_tensorop(layouts=LayoutCombination.TNT, alignments=[16, 16, 16], element_output=cutlass.DataType.e4m3, element_accumulator=cutlass.DataType.f32, cluster_shape=[2, 2, 1], threadblock_shape=[128, 128, 128], stages=None)
add_test_tensorop(layouts=LayoutCombination.TNT, alignments=[16, 16, 16], element_output=cutlass.DataType.e4m3, element_accumulator=cutlass.DataType.f32, cluster_shape=[1, 4, 1], threadblock_shape=[128, 128, 128], stages=None)
add_test_tensorop(layouts=LayoutCombination.TNT, alignments=[16, 16, 16], element_output=cutlass.DataType.e4m3, element_accumulator=cutlass.DataType.f32, cluster_shape=[2, 1, 1], threadblock_shape=[128, 128, 128], stages=None, kernel_schedule=cutlass.KernelScheduleType.TmaWarpSpecializedPingpong, epilogue_schedule=cutlass.EpilogueScheduleType.TmaWarpSpecialized)
add_test_simt = partial(add_test_specialized, opclass=cutlass.OpcodeClass.Simt)
add_test_simt(layouts=LayoutCombination.TNN, alignments=[1, 1, 1], element_output=cutlass.DataType.e4m3, element_accumulator=cutlass.DataType.f32, cluster_shape=[1, 1, 1], threadblock_shape=[64, 32, 8], stages=2)
dtype = cutlass.DataType.e5m2

@unittest.skipIf(device_cc() < cc, 'Device compute capability is insufficient for SM90 tests.')
@unittest.skipIf(cutlass.utils.datatypes.torch_type(dtype) is None, f'Version of torch installed does not contain a datatype match for {dtype}')
class GemmF8E5M2Sm90(unittest.TestCase):
    """
    Wrapper class to which tests will be added dynamically in __main__
    """
    pass
add_test_specialized = partial(add_test_gemm, cls=GemmF8E5M2Sm90, element=dtype, compilation_modes=['nvcc'])
add_test_tensorop = partial(add_test_specialized, opclass=cutlass.OpcodeClass.TensorOp)
add_test_tensorop(layouts=LayoutCombination.TNN, alignments=[16, 16, 16], element_output=dtype, element_accumulator=cutlass.DataType.f32, cluster_shape=[1, 1, 1], threadblock_shape=[128, 128, 128], stages=3)
if __name__ == '__main__':
    unittest.main()