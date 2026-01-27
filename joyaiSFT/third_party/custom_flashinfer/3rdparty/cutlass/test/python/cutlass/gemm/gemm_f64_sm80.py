"""
Low-level functionality tests for GEMM with F64 operands on SM80
"""
from functools import partial
import logging
import unittest
import cutlass
from cutlass.backend.utils.device import device_cc
from utils import LayoutCombination, add_test_gemm
cutlass.set_log_level(logging.WARNING)
cc = 80
dtype = cutlass.DataType.f64

@unittest.skipIf(device_cc() < cc, 'Device compute capability is insufficient for SM80 tests.')
@unittest.skipIf(cutlass.utils.datatypes.torch_type(dtype) is None, f'Version of torch installed does not contain a datatype match for {dtype}')
class GemmF64Sm80(unittest.TestCase):
    """
    Wrapper class to which tests will be added dynamically in __main__
    """
    pass

@unittest.skipIf(device_cc() < cc, 'Device compute capability is insufficient for SM80 tests.')
@unittest.skipIf(cutlass.utils.datatypes.torch_type(dtype) is None, f'Version of torch installed does not contain a datatype match for {dtype}')
class GemmF64Sm80StreamK(unittest.TestCase):
    """
    Wrapper class to which tests will be added dynamically in __main__
    """
    pass
add_test_specialized = partial(add_test_gemm, element=dtype, cc=cc, cluster_shape=[1, 1, 1])
add_test_tensorop = partial(add_test_specialized, opclass=cutlass.OpcodeClass.TensorOp)
add_test_tensorop(cls=GemmF64Sm80, layouts=LayoutCombination.NNN, alignments=[1, 1, 1], element_output=dtype, element_C=dtype, element_accumulator=dtype, threadblock_shape=[128, 128, 16], warp_count=[4, 2, 1], stages=3)
add_test_tensorop(cls=GemmF64Sm80, layouts=LayoutCombination.NTN, alignments=[1, 1, 1], element_output=dtype, element_C=dtype, element_accumulator=dtype, threadblock_shape=[64, 64, 16], warp_count=[2, 2, 1], stages=4)
add_test_tensorop(cls=GemmF64Sm80, layouts=LayoutCombination.TTN, alignments=[1, 1, 1], element_output=dtype, element_C=dtype, element_accumulator=dtype, threadblock_shape=[32, 32, 16], warp_count=[2, 1, 1], stages=5)
add_test_simt = partial(add_test_specialized, opclass=cutlass.OpcodeClass.Simt)
add_test_simt(cls=GemmF64Sm80, layouts=LayoutCombination.NNN, alignments=[1, 1, 1], element_output=dtype, element_C=dtype, element_accumulator=dtype, threadblock_shape=[128, 128, 8], warp_count=[2, 2, 1], stages=2)
add_test_simt(cls=GemmF64Sm80, layouts=LayoutCombination.TNN, alignments=[1, 1, 1], element_output=dtype, element_C=dtype, element_accumulator=dtype, threadblock_shape=[64, 128, 8], warp_count=[1, 2, 1], stages=2)
add_test_simt(cls=GemmF64Sm80, layouts=LayoutCombination.NTN, alignments=[1, 1, 1], element_output=dtype, element_C=dtype, element_accumulator=dtype, threadblock_shape=[128, 64, 8], warp_count=[2, 1, 1], stages=2)
add_test_simt(cls=GemmF64Sm80, layouts=LayoutCombination.TTN, alignments=[1, 1, 1], element_output=dtype, element_C=dtype, element_accumulator=dtype, threadblock_shape=[64, 64, 8], warp_count=[1, 1, 1], stages=2)
add_test_simt(cls=GemmF64Sm80, layouts=LayoutCombination.NNT, alignments=[1, 1, 1], element_output=dtype, element_C=dtype, element_accumulator=dtype, threadblock_shape=[128, 128, 8], warp_count=[2, 2, 1], stages=2)
add_test_streamk = partial(add_test_specialized, opclass=cutlass.OpcodeClass.TensorOp, swizzle=cutlass.swizzle.ThreadblockSwizzleStreamK)
add_test_streamk(cls=GemmF64Sm80StreamK, layouts=LayoutCombination.NTT, alignments=[1, 1, 1], element_output=dtype, element_C=dtype, element_accumulator=dtype, threadblock_shape=[128, 128, 16], warp_count=[4, 2, 1], stages=3)
if __name__ == '__main__':
    unittest.main()