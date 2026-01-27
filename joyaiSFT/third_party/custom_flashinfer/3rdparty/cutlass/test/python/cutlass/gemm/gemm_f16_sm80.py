"""
Low-level functionality tests for GEMM with F16 operands on SM80
"""
from functools import partial
import logging
import unittest
import cutlass
from cutlass.backend.utils.device import device_cc
from utils import LayoutCombination, add_test_gemm
cutlass.set_log_level(logging.WARNING)
cc = 80
dtype = cutlass.DataType.f16

@unittest.skipIf(device_cc() < cc, 'Device compute capability is insufficient for SM80 tests.')
@unittest.skipIf(cutlass.utils.datatypes.torch_type(dtype) is None, f'Version of torch installed does not contain a datatype match for {dtype}')
class GemmF16Sm80(unittest.TestCase):
    """
    Wrapper class to which tests will be added dynamically in __main__
    """
    pass

@unittest.skipIf(device_cc() < cc, 'Device compute capability is insufficient for SM80 tests.')
@unittest.skipIf(cutlass.utils.datatypes.torch_type(dtype) is None, f'Version of torch installed does not contain a datatype match for {dtype}')
class GemmF16Sm80StreamK(unittest.TestCase):
    """
    Wrapper class to which tests will be added dynamically in __main__
    """
    pass
add_test_specialized = partial(add_test_gemm, element=dtype, cc=cc, cluster_shape=[1, 1, 1])
add_test_tensorop = partial(add_test_specialized, opclass=cutlass.OpcodeClass.TensorOp)
add_test_tensorop(cls=GemmF16Sm80, layouts=LayoutCombination.NNN, alignments=[8, 8, 8], element_output=cutlass.DataType.f16, element_C=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f32, threadblock_shape=[128, 128, 32], warp_count=[2, 2, 1], stages=3)
add_test_tensorop(cls=GemmF16Sm80, layouts=LayoutCombination.NNT, alignments=[8, 8, 8], element_output=cutlass.DataType.f16, element_C=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f32, threadblock_shape=[128, 128, 32], warp_count=[2, 2, 1], stages=3)
add_test_tensorop(cls=GemmF16Sm80, layouts=LayoutCombination.NTN, alignments=[8, 8, 8], element_output=cutlass.DataType.f16, element_C=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f32, threadblock_shape=[128, 128, 32], warp_count=[2, 2, 1], stages=3)
add_test_tensorop(cls=GemmF16Sm80, layouts=LayoutCombination.NTT, alignments=[8, 8, 8], element_output=cutlass.DataType.f16, element_C=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f32, threadblock_shape=[128, 128, 32], warp_count=[2, 2, 1], stages=3)
add_test_tensorop(cls=GemmF16Sm80, layouts=LayoutCombination.TNN, alignments=[8, 8, 8], element_output=cutlass.DataType.f16, element_C=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f32, threadblock_shape=[128, 128, 32], warp_count=[2, 2, 1], stages=3)
add_test_tensorop(cls=GemmF16Sm80, layouts=LayoutCombination.TNT, alignments=[8, 8, 8], element_output=cutlass.DataType.f16, element_C=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f32, threadblock_shape=[128, 128, 32], warp_count=[2, 2, 1], stages=3)
add_test_tensorop(cls=GemmF16Sm80, layouts=LayoutCombination.TTN, alignments=[8, 8, 8], element_output=cutlass.DataType.f16, element_C=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f32, threadblock_shape=[128, 128, 32], warp_count=[2, 2, 1], stages=3)
add_test_tensorop(cls=GemmF16Sm80, layouts=LayoutCombination.TTT, alignments=[8, 8, 8], element_output=cutlass.DataType.f16, element_C=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f32, threadblock_shape=[128, 128, 32], warp_count=[2, 2, 1], stages=3)
add_test_tensorop(cls=GemmF16Sm80, layouts=LayoutCombination.TNT, alignments=[8, 8, 8], element_output=cutlass.DataType.f16, element_C=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f32, threadblock_shape=[64, 128, 32], warp_count=[1, 2, 1], stages=3)
add_test_tensorop(cls=GemmF16Sm80, layouts=LayoutCombination.TNT, alignments=[8, 8, 8], element_output=cutlass.DataType.f16, element_C=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f32, threadblock_shape=[128, 64, 32], warp_count=[2, 1, 1], stages=3)
add_test_tensorop(cls=GemmF16Sm80, layouts=LayoutCombination.TNT, alignments=[8, 8, 8], element_output=cutlass.DataType.f16, element_C=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f32, threadblock_shape=[64, 64, 64], warp_count=[1, 1, 1], stages=3)
add_test_tensorop(cls=GemmF16Sm80, layouts=LayoutCombination.TNT, alignments=[4, 4, 8], element_output=cutlass.DataType.f16, element_C=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f32, threadblock_shape=[128, 128, 32], warp_count=[2, 2, 1], stages=3)
add_test_tensorop(cls=GemmF16Sm80, layouts=LayoutCombination.TNT, alignments=[4, 4, 8], element_output=cutlass.DataType.f16, element_C=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f16, threadblock_shape=[128, 128, 32], warp_count=[2, 2, 1], stages=3)
add_test_tensorop(cls=GemmF16Sm80, layouts=LayoutCombination.TNT, alignments=[8, 8, 8], element_output=cutlass.DataType.f16, element_C=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f16, threadblock_shape=[128, 128, 32], warp_count=[2, 2, 1], stages=3)
add_test_tensorop(cls=GemmF16Sm80, layouts=LayoutCombination.TNT, alignments=[8, 8, 8], element_output=cutlass.DataType.f16, element_C=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f32, threadblock_shape=[64, 64, 64], warp_count=[1, 1, 1], stages=5)
add_test_tensorop(cls=GemmF16Sm80, layouts=LayoutCombination.TNT, alignments=[2, 2, 2], element_output=cutlass.DataType.f16, element_C=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f16, threadblock_shape=[128, 128, 32], warp_count=[2, 2, 1], stages=3)
add_test_simt = partial(add_test_specialized, opclass=cutlass.OpcodeClass.Simt)
add_test_simt(cls=GemmF16Sm80, layouts=LayoutCombination.NNN, alignments=[1, 1, 1], element_output=cutlass.DataType.f16, element_C=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f32, threadblock_shape=[128, 128, 8], warp_count=[2, 2, 1], stages=2)
add_test_simt(cls=GemmF16Sm80, layouts=LayoutCombination.TNN, alignments=[1, 1, 1], element_output=cutlass.DataType.f16, element_C=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f32, threadblock_shape=[64, 128, 8], warp_count=[1, 2, 1], stages=2)
add_test_simt(cls=GemmF16Sm80, layouts=LayoutCombination.NTN, alignments=[1, 1, 1], element_output=cutlass.DataType.f16, element_C=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f32, threadblock_shape=[128, 64, 8], warp_count=[2, 1, 1], stages=2)
add_test_simt(cls=GemmF16Sm80, layouts=LayoutCombination.TTN, alignments=[1, 1, 1], element_output=cutlass.DataType.f16, element_C=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f32, threadblock_shape=[64, 64, 8], warp_count=[1, 1, 1], stages=2)
add_test_simt(cls=GemmF16Sm80, layouts=LayoutCombination.NNT, alignments=[1, 1, 1], element_output=cutlass.DataType.f16, element_C=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f16, threadblock_shape=[128, 128, 8], warp_count=[2, 2, 1], stages=2)
add_test_streamk = partial(add_test_specialized, opclass=cutlass.OpcodeClass.TensorOp, swizzle=cutlass.swizzle.ThreadblockSwizzleStreamK)
add_test_streamk(cls=GemmF16Sm80StreamK, layouts=LayoutCombination.NNN, alignments=[8, 8, 8], element_output=cutlass.DataType.f16, element_C=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f32, threadblock_shape=[128, 128, 32], warp_count=[2, 2, 1], stages=3)
add_test_streamk(cls=GemmF16Sm80StreamK, layouts=LayoutCombination.TNT, alignments=[8, 8, 8], element_output=cutlass.DataType.f16, element_C=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f32, threadblock_shape=[64, 64, 64], warp_count=[1, 1, 1], stages=5)
if __name__ == '__main__':
    unittest.main()