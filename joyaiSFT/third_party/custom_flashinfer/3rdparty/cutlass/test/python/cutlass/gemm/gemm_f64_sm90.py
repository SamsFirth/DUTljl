"""
Low-level functionality tests for GEMM with F64 operands on SM90
"""
from functools import partial
import logging
import unittest
import cutlass
from cutlass.backend.utils.device import device_cc
from utils import LayoutCombination, add_test_gemm
cutlass.set_log_level(logging.WARNING)
cc = 90
dtype = cutlass.DataType.f64

@unittest.skipIf(device_cc() < cc, 'Device compute capability is insufficient for SM90 tests.')
@unittest.skipIf(cutlass.utils.datatypes.torch_type(dtype) is None, f'Version of torch installed does not contain a datatype match for {dtype}')
class GemmF64Sm90(unittest.TestCase):
    """
    Wrapper class to which tests will be added dynamically in __main__
    """
    pass
add_test_specialized = partial(add_test_gemm, cls=GemmF64Sm90, alignments=[1, 1, 1], cluster_shape=[1, 1, 1], element=dtype, element_output=dtype, element_accumulator=dtype, compilation_modes=['nvcc'])
add_test_specialized(opclass=cutlass.OpcodeClass.TensorOp, layouts=LayoutCombination.NNT, threadblock_shape=[128, 128, 32], stages=3)
add_test_specialized(opclass=cutlass.OpcodeClass.TensorOp, layouts=LayoutCombination.TNN, threadblock_shape=[128, 128, 32], stages=3)
add_test_specialized(opclass=cutlass.OpcodeClass.Simt, layouts=LayoutCombination.NNN, threadblock_shape=[128, 128, 8], stages=2)
add_test_specialized(opclass=cutlass.OpcodeClass.Simt, layouts=LayoutCombination.TTT, threadblock_shape=[64, 128, 8], stages=2)
if __name__ == '__main__':
    unittest.main()