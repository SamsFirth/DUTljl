"""
Low-level functionality tests for GEMM with mixed operands on SM80
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
class GemmMixedSm80(unittest.TestCase):
    """
    Wrapper class to which tests will be added dynamically in __main__
    """
    pass
add_test_mixed = partial(add_test_gemm, cls=GemmMixedSm80, element=dtype, cc=cc, cluster_shape=[1, 1, 1], opclass=cutlass.OpcodeClass.TensorOp, threadblock_shape=[128, 128, 64], warp_count=[2, 2, 1], stages=3, element_accumulator=cutlass.DataType.f32)
add_test_mixed(element_A=cutlass.DataType.s8, alignments=[16, 8, 8], layouts=LayoutCombination.TNT)
add_test_mixed(element_A=cutlass.DataType.s8, alignments=[16, 8, 8], layouts=LayoutCombination.TNN)
add_test_mixed(element_B=cutlass.DataType.s8, alignments=[8, 16, 8], layouts=LayoutCombination.TNT)
add_test_mixed(element_B=cutlass.DataType.s8, alignments=[8, 16, 8], layouts=LayoutCombination.TNN)
if __name__ == '__main__':
    unittest.main()