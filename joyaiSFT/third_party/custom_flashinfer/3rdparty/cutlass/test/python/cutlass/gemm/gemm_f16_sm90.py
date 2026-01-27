"""
Low-level functionality tests for GEMM with F16 operands on SM90
"""
from functools import partial
import logging
import unittest
import cutlass
from cutlass.backend.utils.device import device_cc
from utils import LayoutCombination, add_test_gemm
cutlass.set_log_level(logging.WARNING)
cc = 90
dtype = cutlass.DataType.f16

@unittest.skipIf(device_cc() < cc, 'Device compute capability is insufficient for SM90 tests.')
@unittest.skipIf(cutlass.utils.datatypes.torch_type(dtype) is None, f'Version of torch installed does not contain a datatype match for {dtype}')
class GemmF16Sm90(unittest.TestCase):
    """
    Wrapper class to which tests will be added dynamically in __main__
    """
    pass
add_test_specialized = partial(add_test_gemm, cls=GemmF16Sm90, element=dtype, warp_count=None, compilation_modes=['nvcc'])
add_test_tensorop = partial(add_test_specialized, opclass=cutlass.OpcodeClass.TensorOp)
add_test_unit_cluster = partial(add_test_tensorop, cluster_shape=[1, 1, 1])
add_test_unit_cluster(layouts=LayoutCombination.NNN, alignments=[8, 8, 8], element_output=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f32, threadblock_shape=[128, 128, 32], stages=3)
add_test_unit_cluster(layouts=LayoutCombination.NNT, alignments=[8, 8, 8], element_output=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f32, threadblock_shape=[128, 128, 32], stages=None)
add_test_unit_cluster(layouts=LayoutCombination.NTN, alignments=[8, 8, 8], element_output=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f32, threadblock_shape=[128, 128, 32], stages=None)
add_test_unit_cluster(layouts=LayoutCombination.NTT, alignments=[8, 8, 8], element_output=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f32, threadblock_shape=[128, 128, 32], stages=None)
add_test_unit_cluster(layouts=LayoutCombination.TNN, alignments=[8, 8, 8], element_output=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f32, threadblock_shape=[128, 128, 32], stages=None)
add_test_unit_cluster(layouts=LayoutCombination.TNT, alignments=[4, 4, 8], element_output=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f32, threadblock_shape=[128, 128, 32], stages=None)
add_test_unit_cluster(layouts=LayoutCombination.TNT, alignments=[4, 4, 8], element_output=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f16, threadblock_shape=[128, 128, 32], stages=None)
add_test_unit_cluster(layouts=LayoutCombination.TNT, alignments=[8, 8, 8], element_output=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f16, threadblock_shape=[128, 128, 32], stages=None)
add_test_unit_cluster(layouts=LayoutCombination.TNT, alignments=[8, 8, 8], element_output=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f32, threadblock_shape=[64, 64, 64], stages=5)
add_test_unit_cluster(layouts=LayoutCombination.TNT, alignments=[2, 2, 2], element_output=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f16, threadblock_shape=[128, 128, 32], stages=None)
add_test_cluster_shape = partial(add_test_tensorop, threadblock_shape=[64, 128, 64], stages=None)
add_test_cluster_shape(layouts=LayoutCombination.TTN, alignments=[8, 8, 8], element_output=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f16, cluster_shape=[2, 2, 1])
add_test_cluster_shape(layouts=LayoutCombination.TNN, alignments=[8, 8, 4], element_output=cutlass.DataType.f32, element_accumulator=cutlass.DataType.f32, cluster_shape=[2, 2, 1])
add_test_cluster_shape(layouts=LayoutCombination.NTN, alignments=[8, 8, 4], element_output=cutlass.DataType.f32, element_accumulator=cutlass.DataType.f32, cluster_shape=[2, 2, 1])
add_test_cluster_shape(layouts=LayoutCombination.NNN, alignments=[8, 8, 4], element_output=cutlass.DataType.f32, element_accumulator=cutlass.DataType.f32, cluster_shape=[2, 2, 1])
add_test_cluster_shape(layouts=LayoutCombination.TTN, alignments=[8, 8, 4], element_output=cutlass.DataType.f32, element_accumulator=cutlass.DataType.f32, cluster_shape=[1, 4, 1])
add_test_cluster_shape(layouts=LayoutCombination.TTN, alignments=[8, 8, 4], element_output=cutlass.DataType.f32, element_accumulator=cutlass.DataType.f32, cluster_shape=[2, 4, 1])
add_test_cluster_shape(layouts=LayoutCombination.TTN, alignments=[8, 8, 4], element_output=cutlass.DataType.f32, element_accumulator=cutlass.DataType.f32, cluster_shape=[4, 1, 1])
add_test_cluster_shape(layouts=LayoutCombination.TTN, alignments=[8, 8, 4], element_output=cutlass.DataType.f32, element_accumulator=cutlass.DataType.f32, cluster_shape=[4, 2, 1])
add_test_schedule = partial(add_test_specialized, layouts=LayoutCombination.TTN, alignments=[8, 8, 4], element_output=cutlass.DataType.f32, element_accumulator=cutlass.DataType.f32, opclass=cutlass.OpcodeClass.TensorOp, threadblock_shape=[128, 128, 64], stages=None)
add_test_schedule(cluster_shape=[1, 1, 1], kernel_schedule=cutlass.KernelScheduleType.TmaWarpSpecializedPingpong, epilogue_schedule=cutlass.EpilogueScheduleType.TmaWarpSpecialized)
add_test_schedule(cluster_shape=[1, 1, 1], kernel_schedule=cutlass.KernelScheduleType.TmaWarpSpecializedCooperative, epilogue_schedule=cutlass.EpilogueScheduleType.TmaWarpSpecializedCooperative)
add_test_schedule(cluster_shape=[2, 1, 1], kernel_schedule=cutlass.KernelScheduleType.TmaWarpSpecializedPingpong, epilogue_schedule=cutlass.EpilogueScheduleType.TmaWarpSpecialized)
add_test_schedule(cluster_shape=[2, 1, 1], kernel_schedule=cutlass.KernelScheduleType.TmaWarpSpecializedCooperative, epilogue_schedule=cutlass.EpilogueScheduleType.TmaWarpSpecializedCooperative)
add_test_simt = partial(add_test_specialized, opclass=cutlass.OpcodeClass.Simt, alignments=[1, 1, 1], cluster_shape=[1, 1, 1], stages=2)
add_test_simt(layouts=LayoutCombination.NNN, element_output=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f32, threadblock_shape=[128, 128, 8])
add_test_simt(layouts=LayoutCombination.TNN, element_output=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f32, threadblock_shape=[64, 128, 8])
add_test_simt(layouts=LayoutCombination.NTN, element_output=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f32, threadblock_shape=[128, 64, 8])
add_test_simt(layouts=LayoutCombination.TTN, element_output=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f32, threadblock_shape=[64, 64, 8])
add_test_simt(layouts=LayoutCombination.NNT, element_output=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f16, threadblock_shape=[128, 128, 8])
add_test_cluster_shape(layouts=LayoutCombination.NNT, alignments=[8, 8, 8], element_output=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f32, threadblock_shape=[128, 128, 32], stages=None, cluster_shape=[2, 1, 1], element_C=cutlass.DataType.void)
if __name__ == '__main__':
    unittest.main()