"""
Low-level functionality tests for Conv2d opreations on SM80
"""
import logging
import unittest
import cutlass
from cutlass.backend.utils.device import device_cc
from conv2d_test_utils import *
cutlass.set_log_level(logging.WARNING)
cc = 80

@unittest.skipIf(device_cc() < cc, 'Device compute capability is invalid for SM80 tests.')
class Conv2dSm80(unittest.TestCase):
    """
    Wrapper class to which tests will be added dynamically in __main__
    """
    pass
conv_problems = get_conv_problems()
for conv_kind in ['fprop', 'wgrad', 'dgrad']:
    add_test(Conv2dSm80, cc, conv_kind, conv_problems, cutlass.DataType.f16, cutlass.DataType.f32, cutlass.DataType.f16, opclass='simt', threadblock_shape=[128, 128, 8], warp_count=[4, 2, 1], stages=2, instruction_shape=[1, 1, 1])
    add_test(Conv2dSm80, cc, conv_kind, conv_problems, cutlass.DataType.f16, cutlass.DataType.f32, cutlass.DataType.f16, opclass='tensor_op', threadblock_shape=[128, 128, 64], warp_count=[2, 2, 1], stages=3, instruction_shape=[16, 8, 16])
    add_test(Conv2dSm80, cc, conv_kind, conv_problems, cutlass.DataType.f16, cutlass.DataType.f16, cutlass.DataType.f16, opclass='tensor_op', threadblock_shape=[128, 128, 64], warp_count=[2, 2, 1], stages=3, instruction_shape=[16, 8, 16], iterator_algorithm='analytic')
    add_test(Conv2dSm80, cc, conv_kind, conv_problems, cutlass.DataType.f16, cutlass.DataType.f32, cutlass.DataType.f32, opclass='tensor_op', threadblock_shape=[128, 128, 64], warp_count=[2, 2, 1], stages=3, instruction_shape=[16, 8, 16])
    add_test(Conv2dSm80, cc, conv_kind, conv_problems, cutlass.DataType.f16, cutlass.DataType.f32, cutlass.DataType.f16, opclass='tensor_op', threadblock_shape=[128, 64, 32], warp_count=[2, 2, 1], stages=3, instruction_shape=[16, 8, 8])
    add_test(Conv2dSm80, cc, conv_kind, conv_problems, cutlass.DataType.f32, cutlass.DataType.f32, cutlass.DataType.f32, opclass='simt', threadblock_shape=[128, 128, 8], warp_count=[4, 2, 1], stages=4, instruction_shape=[1, 1, 1])
    add_test(Conv2dSm80, cc, conv_kind, conv_problems, cutlass.DataType.f32, cutlass.DataType.f32, cutlass.DataType.f32, opclass='tensor_op', threadblock_shape=[128, 128, 16], warp_count=[2, 2, 1], stages=3, instruction_shape=[16, 8, 8])
    add_test(Conv2dSm80, cc, conv_kind, conv_problems, cutlass.DataType.f16, cutlass.DataType.f32, cutlass.DataType.f16, opclass='tensor_op', threadblock_shape=[128, 128, 64], warp_count=[2, 2, 1], stages=3, instruction_shape=[16, 8, 16], split_k_mode='serial', split_k_slices=2)
    add_test(Conv2dSm80, cc, conv_kind, conv_problems, cutlass.DataType.f16, cutlass.DataType.f32, cutlass.DataType.f16, opclass='tensor_op', threadblock_shape=[128, 128, 64], warp_count=[2, 2, 1], stages=3, instruction_shape=[16, 8, 16], split_k_mode='parallel', split_k_slices=5)
    add_test(Conv2dSm80, cc, conv_kind, conv_problems, cutlass.DataType.f16, cutlass.DataType.f32, cutlass.DataType.f16, opclass='tensor_op', threadblock_shape=[128, 64, 32], warp_count=[2, 2, 1], stages=3, instruction_shape=[16, 8, 8], swizzle=4)
for c, tb, stage, inst in zip([2, 1], [[128, 128, 64], [128, 128, 32]], [3, 2], [[16, 8, 16], [16, 8, 8]]):
    add_test(Conv2dSm80, cc, 'fprop', conv2d_few_channel_problemsizes(c), cutlass.DataType.f16, cutlass.DataType.f32, cutlass.DataType.f16, opclass='tensor_op', threadblock_shape=tb, warp_count=[2, 2, 1], stages=stage, instruction_shape=inst, iterator_algorithm='few_channels')
for c in [8, 4, 2]:
    add_test(Conv2dSm80, cc, 'fprop', conv2d_few_channel_problemsizes(c), cutlass.DataType.f16, cutlass.DataType.f32, cutlass.DataType.f16, opclass='tensor_op', threadblock_shape=[128, 128, 64], warp_count=[2, 2, 1], stages=3, instruction_shape=[16, 8, 16], iterator_algorithm='fixed_channels')
for activation in ['relu', 'leaky_relu']:
    for split_k_mode, split_k_slices in zip(['parallel', 'serial', 'parallel'], [1, 7, 5]):
        add_test(Conv2dSm80, cc, 'fprop', conv_problems, cutlass.DataType.f16, cutlass.DataType.f32, cutlass.DataType.f16, opclass='tensor_op', threadblock_shape=[128, 128, 64], warp_count=[2, 2, 1], stages=3, instruction_shape=[16, 8, 16], split_k_mode=split_k_mode, split_k_slices=split_k_slices, activation=activation)
if __name__ == '__main__':
    unittest.main()