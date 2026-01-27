"""
Utilities for enumerating CUTLASS library SM90 kernels
"""
import argparse
import enum
from itertools import product
import math
import logging
import os.path
import shutil
import sys
import copy
from typing import Any, Optional, Sequence, Tuple
try:
    import builtins
    if hasattr(builtins, 'CUTLASS_IGNORE_PACKAGE') and CUTLASS_IGNORE_PACKAGE == True:
        raise ImportError('Disabling attempt to import cutlass_library')
    from cutlass_library.library import *
except ImportError:
    from library import *

def CudaToolkitVersionSatisfies(semantic_ver_string, major, minor, patch=0):
    cuda_version = [11, 0, 132]
    if semantic_ver_string != '':
        for i, x in enumerate([int(x) for x in semantic_ver_string.split('.')]):
            if i < len(cuda_version):
                cuda_version[i] = x
            else:
                cuda_version.append(x)
    return cuda_version >= [major, minor, patch]

def get_wgmma_level_from_global_level(global_level: int):
    return global_level % 10

def get_mma_level_from_global_level(global_level: int):
    return global_level // 10 % 10

def get_cluster_level_from_global_level(global_level: int):
    return global_level // 100 % 10

def get_pruning_level_from_global_level(global_level: int):
    return global_level // 1000 % 10
try:
    from .sm90_shapes import SM90_MMA_MULTIPLIERS, SM90_CLUSTER_SIZES, SM90_WGMMA_SHAPES_TF32_DENSE, SM90_WGMMA_SHAPES_FP16_BF16_DENSE, SM90_WGMMA_SHAPES_FP8_DENSE, SM90_WGMMA_SHAPES_INT8_DENSE
except:
    from sm90_shapes import SM90_MMA_MULTIPLIERS, SM90_CLUSTER_SIZES, SM90_WGMMA_SHAPES_TF32_DENSE, SM90_WGMMA_SHAPES_FP16_BF16_DENSE, SM90_WGMMA_SHAPES_FP8_DENSE, SM90_WGMMA_SHAPES_INT8_DENSE

def generate_tf32_math_instruction_shapes_sm90(level: int):
    assert isinstance(level, int) and level >= 0
    filtered_list_of_wgmma_shapes = [wgmma_shape for wgmma_shape, min_level in SM90_WGMMA_SHAPES_TF32_DENSE.items() if level >= min_level]
    return filtered_list_of_wgmma_shapes

def generate_fp16_bf16_math_instruction_shapes_sm90(level: int):
    assert isinstance(level, int) and level >= 0
    filtered_list_of_wgmma_shapes = [wgmma_shape for wgmma_shape, min_level in SM90_WGMMA_SHAPES_FP16_BF16_DENSE.items() if level >= min_level]
    return filtered_list_of_wgmma_shapes

def generate_fp8_math_instruction_shapes_sm90(level: int):
    assert isinstance(level, int) and level >= 0
    filtered_list_of_wgmma_shapes = [wgmma_shape for wgmma_shape, min_level in SM90_WGMMA_SHAPES_FP8_DENSE.items() if level >= min_level]
    return filtered_list_of_wgmma_shapes

def generate_int8_math_instruction_shapes_sm90(level: int):
    assert isinstance(level, int) and level >= 0
    filtered_list_of_wgmma_shapes = [wgmma_shape for wgmma_shape, min_level in SM90_WGMMA_SHAPES_INT8_DENSE.items() if level >= min_level]
    return filtered_list_of_wgmma_shapes

def generate_tf32_math_instructions_sm90(level: int):
    wgmma_level = get_wgmma_level_from_global_level(level)
    math_instructions = []
    for math_instruction_shape in generate_tf32_math_instruction_shapes_sm90(wgmma_level):
        math_instructions.append(MathInstruction(math_instruction_shape, DataType.tf32, DataType.tf32, DataType.f32, OpcodeClass.TensorOp, MathOperation.multiply_add))
    return math_instructions

def generate_fp16_bf16_math_instructions_sm90(level: int):
    wgmma_level = get_wgmma_level_from_global_level(level)
    math_instructions = []
    for math_instruction_shape in generate_fp16_bf16_math_instruction_shapes_sm90(wgmma_level):
        math_instructions += [MathInstruction(math_instruction_shape, DataType.f16, DataType.f16, DataType.f16, OpcodeClass.TensorOp, MathOperation.multiply_add), MathInstruction(math_instruction_shape, DataType.f16, DataType.f16, DataType.f32, OpcodeClass.TensorOp, MathOperation.multiply_add), MathInstruction(math_instruction_shape, DataType.bf16, DataType.bf16, DataType.f32, OpcodeClass.TensorOp, MathOperation.multiply_add)]
    return math_instructions

def generate_fp8_math_instructions_sm90(level: int):
    wgmma_level = get_wgmma_level_from_global_level(level)
    math_instructions = []
    for math_instruction_shape in generate_fp8_math_instruction_shapes_sm90(wgmma_level):
        math_instructions += [MathInstruction(math_instruction_shape, DataType.e4m3, DataType.e4m3, DataType.f32, OpcodeClass.TensorOp, MathOperation.multiply_add), MathInstruction(math_instruction_shape, DataType.e4m3, DataType.e5m2, DataType.f32, OpcodeClass.TensorOp, MathOperation.multiply_add), MathInstruction(math_instruction_shape, DataType.e5m2, DataType.e4m3, DataType.f32, OpcodeClass.TensorOp, MathOperation.multiply_add), MathInstruction(math_instruction_shape, DataType.e5m2, DataType.e5m2, DataType.f32, OpcodeClass.TensorOp, MathOperation.multiply_add)]
    return math_instructions

def generate_int8_math_instructions_sm90(level: int):
    wgmma_level = get_wgmma_level_from_global_level(level)
    math_instructions = []
    for math_instruction_shape in generate_int8_math_instruction_shapes_sm90(wgmma_level):
        math_instructions += [MathInstruction(math_instruction_shape, DataType.s8, DataType.s8, DataType.s32, OpcodeClass.TensorOp, MathOperation.multiply_add), MathInstruction(math_instruction_shape, DataType.u8, DataType.u8, DataType.s32, OpcodeClass.TensorOp, MathOperation.multiply_add)]
    return math_instructions

def make_sparse_math_instructions(math_instructions):
    sparse_instructions = []
    for inst in math_instructions:
        if inst.opcode_class == OpcodeClass.TensorOp:
            sparse_instructions.append(MathInstruction((inst.instruction_shape[0], inst.instruction_shape[1], inst.instruction_shape[2] * 2), inst.element_a, inst.element_b, inst.element_accumulator, OpcodeClass.SparseTensorOp, inst.math_operation))
    return sparse_instructions

def is_tile_desc_valid(tile_description):
    if tile_description.minimum_compute_capability != 90 or tile_description.maximum_compute_capability != 90:
        return False
    element_a, element_b, element_accum = (tile_description.math_instruction.element_a, tile_description.math_instruction.element_b, tile_description.math_instruction.element_accumulator)
    cluster_shape, cta_shape, inst_shape = (tile_description.cluster_shape, tile_description.threadblock_shape, tile_description.math_instruction.instruction_shape)
    grid_size = cta_shape[0] * cluster_shape[0] + cta_shape[1] * cluster_shape[1] + cta_shape[2] * cluster_shape[2]
    cluster_size = cluster_shape[0] * cluster_shape[1] * cluster_shape[2]
    if cluster_size > 16 or cluster_size < 1:
        return False
    if grid_size < 1:
        return False
    if cta_shape[0] < 64 or cta_shape[0] % 64 != 0:
        return False
    if cta_shape[1] < 16 or cta_shape[1] % 8 != 0:
        return False
    if cta_shape[2] < 16 or cta_shape[2] % 8 != 0:
        return False
    if cta_shape[2] < inst_shape[2] or cta_shape[2] % inst_shape[2] != 0 or cta_shape[2] / inst_shape[2] < 2:
        return False
    if cta_shape[0] > 256 or cta_shape[1] > 256 or cta_shape[2] > 256:
        return False
    return True

def get_mma_multipliers(level: int):
    assert isinstance(level, int) and level >= 0
    mma_level = get_mma_level_from_global_level(level)
    return [mma_mul for mma_mul, mma_min_level in SM90_MMA_MULTIPLIERS.items() if mma_level >= mma_min_level]

def get_cluster_sizes(level: int, is_aligned: bool):
    if not is_aligned:
        return [(1, 1, 1)]
    assert isinstance(level, int) and level >= 0
    cluster_level = get_cluster_level_from_global_level(level)
    return [cluster_size for cluster_size, cluster_min_level in SM90_CLUSTER_SIZES.items() if cluster_level >= cluster_min_level]

def generate_tile_descriptions_sm90(math_instructions, is_aligned: bool, level: int):
    tile_descriptions = set()
    mma_multipliers, cluster_sizes = (get_mma_multipliers(level), get_cluster_sizes(level, is_aligned))
    for math_inst, mma_mul, cluster_size in product(math_instructions, mma_multipliers, cluster_sizes):
        tile_desc = TileDescription(threadblock_shape=[math_inst.instruction_shape[0] * mma_mul[0], math_inst.instruction_shape[1] * mma_mul[1], math_inst.instruction_shape[2] * mma_mul[2]], stages=0, warp_count=[4, 1, 1], math_instruction=math_inst, min_compute=90, max_compute=90, cluster_shape=cluster_size)
        if math_inst.opcode_class == OpcodeClass.SparseTensorOp:
            tile_desc.threadblock_shape[2] = tile_desc.threadblock_shape[2] // 2
        if is_tile_desc_valid(tile_desc):
            tile_descriptions.add(tile_desc)
    return tile_descriptions

def is_tile_desc_compatible_with_cooperative(tile_description):
    return tile_description.threadblock_shape[0] >= 128

def can_tile_desc_use_shmem_in_epilogue(tile_description, data_types):
    dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc, dtype_epi = (data_types['a_type'], data_types['b_type'], data_types['c_type'], data_types['d_type'], data_types['acc_type'], data_types['epi_type'])
    mn = tile_description.threadblock_shape[0] * tile_description.threadblock_shape[1]
    bitsize_c, bitsize_d = (DataTypeSize[dtype_c], DataTypeSize[dtype_d])
    shmem_bits_c, shmem_bits_d = (bitsize_c * mn, bitsize_d * mn)
    shmem_bits_total = shmem_bits_c + shmem_bits_d
    if shmem_bits_total > 2 ** 20:
        return False
    return True

def get_valid_schedules(tile_description, cuda_version, is_aligned, data_types, layout, instantiation_level, enable_fp8_fast_acc=True):
    level = get_pruning_level_from_global_level(instantiation_level)
    schedules = []
    stream_k_schedules = []
    if not is_tile_desc_valid(tile_description):
        return (schedules, stream_k_schedules)
    FP16_TYPES = [DataType.f16, DataType.bf16]
    is_fp16 = data_types['a_type'] in FP16_TYPES and data_types['b_type'] in FP16_TYPES
    FP8_TYPES = [DataType.e4m3, DataType.e5m2]
    is_fp8 = data_types['a_type'] in FP8_TYPES and data_types['b_type'] in FP8_TYPES
    can_do_fp8_fast_accum = is_fp8 and enable_fp8_fast_acc
    FP32_TYPES = [DataType.f32, DataType.tf32]
    is_fp32 = data_types['a_type'] in FP32_TYPES and data_types['b_type'] in FP32_TYPES
    requires_transposed_epilogue = is_fp32 and layout[0][0] == LayoutType.RowMajor and (layout[1][0] == LayoutType.RowMajor)
    is_sparse = tile_description.math_instruction.opcode_class == OpcodeClass.SparseTensorOp
    can_do_cooperative = is_tile_desc_compatible_with_cooperative(tile_description)
    can_do_tma_epilogue = is_aligned and (not requires_transposed_epilogue) and can_tile_desc_use_shmem_in_epilogue(tile_description, data_types)
    default_epilogue = EpilogueScheduleType.NoSmemWarpSpecialized if not requires_transposed_epilogue else EpilogueScheduleType.EpilogueTransposed
    auto_epilogue = EpilogueScheduleType.ScheduleAuto if not requires_transposed_epilogue else EpilogueScheduleType.EpilogueTransposed
    cta_m, cta_n, cta_k = (tile_description.threadblock_shape[0], tile_description.threadblock_shape[1], tile_description.threadblock_shape[2])
    c_type = data_types['c_type']
    d_type = data_types['d_type']
    is_void_c = c_type == DataType.void
    if level < 1:
        if is_fp16 and cta_m <= 64 and (cta_n <= 128) and (cta_k <= 64):
            return ([], [])
        is_large_fp8_tile = is_fp8 and cta_m >= 256 and (cta_n >= 128) and (cta_k >= 128)
        if is_large_fp8_tile:
            if not is_void_c or d_type not in FP8_TYPES:
                return ([], [])
            if CudaToolkitVersionSatisfies(cuda_version, 12, 1) and can_do_cooperative and can_do_tma_epilogue:
                return ([[KernelScheduleType.TmaWarpSpecializedCooperative if not is_sparse else KernelScheduleType.TmaWarpSpecializedCooperativeFP8FastAccum, EpilogueScheduleType.TmaWarpSpecializedCooperative], [KernelScheduleType.TmaWarpSpecializedCooperativeFP8FastAccum, EpilogueScheduleType.TmaWarpSpecializedCooperative]], [])
            return ([], [])
        if is_fp8 and (not is_large_fp8_tile):
            valid_dtypes_for_c = [DataType.f32, DataType.bf16, DataType.f16]
            if c_type not in valid_dtypes_for_c or (d_type not in FP8_TYPES and c_type != d_type):
                return ([], [])
        if is_fp32 and is_void_c:
            return ([], [])
    if is_void_c and (not can_do_tma_epilogue):
        return ([], [])
    if not is_aligned:
        schedules = [[KernelScheduleType.CpAsyncWarpSpecialized, default_epilogue]]
        stream_k_schedules = []
        if CudaToolkitVersionSatisfies(cuda_version, 12, 1) and can_do_cooperative:
            schedules.append([KernelScheduleType.CpAsyncWarpSpecializedCooperative, default_epilogue])
            stream_k_schedules.append([KernelScheduleType.CpAsyncWarpSpecializedCooperative, default_epilogue])
        return (schedules, stream_k_schedules)
    schedules = []
    if level >= 1 or not is_void_c:
        if not is_fp8:
            schedules.append([KernelScheduleType.ScheduleAuto, auto_epilogue])
        if not (is_fp8 and is_sparse):
            schedules.append([KernelScheduleType.TmaWarpSpecialized, default_epilogue])
    stream_k_schedules = []
    if CudaToolkitVersionSatisfies(cuda_version, 12, 1):
        if not is_fp8 or level >= 1:
            schedules.append([KernelScheduleType.TmaWarpSpecializedPingpong, default_epilogue])
        if can_do_fp8_fast_accum:
            schedules.append([KernelScheduleType.TmaWarpSpecializedFP8FastAccum, default_epilogue])
            schedules.append([KernelScheduleType.TmaWarpSpecializedPingpongFP8FastAccum, default_epilogue])
        if can_do_cooperative:
            if not (is_fp8 and is_sparse):
                schedules.append([KernelScheduleType.TmaWarpSpecializedCooperative, default_epilogue])
                stream_k_schedules.append([KernelScheduleType.TmaWarpSpecializedCooperative, default_epilogue])
            if can_do_fp8_fast_accum:
                schedules.append([KernelScheduleType.TmaWarpSpecializedCooperativeFP8FastAccum, default_epilogue])
                stream_k_schedules.append([KernelScheduleType.TmaWarpSpecializedCooperativeFP8FastAccum, default_epilogue])
        if can_do_tma_epilogue:
            assert not requires_transposed_epilogue
            if not is_fp8 or level >= 1:
                schedules.append([KernelScheduleType.TmaWarpSpecializedPingpong, EpilogueScheduleType.TmaWarpSpecialized])
            if can_do_fp8_fast_accum:
                schedules.append([KernelScheduleType.TmaWarpSpecializedPingpongFP8FastAccum, EpilogueScheduleType.TmaWarpSpecialized])
            if can_do_cooperative:
                if not (is_fp8 and is_sparse):
                    schedules.append([KernelScheduleType.TmaWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative])
                    stream_k_schedules.append([KernelScheduleType.TmaWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative])
                if can_do_fp8_fast_accum:
                    schedules.append([KernelScheduleType.TmaWarpSpecializedCooperativeFP8FastAccum, EpilogueScheduleType.TmaWarpSpecializedCooperative])
                    stream_k_schedules.append([KernelScheduleType.TmaWarpSpecializedCooperativeFP8FastAccum, EpilogueScheduleType.TmaWarpSpecializedCooperative])
    return (schedules, stream_k_schedules)

def generate_data_types_from_math_instruction(math_instruction, element_source=None, element_dest=None, element_epilogue=None):
    element_a, element_b = (math_instruction.element_a, math_instruction.element_b)
    element_accumulator = math_instruction.element_accumulator
    element_c = element_source or element_accumulator
    element_d = element_dest or element_accumulator
    element_epilogue = element_epilogue or element_accumulator
    data_types = {'a_type': element_a, 'b_type': element_b, 'c_type': element_c, 'd_type': element_d, 'acc_type': element_accumulator, 'epi_type': element_epilogue}
    return data_types

def fix_alignments(data_types, layout, alignment_bits=128):
    operand_keys = ['a_type', 'b_type', 'c_type']
    operands_to_fix = ['c_type']
    new_layout = []
    assert len(layout) == len(operand_keys)
    for i, k in enumerate(operand_keys):
        assert k in data_types and data_types[k] in DataTypeSize
        dtype = data_types[k]
        dtype_size_bits = DataTypeSize[dtype]
        layout_type = layout[i][0]
        layout_alignment = layout[i][1]
        if k in operands_to_fix and dtype_size_bits >= 1:
            layout_alignment = alignment_bits // dtype_size_bits
        new_layout.append([layout_type, layout_alignment])
    return new_layout