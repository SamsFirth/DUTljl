from __future__ import annotations
from typing import Callable, Sequence
from numpy.typing import DTypeLike
from .constants import GGML_QUANT_SIZES, GGMLQuantizationType
from .lazy import LazyNumpyTensor
import numpy as np

def quant_shape_to_byte_shape(shape: Sequence[int], quant_type: GGMLQuantizationType):
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    if shape[-1] % block_size != 0:
        raise ValueError(f'Quantized tensor row size ({shape[-1]}) is not a multiple of {quant_type.name} block size ({block_size})')
    return (*shape[:-1], shape[-1] // block_size * type_size)

def quant_shape_from_byte_shape(shape: Sequence[int], quant_type: GGMLQuantizationType):
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    if shape[-1] % type_size != 0:
        raise ValueError(f'Quantized tensor bytes per row ({shape[-1]}) is not a multiple of {quant_type.name} type size ({type_size})')
    return (*shape[:-1], shape[-1] // type_size * block_size)

def __compute_fp32_to_bf16(n: np.ndarray) -> np.ndarray:
    n = n.astype(np.float32, copy=False).view(np.int32)
    n = np.where(n & 2147483647 > 2139095040, n & 4294901760 | 64 << 16, n)
    n = np.where(n & 2139095040 == 0, n & 2147483648, n)
    n = n + (32767 + (n >> 16 & 1)) >> 16
    return n.astype(np.int16)

def __apply_over_grouped_rows(func: Callable[[np.ndarray], np.ndarray], arr: np.ndarray, otype: DTypeLike, oshape: tuple[int, ...]) -> np.ndarray:
    rows = arr.reshape((-1, arr.shape[-1]))
    osize = 1
    for dim in oshape:
        osize *= dim
    out = np.empty(shape=osize, dtype=otype)
    n_groups = rows.shape[0] // 16
    np.concatenate([func(group).ravel() for group in np.array_split(rows, n_groups)], axis=0, out=out)
    return out.reshape(oshape)

def __quantize_bf16_array(n: np.ndarray) -> np.ndarray:
    return __apply_over_grouped_rows(__compute_fp32_to_bf16, arr=n, otype=np.int16, oshape=n.shape)
__quantize_bf16_lazy = LazyNumpyTensor._wrap_fn(__quantize_bf16_array, meta_noop=np.int16)

def quantize_bf16(n: np.ndarray):
    if type(n) is LazyNumpyTensor:
        return __quantize_bf16_lazy(n)
    else:
        return __quantize_bf16_array(n)
__q8_block_size, __q8_type_size = GGML_QUANT_SIZES[GGMLQuantizationType.Q8_0]

def can_quantize_to_q8_0(n: np.ndarray) -> bool:
    return n.shape[-1] % __q8_block_size == 0

def np_roundf(n: np.ndarray) -> np.ndarray:
    a = abs(n)
    floored = np.floor(a)
    b = floored + np.floor(2 * (a - floored))
    return np.sign(n) * b

def __quantize_q8_0_shape_change(s: tuple[int, ...]) -> tuple[int, ...]:
    return (*s[:-1], s[-1] // __q8_block_size * __q8_type_size)

def __quantize_q8_0_rows(n: np.ndarray) -> np.ndarray:
    shape = n.shape
    assert shape[-1] % __q8_block_size == 0
    n_blocks = n.size // __q8_block_size
    blocks = n.reshape((n_blocks, __q8_block_size)).astype(np.float32, copy=False)
    d = abs(blocks).max(axis=1, keepdims=True) / 127
    with np.errstate(divide='ignore'):
        id = np.where(d == 0, 0, 1 / d)
    qs = np_roundf(blocks * id)
    d = d.astype(np.float16).view(np.uint8)
    qs = qs.astype(np.int8).view(np.uint8)
    assert d.shape[1] + qs.shape[1] == __q8_type_size
    return np.concatenate([d, qs], axis=1).reshape(__quantize_q8_0_shape_change(shape))

def __quantize_q8_0_array(n: np.ndarray) -> np.ndarray:
    return __apply_over_grouped_rows(__quantize_q8_0_rows, arr=n, otype=np.uint8, oshape=__quantize_q8_0_shape_change(n.shape))
__quantize_q8_0_lazy = LazyNumpyTensor._wrap_fn(__quantize_q8_0_array, meta_noop=(np.uint8, __quantize_q8_0_shape_change))

def quantize_q8_0(data: np.ndarray):
    if type(data) is LazyNumpyTensor:
        return __quantize_q8_0_lazy(data)
    else:
        return __quantize_q8_0_array(data)