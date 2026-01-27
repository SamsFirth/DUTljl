"""
Functions for manipulating IntTuples
"""
from functools import reduce
from itertools import chain
from typing import Union
from .typing import Integer

def is_int(x):
    return isinstance(x, Integer)

def is_tuple(x):
    return isinstance(x, tuple)

def flatten(t):
    if is_tuple(t):
        if len(t) == 0:
            return ()
        else:
            return tuple((i for a in t for i in flatten(a)))
    else:
        return (t,)

def signum(a):
    return bool(a > 0) - bool(a < 0)

def product(a):
    if is_tuple(a):
        return reduce(lambda val, elem: val * product(elem), a, 1)
    else:
        return a

def inner_product(a, b):
    if is_tuple(a):
        assert len(a) == len(b)
        return sum((inner_product(x, y) for x, y in zip(a, b)))
    else:
        assert not is_tuple(b)
        return a * b

def tuple_max(a):
    if is_tuple(a):
        return max((tuple_max(x) for x in a))
    else:
        return a

def elem_scale(a, b):
    if is_tuple(a):
        if is_tuple(b):
            assert len(a) == len(b)
            return tuple((elem_scale(x, y) for x, y in zip(a, b)))
        else:
            assert False
    elif is_tuple(b):
        return elem_scale(a, product(b))
    else:
        return a * b

def shape_div(a, b):
    if is_tuple(a):
        if is_tuple(b):
            assert len(a) == len(b)
            return tuple((shape_div(x, y) for x, y in zip(a, b)))
        else:
            r = []
            for v in a:
                r.append(shape_div(v, b))
                b = shape_div(b, product(v))
            return tuple(r)
    elif is_tuple(b):
        return shape_div(a, product(b))
    else:
        assert a % b == 0 or b % a == 0
        if a % b == 0:
            return a // b
        else:
            return signum(a * b)

def prefix_product(a, init=1):
    if is_tuple(a):
        if is_tuple(init):
            assert len(a) == len(init)
            return tuple((prefix_product(x, i) for x, i in zip(a, init)))
        else:
            r = []
            for v in a:
                r.append(prefix_product(v, init))
                init = init * product(v)
            return tuple(r)
    elif is_tuple(init):
        assert False
    else:
        return init

def idx2crd(idx, shape, stride=None):
    if stride is None:
        stride = prefix_product(shape)
    if is_tuple(idx):
        if is_tuple(shape):
            assert len(idx) == len(shape) and len(idx) == len(stride)
            return tuple((idx2crd(i, s, d) for i, s, d in zip(idx, shape, stride)))
        else:
            assert False
    elif is_tuple(shape):
        assert len(shape) == len(stride)
        return tuple((idx2crd(idx, s, d) for s, d in zip(shape, stride)))
    else:
        return idx // stride % shape

def crd2idx(crd, shape, stride=None):
    if stride is None:
        stride = prefix_product(shape)
    if is_tuple(crd):
        if is_tuple(shape):
            assert len(crd) == len(shape) and len(crd) == len(stride)
            return sum((crd2idx(c, s, d) for c, s, d in zip(crd, shape, stride)))
        else:
            assert False, f'crd={crd}, shape={shape}'
    else:
        if crd is None:
            crd = 0
        if is_tuple(shape):
            assert len(shape) == len(stride)
            result = 0
            for i in range(len(shape) - 1):
                result += crd2idx(crd % product(shape[i]), shape[i], stride[i])
                crd = crd // product(shape[i])
            return result + crd2idx(crd, shape[-1], stride[-1])
        else:
            return crd * stride

def crd2crd(crd, dst_shape, src_shape=None):
    if is_tuple(crd):
        if is_tuple(dst_shape):
            assert len(crd) == len(dst_shape)
            return tuple((crd2crd(x, y) for x, y in zip(crd, dst_shape)))
        else:
            assert src_shape is not None
            return crd2idx(crd, src_shape)
    elif is_tuple(dst_shape):
        return idx2crd(crd, dst_shape)
    else:
        assert crd < dst_shape
        return crd

def slice_(crd: Union[None, tuple, int], trg: Union[tuple, int]):
    if is_tuple(crd):
        if is_tuple(trg):
            assert len(crd) == len(trg)
            return tuple(chain(*filter(lambda x: x != (), [slice_(c, s) for c, s in zip(crd, trg)])))
        else:
            assert False
    elif crd is None:
        return (trg,)
    else:
        return ()

def has_none(a: Union[None, tuple, int]):
    if is_tuple(a):
        return any((has_none(v) for v in a))
    else:
        return a is None