"""
Definition of CuTe Layouts and functions to manipulate them
"""
from itertools import chain
from typing import Union
from .int_tuple import *

class LayoutBase:
    pass

def is_layout(x):
    return isinstance(x, LayoutBase)

class Layout(LayoutBase):

    def __init__(self, _shape, _stride=None):
        self.shape = _shape
        if _stride is None:
            self.stride = prefix_product(self.shape)
        else:
            self.stride = _stride

    def __eq__(self, other):
        return self.shape == other.shape and self.stride == other.stride

    def __len__(self):
        if is_tuple(self.shape):
            return len(self.shape)
        else:
            return 1

    def __call__(self, *args):
        """
    Map a logical coordinate to a linear index (Coord has no Underscore slice operators)
    OR
    Slice the layout and return the sublayout (Coord has an Underscore slice op)

    Follow the same behavior of `Layout::operator(Coord const&)` in cute C++
    """
        if has_none(args):
            if len(args) == 1:
                return Layout(slice_(args[0], self.shape), slice_(args[0], self.stride))
            else:
                return Layout(slice_(args, self.shape), slice_(args, self.stride))
        elif len(args) == 1:
            return crd2idx(args[0], self.shape, self.stride)
        else:
            return crd2idx(args, self.shape, self.stride)

    def __getitem__(self, i):
        if is_tuple(self.shape):
            return Layout(self.shape[i], self.stride[i])
        else:
            assert i == 0
            return Layout(self.shape, self.stride)

    def size(self):
        return product(self.shape)

    def cosize(self):
        return self(self.size() - 1) + 1

    def __str__(self):
        return f'{self.shape}:{self.stride}'

    def __repr__(self):
        return f'Layout({self.shape},{self.stride})'

def make_layout(*layouts):
    if len(layouts) == 1 and (not is_layout(layouts[0])):
        layouts = layouts[0]
    shape, stride = zip(*((a.shape, a.stride) for a in layouts))
    return Layout(shape, stride)

def size(layout):
    if is_layout(layout):
        return layout.size()
    return product(layout)

def cosize(layout):
    return layout.cosize()

def coalesce(layout, profile=None):
    if is_tuple(profile):
        assert len(layout) >= len(profile)
        return make_layout(chain((coalesce(layout[i], profile[i]) for i in range(0, len(profile))), (layout[i] for i in range(len(profile), len(layout)))))
    result_shape = [1]
    result_stride = [0]
    for shape, stride in zip(flatten(layout.shape), flatten(layout.stride)):
        if shape == 1:
            continue
        elif result_shape[-1] == 1:
            result_shape[-1] = shape
            result_stride[-1] = stride
        elif result_shape[-1] * result_stride[-1] == stride:
            result_shape[-1] = result_shape[-1] * shape
        else:
            result_shape.append(shape)
            result_stride.append(stride)
    if len(result_shape) == 1:
        return Layout(result_shape[0], result_stride[0])
    else:
        return Layout(tuple(result_shape), tuple(result_stride))

def filter(layout, profile=None):
    if is_tuple(profile):
        assert len(layout) >= len(profile)
        return make_layout(chain((filter(layout[i], profile[i]) for i in range(0, len(profile))), (layout[i] for i in range(len(profile), len(layout)))))
    result_shape = []
    result_stride = []
    for shape, stride in zip(flatten(layout.shape), flatten(layout.stride)):
        if not (shape == 1 or stride == 0):
            result_shape.append(shape)
            result_stride.append(stride)
    if len(result_shape) == 0:
        return Layout(1, 0)
    else:
        return coalesce(Layout(tuple(result_shape), tuple(result_stride)))

def composition(layoutA, layoutB):
    if layoutB is None:
        return layoutA
    elif is_int(layoutB):
        return composition(layoutA, Layout(layoutB))
    elif is_tuple(layoutB):
        assert len(layoutA) >= len(layoutB)
        return make_layout(chain((composition(layoutA[i], layoutB[i]) for i in range(0, len(layoutB))), (layoutA[i] for i in range(len(layoutB), len(layoutA)))))
    elif is_tuple(layoutB.shape):
        return make_layout((composition(layoutA, layoutB_i) for layoutB_i in layoutB))
    if layoutB.stride == 0:
        return Layout(layoutB.shape, 0)
    else:
        result_shape = []
        result_stride = []
        rest_shape = layoutB.shape
        rest_stride = layoutB.stride
        for s, d in zip(flatten(layoutA.shape)[:-1], flatten(layoutA.stride)[:-1]):
            s1 = shape_div(s, rest_stride)
            result_shape.append(min(s1, rest_shape))
            result_stride.append(rest_stride * d)
            rest_shape = shape_div(rest_shape, abs(s1))
            rest_stride = shape_div(rest_stride, s)
        result_shape.append(rest_shape)
        result_stride.append(rest_stride * flatten(layoutA.stride)[-1])
        return coalesce(Layout(tuple(result_shape), tuple(result_stride)))

def complement(layout, max_idx=1):
    if is_int(layout):
        return complement(Layout(layout))
    result_shape = []
    result_stride = []
    current_idx = 1
    sorted_DS = sorted(zip(flatten(layout.stride), flatten(layout.shape)))
    for stride, shape in sorted_DS:
        if stride == 0 or shape == 1:
            continue
        in_bound = current_idx <= shape * stride
        assert type(in_bound) is not bool or in_bound
        result_shape.append(stride // current_idx)
        result_stride.append(current_idx)
        current_idx = shape * stride
    result_shape.append((max_idx + current_idx - 1) // current_idx)
    result_stride.append(current_idx)
    return coalesce(Layout(tuple(result_shape), tuple(result_stride)))

def right_inverse(layout):
    if layout is None:
        return None
    elif is_int(layout):
        return Layout(layout)
    result_shape = []
    result_stride = []
    current_idx = 1
    flat_shape = flatten(layout.shape)
    flat_stride = flatten(layout.stride)
    sorted_DSA = sorted(zip(flat_stride, flat_shape, prefix_product(flat_shape)))
    for stride, shape, rstride in sorted_DSA:
        if shape == 1:
            continue
        if current_idx != stride:
            break
        result_shape.append(shape)
        result_stride.append(rstride)
        current_idx = shape * stride
    return coalesce(Layout(tuple(result_shape), tuple(result_stride)))

def left_inverse(layout):
    if layout is None:
        return None
    elif is_int(layout):
        return Layout(layout)
    return right_inverse(make_layout(layout, complement(layout)))

def logical_divide(layoutA, layoutB):
    if layoutB is None:
        return layoutA
    elif is_int(layoutB):
        return logical_divide(layoutA, Layout(layoutB))
    elif is_tuple(layoutB):
        assert len(layoutA) >= len(layoutB)
        return make_layout(chain((logical_divide(layoutA[i], layoutB[i]) for i in range(0, len(layoutB))), (layoutA[i] for i in range(len(layoutB), len(layoutA)))))
    return composition(layoutA, make_layout(layoutB, complement(layoutB, size(layoutA))))

def logical_product(layoutA, layoutB):
    if layoutB is None:
        return layoutA
    elif is_int(layoutB):
        return logical_divide(layoutA, Layout(layoutB))
    elif is_tuple(layoutB):
        assert len(layoutA) >= len(layoutB)
        return make_layout(chain((logical_product(layoutA[i], layoutB[i]) for i in range(0, len(layoutB))), (layoutA[i] for i in range(len(layoutB), len(layoutA)))))
    return make_layout(layoutA, composition(complement(layoutA, size(layoutA) * cosize(layoutB)), layoutB))

def hier_unzip(splitter, layoutA, layoutB):
    if layoutB is None:
        return make_layout(Layout(1, 0), layoutA)
    elif is_tuple(layoutB):
        assert len(layoutA) >= len(layoutB)
        split = make_layout((hier_unzip(splitter, layoutA[i], layoutB[i]) for i in range(0, len(layoutB))))
        return make_layout(make_layout((split[i][0] for i in range(0, len(layoutB)))), make_layout(chain((split[i][1] for i in range(0, len(layoutB))), (layoutA[i] for i in range(len(layoutB), len(layoutA))))))
    return splitter(layoutA, layoutB)

def zipped_divide(layoutA, layoutB):
    return hier_unzip(logical_divide, layoutA, layoutB)

def tiled_divide(layoutA, layoutB):
    result = zipped_divide(layoutA, layoutB)
    return make_layout([result[0]] + [result[1][i] for i in range(len(result[1]))])

def zipped_product(layoutA, layoutB):
    return hier_unzip(logical_product, layoutA, layoutB)

def tiled_product(layoutA, layoutB):
    result = zipped_product(layoutA, layoutB)
    return make_layout([result[0]] + [result[1][i] for i in range(len(result[1]))])

def slice_and_offset(crd: tuple, layout: Layout):
    return (Layout(slice_(crd, layout.shape), slice_(crd, layout.stride)), crd2idx(crd, layout.shape, layout.stride))