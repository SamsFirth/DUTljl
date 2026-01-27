"""
Collection of builtin functions used for host reference in EVT
"""
import numpy as np
from cutlass.utils.datatypes import is_cupy_tensor, is_numpy_tensor, is_torch_available, is_torch_tensor
if is_torch_available():
    import torch

def multiply_add(x, y, z):
    return x * y + z

def sum(x, dim):
    if is_numpy_tensor(x):
        return x.sum(axis=tuple(dim))
    elif is_torch_tensor(x):
        return torch.sum(x, dim)

def max(x, dim):
    if is_numpy_tensor(x):
        return x.max(axis=tuple(dim))
    elif is_torch_tensor(x):
        return torch.amax(x, dim)

def permute(x, indices: tuple):
    if is_numpy_tensor(x):
        return np.transpose(x, axes=indices)
    elif is_torch_tensor(x):
        return x.permute(*indices)

def reshape(x, new_shape: tuple):
    if is_numpy_tensor(x):
        return np.reshape(x, newshape=new_shape)
    elif is_torch_tensor(x):
        return x.view(new_shape)