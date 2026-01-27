"""
High-level class for tensor
"""
from cutlass_library import LayoutType
from cutlass.backend.evt.ir.layout_algorithm import Layout, broadcast, canonicalization, permutation, reshape, _reverse_tuple
from cutlass.utils.datatypes import get_datatype_and_layout, get_tensor_shape, library_type

class Tensor:
    """
    The tensor abstracts the data type
    """

    def __init__(self, tensor=None, element=None, shape=None, layout_tag=None, is_constant=False) -> None:
        if element is not None and tensor is not None:
            raise Exception(f'Must not specify both element and tensor')
        elif shape is not None and tensor is not None:
            raise Exception(f'Must not specify both shape and tensor')
        elif layout_tag is not None and tensor is not None:
            raise Exception(f'Must not specify both layout_tag and tensor')
        elif (element is None or layout_tag is None or shape is None) and tensor is None:
            raise Exception(f'Must specify one of (element, shape, layout) or (tensor)')
        if isinstance(tensor, Tensor):
            self.__dict__.update(vars(tensor))
        else:
            if tensor is None:
                self.element = library_type(element)
            else:
                self.element, layout_tag = get_datatype_and_layout(tensor)
                shape = get_tensor_shape(tensor)
            if layout_tag == LayoutType.RowMajor:
                self.layout = Layout(shape[::-1])
            elif layout_tag == LayoutType.ColumnMajor:
                self.layout = permutation(Layout(shape), [idx for idx in reversed(range(len(shape)))])
            self.layout = canonicalization(self.layout)
            self.is_constant = is_constant
            if is_constant and tensor is not None:
                self.value = tensor

    @property
    def shape(self):
        """
        Returns the RowMajor layout shape
        """
        return _reverse_tuple(self.layout.shape)

    @property
    def stride(self):
        """
        Returns the RowMajor layout stride
        """
        return _reverse_tuple(self.layout.stride)

    @property
    def rank(self):
        """
        Returns the rank of the tensor
        """
        return len(self.shape)

    def broadcast(self, shape):
        """
        Broadcast self.layout to shape
        """
        assert isinstance(shape, tuple)
        self.layout = broadcast(self.layout, _reverse_tuple(shape))

    def reshape(self, shape):
        """
        Reshape self.layout to shape
        """
        assert isinstance(shape, tuple)
        reverse_shape = _reverse_tuple(shape)
        self.layout = reshape(self.layout, reverse_shape)

    def permute(self, indices):
        """
        Permute self.layout according to indices
        """
        length = len(indices)
        indices = [length - idx - 1 for idx in indices]
        self.layout = permutation(self.layout, indices[::-1])