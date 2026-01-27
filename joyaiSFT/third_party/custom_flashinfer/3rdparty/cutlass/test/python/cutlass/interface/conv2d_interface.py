"""
Tests the high-level Conv2d interface
"""
from math import ceil
import unittest
import cutlass
import cutlass.utils.datatypes as datatypes
from cutlass.backend.utils.device import device_cc
from utils import ExpectException
import os

class Conv2dEquivalence:
    """
    Helper class for testing the equivalence of different constructions of the Conv2d interface
    """

    def __init__(self, conv_kind, element_A, element_B, element_C, element_D, element_accumulator, alignment_A, alignment_B, alignment_C):
        self.element_A = element_A
        self.element_B = element_B
        self.element_C = element_C
        self.element_D = element_D
        self.element_accumulator = element_accumulator
        self.alignment_A = alignment_A
        self.alignment_B = alignment_B
        self.alignment_C = alignment_C
        self.conv_kind = conv_kind
        self.plan = cutlass.op.Conv2d(kind=self.conv_kind, element_A=element_A, element_B=element_B, element_C=element_C, element_D=element_D, element_accumulator=element_accumulator)
        self.op = self.plan.construct(alignment_A=self.alignment_A, alignment_B=self.alignment_B, alignment_C=self.alignment_C)

    def _plans_equal(self, other_plan) -> bool:
        """
        Compares whether two plans are equal

        :param other_plan: plan to compare against the default Conv2d
        :type other_plan: cutlass.op.Conv2d

        :return: whether `other_plan` is equivalent to `self.plan`
        :rtype: bool
        """
        other_op = other_plan.construct(alignment_A=self.alignment_A, alignment_B=self.alignment_B, alignment_C=self.alignment_C)
        return self.op.rt_module.emit() == other_op.rt_module.emit()

    def generic_test(self):
        """
        Tests the equivalence of various constructions of the Conv2d interface when using CUTLASS data types
        and layouts for constructing the Conv2d interface
        """
        if not datatypes.is_numpy_available():
            return
        plan_other = cutlass.op.Conv2d(kind=self.conv_kind, element_A=self.element_A, element_B=self.element_B, element_C=self.element_C, element_D=self.element_D, element_accumulator=self.element_accumulator)
        assert self._plans_equal(plan_other)
        plan_other = cutlass.op.Conv2d(kind=self.conv_kind, element_B=self.element_B, element_C=self.element_C, element_D=self.element_D, element_accumulator=self.element_accumulator, element=self.element_A)
        assert self._plans_equal(plan_other)
        plan_other = cutlass.op.Conv2d(kind=self.conv_kind, element_C=self.element_C, element_D=self.element_D, element_accumulator=self.element_accumulator, element=self.element_A)
        assert self._plans_equal(plan_other)
        if self.element_C == self.element_accumulator:
            plan_other = cutlass.op.Conv2d(kind=self.conv_kind, element_C=self.element_C, element_D=self.element_D, element=self.element_A)
            assert self._plans_equal(plan_other)
        if self.element_A == self.element_B and self.element_A == self.element_C and (self.element_A == self.element_D) and (self.element_A == self.element_accumulator):
            plan_other = cutlass.op.Conv2d(kind=self.conv_kind, element=self.element_A)
            assert self._plans_equal(plan_other)

    def numpy_test(self):
        """
        Tests the equivalence of various constructions of the Conv2d interface when using numpy as a frontend
        """
        if not datatypes.is_numpy_available():
            return
        import numpy as np
        type_A = datatypes.numpy_type(self.element_A)
        type_B = datatypes.numpy_type(self.element_B)
        type_C = datatypes.numpy_type(self.element_C)
        type_D = datatypes.numpy_type(self.element_D)
        type_accum = datatypes.numpy_type(self.element_accumulator)
        size = (2, 2)
        A = np.zeros(size, dtype=type_A)
        B = np.zeros(size, dtype=type_B)
        C = np.zeros(size, dtype=type_C)
        D = np.zeros(size, dtype=type_D)
        return self.tensor_test(type_A, type_B, type_C, type_D, type_accum, A, B, C, D)

    def torch_test(self):
        """
        Tests the equivalence of various constructions of the Conv2d interface when using torch as a frontend
        """
        if not datatypes.is_torch_available():
            return
        import torch
        type_A = datatypes.torch_type(self.element_A)
        type_B = datatypes.torch_type(self.element_B)
        type_C = datatypes.torch_type(self.element_C)
        type_D = datatypes.torch_type(self.element_D)
        type_accum = datatypes.torch_type(self.element_accumulator)
        size = (2, 2)
        A = torch.empty(size, dtype=type_A)
        B = torch.empty(size, dtype=type_B)
        C = torch.empty(size, dtype=type_C)
        D = torch.empty(size, dtype=type_D)
        return self.tensor_test(type_A, type_B, type_C, type_D, type_accum, A, B, C, D)

    def tensor_test(self, type_A, type_B, type_C, type_D, type_accum, A, B, C, D):
        plan_np = cutlass.op.Conv2d(kind=self.conv_kind, A=A, B=B, C=C, D=D, element_accumulator=type_accum)
        assert self._plans_equal(plan_np)
        plan_np = cutlass.op.Conv2d(kind=self.conv_kind, B=B, C=C, D=D, element_accumulator=type_accum, element_A=type_A)
        assert self._plans_equal(plan_np)
        if type_A == type_B:
            plan_np = cutlass.op.Conv2d(kind=self.conv_kind, C=C, D=D, element_accumulator=type_accum, element=type_A)
            assert self._plans_equal(plan_np)
        if type_C == type_accum:
            plan_np = cutlass.op.Conv2d(kind=self.conv_kind, A=A, B=B, C=C, D=D)
            assert self._plans_equal(plan_np)
        if type_A == type_B and type_A == type_C and (type_A == type_D) and (type_A == type_accum):
            plan_np = cutlass.op.Conv2d(kind=self.conv_kind, element=type_A)
            assert self._plans_equal(plan_np)

    def test_all(self):
        """
        Runs all tests on the Gemm interface
        """
        self.generic_test()
        self.numpy_test()
        self.torch_test()

@unittest.skipIf(device_cc() <= 80, 'Device compute capability is insufficient for SM80 tests.')
class ConvEquivalenceTest(unittest.TestCase):
    """
    Tests the equivalence of different constructions of the Conv2d interface
    """
    pass
type2alignment = {cutlass.DataType.f16: 8, cutlass.DataType.f32: 4}

def add_test(conv_kind, element_A, element_B, element_C, element_D, element_accumulator):
    test_name = f'test_conv2d_{conv_kind}_{element_A}_{element_B}_{element_C}_{element_D}_{element_accumulator}'

    def run(self):
        conv2d_eq = Conv2dEquivalence(conv_kind=conv_kind, element_A=element_A, element_B=element_B, element_C=element_C, element_D=element_D, element_accumulator=element_accumulator, alignment_A=type2alignment[element_A], alignment_B=type2alignment[element_B], alignment_C=type2alignment[element_C])
        conv2d_eq.test_all()
    setattr(ConvEquivalenceTest, test_name, run)
for conv_kind in ['fprop', 'wgrad', 'dgrad']:
    for types in [[cutlass.DataType.f16, cutlass.DataType.f16, cutlass.DataType.f16, cutlass.DataType.f16, cutlass.DataType.f16], [cutlass.DataType.f16, cutlass.DataType.f16, cutlass.DataType.f16, cutlass.DataType.f16, cutlass.DataType.f32], [cutlass.DataType.f16, cutlass.DataType.f16, cutlass.DataType.f32, cutlass.DataType.f32, cutlass.DataType.f16], [cutlass.DataType.f16, cutlass.DataType.f16, cutlass.DataType.f32, cutlass.DataType.f32, cutlass.DataType.f32], [cutlass.DataType.f32, cutlass.DataType.f32, cutlass.DataType.f32, cutlass.DataType.f32, cutlass.DataType.f32]]:
        add_test(conv_kind, types[0], types[1], types[2], types[3], types[4])

@unittest.skipIf(device_cc() <= 80, 'Device compute capability is insufficient for SM80 tests.')
class Conv2dErrorTests(unittest.TestCase):
    """
    Tests various error scenarios that arise with the high-level Gemm interface
    """

    def test_alignment(self):
        """
        Tests case in which the alignment specified is unsupported
        """
        plan = cutlass.op.Conv2d(kind='fprop', element=cutlass.DataType.f16)
        with ExpectException(True, 'Alignment 3 is not supported for F16. The construction should fail.'):
            op = plan.construct(alignment_A=3, alignment_B=3, alignment_C=3)

    def test_invalid_tile_description(self):
        """
        Tests scenarios in which an invalid tile description is provided for a given CC
        """
        plan = cutlass.op.Conv2d(kind='fprop', element=cutlass.DataType.f16)
        td = plan.tile_descriptions()[0]
        td.threadblock_shape = [17, 32, 5]
        plan.tile_description = td
        with ExpectException(True, 'The threadblock shape is invalid. The compilation should fail.'):
            plan.compile()
        os.remove('./cutlass_python_compilation_device_error.txt')
if __name__ == '__main__':
    unittest.main()