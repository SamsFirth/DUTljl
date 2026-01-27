"""
Tests the high-level GEMM interface
"""
from math import ceil
import unittest
import cutlass
import cutlass.utils.datatypes as datatypes
from cutlass.backend.utils.device import device_cc
from utils import ExpectException

class GemmEquivalence:
    """
    Helper class for testing the equivalence of different constructions of the Gemm interface
    """

    def __init__(self, element_A, element_B, element_C, element_D, element_accumulator, layout_A, layout_B, layout_C, alignment_A, alignment_B, alignment_C):
        self.element_A = element_A
        self.element_B = element_B
        self.element_C = element_C
        self.element_D = element_D
        self.element_accumulator = element_accumulator
        self.layout_A = layout_A
        self.layout_B = layout_B
        self.layout_C = layout_C
        self.alignment_A = alignment_A
        self.alignment_B = alignment_B
        self.alignment_C = alignment_C
        self.plan = cutlass.op.Gemm(element_A=element_A, element_B=element_B, element_C=element_C, element_D=element_D, element_accumulator=element_accumulator, layout_A=layout_A, layout_B=layout_B, layout_C=layout_C)
        self.op = self.plan.construct(alignment_A=alignment_A, alignment_B=alignment_B, alignment_C=alignment_C)

    def _plans_equal(self, other_plan) -> bool:
        """
        Compares whether two plans are equal

        :param other_plan: plan to compare against the default GEMM
        :type other_plan: cutlass.op.Gemm

        :return: whether `other_plan` is equivalent to `self.plan`
        :rtype: bool
        """
        other_op = other_plan.construct(alignment_A=self.alignment_A, alignment_B=self.alignment_B, alignment_C=self.alignment_C)
        return self.op.rt_module.emit() == other_op.rt_module.emit()

    def generic_test(self):
        """
        Tests the equivalence of various constructions of the Gemm interface when using CUTLASS data types
        and layouts for constructing the Gemm interface
        """
        if not datatypes.is_numpy_available():
            return
        plan_other = cutlass.op.Gemm(element_A=self.element_A, element_B=self.element_B, element_C=self.element_C, element_D=self.element_D, element_accumulator=self.element_accumulator, layout_A=self.layout_A, layout_B=self.layout_B, layout_C=self.layout_C)
        assert self._plans_equal(plan_other)
        plan_other = cutlass.op.Gemm(element_B=self.element_B, element_C=self.element_C, element_D=self.element_D, element_accumulator=self.element_accumulator, layout_B=self.layout_B, layout_C=self.layout_C, element=self.element_A, layout=self.layout_A)
        assert self._plans_equal(plan_other)
        if self.element_A == self.element_B and self.layout_A == self.layout_B:
            plan_other = cutlass.op.Gemm(element_C=self.element_C, element_D=self.element_D, element_accumulator=self.element_accumulator, layout_C=self.layout_C, element=self.element_A, layout=self.layout_A)
            assert self._plans_equal(plan_other)
        if self.element_C == self.element_accumulator:
            plan_other = cutlass.op.Gemm(element_A=self.element_A, element_B=self.element_B, element_C=self.element_C, element_D=self.element_D, layout_A=self.layout_A, layout_B=self.layout_B, layout_C=self.layout_C)
            assert self._plans_equal(plan_other)
        if self.element_A == self.element_B and self.element_A == self.element_C and (self.element_A == self.element_D) and (self.element_A == self.element_accumulator) and (self.layout_A == self.layout_B) and (self.layout_A == self.layout_C):
            plan_other = cutlass.op.Gemm(element=self.element_A, layout=self.layout_A)
            assert self._plans_equal(plan_other)

    def numpy_test(self):
        """
        Tests the equivalence of various constructions of the Gemm interface when using numpy as a frontend
        """
        if not datatypes.is_numpy_available():
            return
        import numpy as np
        type_A = datatypes.numpy_type(self.element_A)
        type_B = datatypes.numpy_type(self.element_B)
        type_C = datatypes.numpy_type(self.element_C)
        type_D = datatypes.numpy_type(self.element_D)
        type_accum = datatypes.numpy_type(self.element_accumulator)
        layout_to_order = {cutlass.LayoutType.RowMajor: 'C', cutlass.LayoutType.ColumnMajor: 'F'}
        size = (2, 2)
        A = np.zeros(size, order=layout_to_order[self.layout_A], dtype=type_A)
        B = np.zeros(size, order=layout_to_order[self.layout_B], dtype=type_B)
        C = np.zeros(size, order=layout_to_order[self.layout_C], dtype=type_C)
        D = np.zeros(size, order=layout_to_order[self.layout_C], dtype=type_D)
        plan_np = cutlass.op.Gemm(A=A, B=B, C=C, D=D, element_accumulator=type_accum)
        assert self._plans_equal(plan_np)
        plan_np = cutlass.op.Gemm(B=B, C=C, D=D, element_accumulator=type_accum, element_A=type_A, layout_A=self.layout_A)
        assert self._plans_equal(plan_np)
        if type_A == type_B and self.layout_A == self.layout_B:
            plan_np = cutlass.op.Gemm(C=C, D=D, element_accumulator=type_accum, element=type_A, layout=self.layout_A)
            assert self._plans_equal(plan_np)
        if type_C == type_accum:
            plan_np = cutlass.op.Gemm(A=A, B=B, C=C, D=D)
            assert self._plans_equal(plan_np)
        if type_A == type_B and type_A == type_C and (type_A == type_D) and (type_A == type_accum) and (self.layout_A == self.layout_B) and (self.layout_A == self.layout_C):
            plan_np = cutlass.op.Gemm(element=type_A, layout=self.layout_A)
            assert self._plans_equal(plan_np)

    def test_all(self):
        """
        Runs all tests on the Gemm interface
        """
        self.generic_test()
        self.numpy_test()

class GemmEquivalenceTest(unittest.TestCase):
    """
    Tests the equivalence of different constructions of the Gemm interface
    """

    @unittest.skipIf(device_cc() < 70, 'Device compute capability is insufficient for FP16 Tensor Core tests.')
    def test_gemm_equivalence_f16_f16_f16_f16_f16_ttt_8_8_8(self):
        gemm_eq = GemmEquivalence(element_A=cutlass.DataType.f16, element_B=cutlass.DataType.f16, element_C=cutlass.DataType.f16, element_D=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f16, layout_A=cutlass.LayoutType.RowMajor, layout_B=cutlass.LayoutType.RowMajor, layout_C=cutlass.LayoutType.RowMajor, alignment_A=8, alignment_B=8, alignment_C=8)
        gemm_eq.test_all()

    @unittest.skipIf(device_cc() < 70, 'Device compute capability is insufficient for FP16 Tensor Core tests.')
    def test_gemm_equivalence_f16_f16_f16_f16_f32_ntn_8_8_8(self):
        gemm_eq = GemmEquivalence(element_A=cutlass.DataType.f16, element_B=cutlass.DataType.f16, element_C=cutlass.DataType.f16, element_D=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f32, layout_A=cutlass.LayoutType.ColumnMajor, layout_B=cutlass.LayoutType.RowMajor, layout_C=cutlass.LayoutType.ColumnMajor, alignment_A=8, alignment_B=8, alignment_C=8)
        gemm_eq.test_all()

    @unittest.skipIf(device_cc() < 70, 'Device compute capability is insufficient for FP16 Tensor Core tests.')
    def test_gemm_equivalence_f16_f16_f16_f16_f16_ttt_4_4_4(self):
        gemm_eq = GemmEquivalence(element_A=cutlass.DataType.f16, element_B=cutlass.DataType.f16, element_C=cutlass.DataType.f16, element_D=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f16, layout_A=cutlass.LayoutType.RowMajor, layout_B=cutlass.LayoutType.RowMajor, layout_C=cutlass.LayoutType.RowMajor, alignment_A=8, alignment_B=8, alignment_C=8)
        gemm_eq.test_all()

    @unittest.skipIf(device_cc() < 80, 'Device compute capability is insufficient for F64 Tensor Core tests.')
    def test_gemm_equivalence_f64_f64_f64_f64_f64_tnt_1_1_1(self):
        gemm_eq = GemmEquivalence(element_A=cutlass.DataType.f64, element_B=cutlass.DataType.f64, element_C=cutlass.DataType.f64, element_D=cutlass.DataType.f64, element_accumulator=cutlass.DataType.f64, layout_A=cutlass.LayoutType.RowMajor, layout_B=cutlass.LayoutType.ColumnMajor, layout_C=cutlass.LayoutType.RowMajor, alignment_A=1, alignment_B=1, alignment_C=1)
        gemm_eq.test_all()

class GemmErrorTests(unittest.TestCase):
    """
    Tests various error scenarios that arise with the high-level Gemm interface
    """

    def test_alignment(self):
        """
        Tests case in which the alignment specified is unsupported
        """
        plan = cutlass.op.Gemm(element=cutlass.DataType.f16, layout=cutlass.LayoutType.RowMajor)
        with ExpectException(True, 'Alignment 16 is not supported for F16. The construction should fail.'):
            op = plan.construct(alignment_A=16, alignment_B=16, alignment_C=16)

    def test_tensorop_availability(self):
        """
        Tests case in which only SIMT operations are available but TensorOp is requested
        """
        cc = device_cc()
        supports_tensorop_f64 = cc >= 80
        plan = cutlass.op.Gemm(cc=cc, element=cutlass.DataType.f64, layout=cutlass.LayoutType.RowMajor)
        error_msg = f'Incorrectly raised an exception for availability of TensorOp with F64 operands on SM{cc}'
        with ExpectException(not supports_tensorop_f64, error_msg):
            plan.opclass = cutlass.OpcodeClass.TensorOp
        expected_opclass = cutlass.OpcodeClass.TensorOp if supports_tensorop_f64 else cutlass.OpcodeClass.Simt
        assert plan.opclass == expected_opclass, f'Expected opclass to be {expected_opclass}, but received {plan.opclass} for SM{cc}'

    @unittest.skipIf(device_cc() < 70, 'Device compute capability is insufficient for F16 Tensor Core tests.')
    def test_opclass_switch(self):
        """
        Tests cases in which the opcode class in question is switched (e.g., from TensorOp to SIMT)
        """
        plan = cutlass.op.Gemm(element=cutlass.DataType.f16, layout=cutlass.LayoutType.RowMajor)
        assert plan.opclass == cutlass.OpcodeClass.TensorOp
        for td in plan.tile_descriptions():
            assert td.math_instruction.opcode_class == cutlass.OpcodeClass.TensorOp
        plan.opclass = cutlass.OpcodeClass.Simt
        for td in plan.tile_descriptions():
            assert td.math_instruction.opcode_class == cutlass.OpcodeClass.Simt

    def test_invalid_tile_description(self):
        """
        Tests scenarios in which an invalid tile description is provided for a given CC
        """
        cc = device_cc()
        plan = cutlass.op.Gemm(cc=cc, element=cutlass.DataType.f16, layout=cutlass.LayoutType.RowMajor)
        td = plan.tile_descriptions()[0]
        stages = td.stages
        with ExpectException(cc < 90, f'Requested zero stages'):
            td.stages = 0
            plan.construct(td)
        if cc < 90:
            with ExpectException(cc < 80, f'Requested more than 2 stages on SM{cc}'):
                td.stages = 3
                plan.construct(td)
        else:
            original_kschedule = td.kernel_schedule
            original_eschedule = td.epilogue_schedule
            with ExpectException(False, f'Incorrectly flagged an error for insufficient shared memory'):
                td.kernel_schedule = cutlass.KernelScheduleType.TmaWarpSpecializedPingpong
                td.epilogue_schedule = cutlass.EpilogueScheduleType.NoSmemWarpSpecialized
                td.stages = 3
                plan.construct(td)
            td.kernel_schedule = original_kschedule
            td.epilogue_schedule = original_eschedule
        with ExpectException(True, f'Requested too many stages'):
            td.stages = 100
            plan.construct(td)
        td.stages = stages
        cluster_shape = td.cluster_shape
        with ExpectException(cc < 90, f'Requested non-unit cluster shape on SM{cc}'):
            td.cluster_shape = [2, 1, 1]
            plan.construct(td)
        td.cluster_shape = cluster_shape
        with ExpectException(cc < 90, f'Requested a non-auto schedule on SM{cc}'):
            td.kernel_schedule = cutlass.KernelScheduleType.TmaWarpSpecializedPingpong
            td.epilogue_schedule = cutlass.EpilogueScheduleType.TmaWarpSpecialized
            plan.construct(td)
        with ExpectException(True, f'Requested a non-auto kernel schedule with an auto epilogue schedule'):
            td.kernel_schedule = cutlass.KernelScheduleType.TmaWarpSpecializedPingpong
            td.epilogue_schedule = cutlass.EpilogueScheduleType.ScheduleAuto
            plan.construct(td)
        with ExpectException(True, f'Requested an auto kernel schedule with a non-auto epilogue schedule'):
            td.kernel_schedule = cutlass.KernelScheduleType.ScheduleAuto
            td.epilogue_schedule = cutlass.EpilogueScheduleType.TmaWarpSpecialized
            plan.construct(td)
        with ExpectException(cc < 90, f'Requested a tile scheduler on SM{cc}'):
            td.kernel_schedule = cutlass.KernelScheduleType.TmaWarpSpecializedCooperative
            td.epilogue_schedule = cutlass.EpilogueScheduleType.TmaWarpSpecializedCooperative
            td.tile_scheduler = cutlass.TileSchedulerType.StreamK
            plan.construct(td)
        ops = {}
        for i, td in enumerate(plan.tile_descriptions()):
            op = plan.construct(td)
            code_str = op.rt_module.emit()
            if code_str in ops:
                conflicting_td = ops[code_str]
                assert False, f'Multiple tile descriptions emitted {code_str}\nTile descriptions are:\n{td}\n{conflicting_td}'
if __name__ == '__main__':
    unittest.main()