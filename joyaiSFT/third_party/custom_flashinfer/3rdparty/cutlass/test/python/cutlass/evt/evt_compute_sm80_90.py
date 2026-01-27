"""
Unit test for compute node in SM90
"""
import logging
import unittest
import cutlass
from cutlass.backend import *
from cutlass.epilogue import *
from cutlass import swizzle
from utils.evt_testbed import EVTTestBed, EVTTestCaseBase
cutlass.set_log_level(logging.WARNING)

@unittest.skipIf(device_cc() not in [80, 86, 89, 90], 'This unittest is only supported on CC [80, 86, 89, 90]')
class TestEVTCompute(EVTTestCaseBase):

    def test_arith(self):
        """
        Test Arithmatic op
        """

        def evt_arith_compute(accum, C, alpha, beta, gamma):
            D = ((accum + C) * alpha - gamma) / beta
            return D
        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {'accum': self.fake_tensor(self.element, (l, m, n)), 'C': self.fake_tensor(self.element, (l, m, n)), 'alpha': 1.5, 'beta': 0.5, 'gamma': 2.5, 'D': self.fake_tensor(self.element, (l, m, n))}
            launcher = EVTTestBed(self.element, evt_arith_compute, example_inputs)
            input_keys = ['C', 'alpha', 'beta', 'gamma']
            result_keys = ['D']
            launcher.verify((m, n, k), input_keys, result_keys, l)

    def test_func_call(self):
        """
        Test Function call
        """

        def evt_func_call(accum, C, alpha, beta, gamma):
            D = multiply_add(relu(accum + alpha) + C, beta, gamma)
            return D
        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {'accum': self.fake_tensor(self.element, (l, m, n)), 'C': self.fake_tensor(self.element, (l, m, n)), 'alpha': 1.5, 'beta': 0.5, 'gamma': 2.5, 'D': self.fake_tensor(self.element, (l, m, n))}
            launcher = EVTTestBed(self.element, evt_func_call, example_inputs)
            input_keys = ['C', 'alpha', 'beta', 'gamma']
            result_keys = ['D']
            launcher.verify((m, n, k), input_keys, result_keys, l)
if __name__ == '__main__':
    unittest.main()