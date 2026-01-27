"""
Unit tests for pycute.int_tuple
"""
import unittest
from pycute import *

class TestIntTuple(unittest.TestCase):

    def test_product(self):
        self.assertEqual(product(2), 2)
        self.assertEqual(product((3, 2)), 6)
        self.assertEqual(product(product(((2, 3), 4))), 24)

    def test_inner_product(self):
        self.assertEqual(inner_product(2, 3), 6)
        self.assertEqual(inner_product((1, 2), (3, 2)), 7)
        self.assertEqual(inner_product(((2, 3), 4), ((2, 1), 2)), 15)

    def test_shape_div(self):
        self.assertEqual(shape_div((3, 4), 6), (1, 2))
        self.assertEqual(shape_div((3, 4), 12), (1, 1))
        self.assertEqual(shape_div((3, 4), 36), (1, 1))
        self.assertEqual(shape_div(((3, 4), 6), 36), ((1, 1), 2))
        self.assertEqual(shape_div((6, (3, 4)), 36), (1, (1, 2)))

    def test_prefix_product(self):
        self.assertEqual(prefix_product(2), 1)
        self.assertEqual(prefix_product((3, 2)), (1, 3))
        self.assertEqual(prefix_product((3, 2, 4)), (1, 3, 6))
        self.assertEqual(prefix_product(((2, 3), 4)), ((1, 2), 6))
        self.assertEqual(prefix_product(((2, 3), (2, 1, 2), (5, 2, 1))), ((1, 2), (6, 12, 12), (24, 120, 240)))