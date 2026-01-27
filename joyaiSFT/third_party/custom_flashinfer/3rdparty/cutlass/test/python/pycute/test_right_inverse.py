"""
Unit tests for pycute.left_inverse
"""
import logging
import unittest
from pycute import *
_LOGGER = logging.getLogger(__name__)

class TestRightInverse(unittest.TestCase):

    def helper_test_right_inverse(self, layout):
        inv_layout = right_inverse(layout)
        _LOGGER.debug(f'{layout}  =>  {inv_layout}')
        for i in range(size(inv_layout)):
            self.assertEqual(layout(inv_layout(i)), i)

    def test_right_inverse(self):
        test = Layout(1, 0)
        self.helper_test_right_inverse(test)
        test = Layout((1, 1), (0, 0))
        self.helper_test_right_inverse(test)
        test = Layout((3, 7), (0, 0))
        self.helper_test_right_inverse(test)
        test = Layout(1, 1)
        self.helper_test_right_inverse(test)
        test = Layout(4, 0)
        self.helper_test_right_inverse(test)
        test = Layout(4, 1)
        self.helper_test_right_inverse(test)
        test = Layout(4, 2)
        self.helper_test_right_inverse(test)
        test = Layout((2, 4), (0, 2))
        self.helper_test_right_inverse(test)
        test = Layout((8, 4), (1, 8))
        self.helper_test_right_inverse(test)
        test = Layout((8, 4), (4, 1))
        self.helper_test_right_inverse(test)
        test = Layout((2, 4, 6), (1, 2, 8))
        self.helper_test_right_inverse(test)
        test = Layout((2, 4, 6), (4, 1, 8))
        self.helper_test_right_inverse(test)
        test = Layout((4, 2), (1, 16))
        self.helper_test_right_inverse(test)
if __name__ == '__main__':
    unittest.main()