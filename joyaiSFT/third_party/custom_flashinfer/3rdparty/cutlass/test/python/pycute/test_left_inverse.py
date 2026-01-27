"""
Unit tests for pycute.left_inverse
"""
import logging
import unittest
from pycute import *
_LOGGER = logging.getLogger(__name__)

class TestLeftInverse(unittest.TestCase):

    def helper_test_left_inverse(self, layout):
        inv_layout = left_inverse(layout)
        _LOGGER.debug(f'{layout}  =>  {inv_layout}')
        for i in range(size(layout)):
            self.assertEqual(inv_layout(layout(i)), i)

    def test_left_inverse(self):
        test = Layout(1, 0)
        self.helper_test_left_inverse(test)
        test = Layout((1, 1), (0, 0))
        self.helper_test_left_inverse(test)
        test = Layout(1, 1)
        self.helper_test_left_inverse(test)
        test = Layout(4, 1)
        self.helper_test_left_inverse(test)
        test = Layout(4, 2)
        self.helper_test_left_inverse(test)
        test = Layout((8, 4), (1, 8))
        self.helper_test_left_inverse(test)
        test = Layout((8, 4), (4, 1))
        self.helper_test_left_inverse(test)
        test = Layout((2, 4, 6), (1, 2, 8))
        self.helper_test_left_inverse(test)
        test = Layout((2, 4, 6), (4, 1, 8))
        self.helper_test_left_inverse(test)
        test = Layout((4, 2), (1, 16))
        self.helper_test_left_inverse(test)
if __name__ == '__main__':
    unittest.main()