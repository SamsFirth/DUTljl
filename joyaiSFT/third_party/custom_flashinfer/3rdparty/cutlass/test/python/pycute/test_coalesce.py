"""
Unit tests for pycute.coalesce
"""
import logging
import unittest
from pycute import *
_LOGGER = logging.getLogger(__name__)

class TestCoalesce(unittest.TestCase):

    def helper_test_coalesce(self, layout):
        layoutR = coalesce(layout)
        _LOGGER.debug(f'{layout}  =>  {layoutR}')
        self.assertEqual(size(layoutR), size(layout))
        for i in range(size(layout)):
            self.assertEqual(layoutR(i), layout(i))

    def test_coalesce(self):
        layout = Layout(1, 0)
        self.helper_test_coalesce(layout)
        layout = Layout(1, 1)
        self.helper_test_coalesce(layout)
        layout = Layout((2, 4))
        self.helper_test_coalesce(layout)
        layout = Layout((2, 4, 6))
        self.helper_test_coalesce(layout)
        layout = Layout((2, 4, 6), (1, 6, 2))
        self.helper_test_coalesce(layout)
        layout = Layout((2, 1, 6), (1, 7, 2))
        self.helper_test_coalesce(layout)
        layout = Layout((2, 1, 6), (4, 7, 8))
        self.helper_test_coalesce(layout)
        layout = Layout((2, (4, 6)))
        self.helper_test_coalesce(layout)
        layout = Layout((2, 4), (4, 1))
        self.helper_test_coalesce(layout)
        layout = Layout((2, 4, 6), (24, 6, 1))
        self.helper_test_coalesce(layout)
        layout = Layout((2, 1, 3), (2, 4, 4))
        self.helper_test_coalesce(layout)
        layout = Layout(((2, 2), (2, 2)), ((1, 4), (8, 32)))
        self.helper_test_coalesce(layout)
if __name__ == '__main__':
    unittest.main()