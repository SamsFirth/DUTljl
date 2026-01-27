"""
Unit tests for pycute.complement
"""
import logging
import unittest
from pycute import *
_LOGGER = logging.getLogger(__name__)

class TestComplement(unittest.TestCase):

    def helper_test_complement(self, layout):
        layoutR = complement(layout)
        _LOGGER.debug(f'{layout}  =>  {layoutR}')
        for a in range(size(layout)):
            for b in range(size(layoutR)):
                assert layout(a) != layoutR(b) or (layout(a) == 0 and layoutR(b) == 0)

    def test_complement(self):
        test = Layout(1, 0)
        self.helper_test_complement(test)
        test = Layout(1, 1)
        self.helper_test_complement(test)
        test = Layout(4, 0)
        self.helper_test_complement(test)
        test = Layout((2, 4), (1, 2))
        self.helper_test_complement(test)
        test = Layout((2, 3), (1, 2))
        self.helper_test_complement(test)
        test = Layout((2, 4), (1, 4))
        self.helper_test_complement(test)
        test = Layout((2, 4, 8), (8, 1, 64))
        self.helper_test_complement(test)
        test = Layout(((2, 2), (2, 2)), ((1, 4), (8, 32)))
        self.helper_test_complement(test)
        test = Layout((2, (3, 4)), (3, (1, 6)))
        self.helper_test_complement(test)
        test = Layout((4, 6), (1, 6))
        self.helper_test_complement(test)
        test = Layout((4, 10), (1, 10))
        self.helper_test_complement(test)
if __name__ == '__main__':
    unittest.main()