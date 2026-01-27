"""
Unit tests for pycute.typing
"""
import logging
import unittest
from pycute import *
_LOGGER = logging.getLogger(__name__)

class TestTyping(unittest.TestCase):

    def helper_test_typing(self, _cls, _obj, cls, expected: bool):
        _LOGGER.debug(f'issubclass({_cls}, {cls})')
        _LOGGER.debug(f'isinstance({_obj}, {cls})')
        self.assertEqual(expected, issubclass(_cls, cls))
        self.assertEqual(expected, isinstance(_obj, cls))

    def test_typing(self):
        self.helper_test_typing(int, 1, Integer, True)
        self.helper_test_typing(float, 1.0, Integer, False)
        self.helper_test_typing(str, 'hi', Integer, False)
        self.helper_test_typing(bool, False, Integer, False)
if __name__ == '__main__':
    unittest.main()