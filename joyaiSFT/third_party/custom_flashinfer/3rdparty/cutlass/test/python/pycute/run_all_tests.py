"""
Utility script for discovering and running all PyCuTe tests
"""
import argparse
import logging
import pathlib
import unittest

def numeric_log_level(log_level: str) -> int:
    """
  Converts the string identifier of the log level into the numeric identifier used
  in setting the log level

  :param x: string representation of log level (e.g., 'INFO', 'DEBUG')
  :type x: str

  :return: numeric representation of log level
  :rtype: int
  """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    return numeric_level
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-level', default='info', type=numeric_log_level, required=False, help='Logging level to be used by the generator script')
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)
    loader = unittest.TestLoader()
    script_dir = str(pathlib.Path(__file__).parent.resolve()) + '/'
    tests = loader.discover(script_dir, 'test_*.py')
    test_runner = unittest.runner.TextTestRunner()
    results = test_runner.run(tests)
    if not results.wasSuccessful():
        raise Exception('Test cases failed')