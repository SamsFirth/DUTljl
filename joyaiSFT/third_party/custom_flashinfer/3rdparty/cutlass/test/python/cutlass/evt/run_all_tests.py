import pathlib
import unittest
if __name__ == '__main__':
    loader = unittest.TestLoader()
    script_dir = str(pathlib.Path(__file__).parent.resolve()) + '/'
    tests = loader.discover(script_dir, 'evt_*.py')
    testRunner = unittest.runner.TextTestRunner()
    results = testRunner.run(tests)
    if not results.wasSuccessful():
        raise Exception('Test cases failed')