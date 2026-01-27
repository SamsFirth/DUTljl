"""
Tests for a successful installation of the CUTLASS Python interface
"""
import os
import unittest
import cutlass
import cutlass_library

class InstallationTest(unittest.TestCase):

    def test_cutlass_source_paths(self):
        """
        Tests that CUTLASS source is available as part of the cutlass and cutlass_library packages
        """
        src_file = 'include/cutlass/cutlass.h'
        library_file = os.path.join(cutlass_library.source_path, src_file)
        cutlass_file = os.path.join(cutlass.CUTLASS_PATH, src_file)
        assert os.path.isfile(library_file), f'Unable to locate file {library_file}. Installation has not succeeded.'
        assert os.path.isfile(cutlass_file), f'Unable to locate file {cutlass_file}. Installation has not succeeded.'
if __name__ == '__main__':
    unittest.main()