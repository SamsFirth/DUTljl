"""Unit test utilities for Google C++ Mocking Framework."""
import os
from googletest.test import gtest_test_utils

def GetSourceDir():
    """Returns the absolute path of the directory where the .py files are."""
    return gtest_test_utils.GetSourceDir()

def GetTestExecutablePath(executable_name):
    """Returns the absolute path of the test binary given its name.

  The function will print a message and abort the program if the resulting file
  doesn't exist.

  Args:
    executable_name: name of the test binary that the test script runs.

  Returns:
    The absolute path of the test binary.
  """
    return gtest_test_utils.GetTestExecutablePath(executable_name)

def GetExitStatus(exit_code):
    """Returns the argument to exit(), or -1 if exit() wasn't called.

  Args:
    exit_code: the result value of os.system(command).
  """
    if os.name == 'nt':
        return exit_code
    elif os.WIFEXITED(exit_code):
        return os.WEXITSTATUS(exit_code)
    else:
        return -1
Subprocess = gtest_test_utils.Subprocess
TestCase = gtest_test_utils.TestCase
environ = gtest_test_utils.environ
SetEnvVar = gtest_test_utils.SetEnvVar
PREMATURE_EXIT_FILE_ENV_VAR = gtest_test_utils.PREMATURE_EXIT_FILE_ENV_VAR

def Main():
    """Runs the unit test."""
    gtest_test_utils.Main()