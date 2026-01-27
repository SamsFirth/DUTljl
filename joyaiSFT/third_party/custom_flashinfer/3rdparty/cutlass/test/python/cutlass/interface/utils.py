"""
Helper functions & classes for interface test
"""

class ExpectException:
    """
    Utility class to assert that an exception was raised when expected

    Example:

    .. highlight:: python
    .. code-block:: python

        with ExceptionExpected(True, 'Division by zero'):
            x = 1.0 / 0.0

    :param exception_expected: whether an exception is expected to be raised
    :type exception_expected: bool
    :param message: message to print if an exception is raised when not expected or vice versa
    :type message: str
    """

    def __init__(self, exception_expected: bool, message: str='', verify_msg=False):
        self.exception_expected = exception_expected
        self.message = message
        self.verify_msg = verify_msg

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        exception_raised = exc_type is not None
        assert self.exception_expected == exception_raised, self.message
        if self.verify_msg:
            exc_message = f'{exc_type.__name__}: {exc_val}'
            assert exc_message == self.message, f'expect error message {self.message}, got {exc_message}'
        return True