from unittest.mock import Mock, patch

from sdv.errors import log_exc_stacktrace


@patch('sdv.errors.traceback')
def test_log_exception(traceback_mock):
    """Test that the sys.excinfo is logged in a debug statement."""
    # Setup
    error = Mock(spec=Exception)
    logger = Mock()
    traceback_mock.format_exception.return_value = [
        'error line 1\n',
        'error line 2\n',
        'error line 3\n',
    ]

    # Run
    log_exc_stacktrace(logger, error)

    # Assert
    logger.debug.assert_called_once_with('error line 1\nerror line 2\nerror line 3\n')
