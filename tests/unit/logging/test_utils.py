"""Test ``SDV`` logging utilities."""
from unittest.mock import Mock, mock_open, patch

from sdv.logging.utils import disable_single_table_logger, get_sdv_logger_config


def test_get_sdv_logger_config():
    """Test the ``get_sdv_logger_config``.

    Test that a ``yaml_content`` is being converted to ``dictionary`` and is returned
    by the ``get_sdv_logger_config``.
    """
    yaml_content = """
    log_registry: 'local'
    loggers:
        test_logger:
            level: DEBUG
            handlers:
                class: logging.StreamHandler
    """
    # Run
    with patch('builtins.open', mock_open(read_data=yaml_content)):
        # Test if the function returns a dictionary
        logger_conf = get_sdv_logger_config()

    # Assert
    assert isinstance(logger_conf, dict)
    assert logger_conf == {
        'log_registry': 'local',
        'loggers': {
            'test_logger': {
                'level': 'DEBUG',
                'handlers': {
                    'class': 'logging.StreamHandler'
                }
            }
        }
    }


@patch('sdv.logging.utils.logging.getLogger')
def test_disable_single_table_logger(mock_getlogger):
    # Setup
    mock_logger = Mock()
    handler = Mock()
    mock_logger.handlers = [handler]
    mock_logger.removeHandler.side_effect = lambda x: mock_logger.handlers.pop(0)
    mock_logger.addHandler.side_effect = lambda x: mock_logger.handlers.append(x)
    mock_getlogger.return_value = mock_logger

    # Run
    with disable_single_table_logger():
        assert len(mock_logger.handlers) == 0

    # Assert
    assert len(mock_logger.handlers) == 1
