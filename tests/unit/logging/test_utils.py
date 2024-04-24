"""Test ``SDV`` logging utilities."""
import logging
from unittest.mock import Mock, mock_open, patch

from sdv.logging.utils import get_sdv_logger, get_sdv_logger_config


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


@patch('sdv.logging.utils.logging.StreamHandler')
@patch('sdv.logging.utils.logging.getLogger')
@patch('sdv.logging.utils.get_sdv_logger_config')
def test_get_sdv_logger(mock_get_sdv_logger_config, mock_getlogger, mock_streamhandler):
    # Setup
    mock_logger_conf = {
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
    mock_get_sdv_logger_config.return_value = mock_logger_conf
    mock_logger_instance = Mock()
    mock_getlogger.return_value = mock_logger_instance

    # Run
    get_sdv_logger('test_logger')

    # Assert
    mock_logger_instance.setLevel.assert_called_once_with(logging.DEBUG)
    mock_logger_instance.addHandler.assert_called_once()
