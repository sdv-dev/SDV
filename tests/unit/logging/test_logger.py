"""Test ``SDV`` logger."""
import logging
from unittest.mock import Mock, patch

from sdv.logging.logger import get_sdv_logger


@patch('sdv.logging.logger.logging.StreamHandler')
@patch('sdv.logging.logger.logging.getLogger')
@patch('sdv.logging.logger.get_sdv_logger_config')
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
