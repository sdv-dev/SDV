"""Test ``SDV`` logger."""
import logging
from unittest.mock import Mock, patch

from sdv.logging.logger import CSVHandler, get_sdv_logger


class TestCSVHandler:

    def test_format(self):
        """Test CSV formatter correctly formats the log entry."""
        # Setup
        instance = CSVHandler()
        instance.writer = Mock()
        instance.output = Mock()
        record = Mock()
        record.msg = {
            'EVENT': 'Instance',
            'TIMESTAMP': '2024-04-19 16:20:10.037183',
            'SYNTHESIZER CLASS NAME': 'GaussianCopulaSynthesizer',
            'SYNTHESIZER ID': 'GaussainCopulaSynthesizer_1.0.0_92aff11e9a5649d1a280990d1231a5f5',
        }
        record.levelname = 'INFO'

        # Run
        instance.format(record)

        # Assert
        instance.writer.writerow.assert_called_once_with({'LEVEL': 'INFO', **record.msg})


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
    mock_logger_instance.handlers = []
    mock_getlogger.return_value = mock_logger_instance

    # Run
    get_sdv_logger('test_logger')

    # Assert
    mock_logger_instance.setLevel.assert_called_once_with(logging.DEBUG)
    mock_logger_instance.addHandler.assert_called_once()
