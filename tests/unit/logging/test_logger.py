"""Test ``SDV`` logger."""
import logging
from unittest.mock import Mock, patch

from sdv.logging.logger import CSVFormatter, get_sdv_logger


class TestCSVFormatter:

    def test_format(self):
        """Test CSV formatter correctly formats the log entry."""
        # Setup
        instance = CSVFormatter()
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


@patch('sdv.logging.logger.CSVFormatter')
@patch('sdv.logging.logger.logging.FileHandler')
@patch('sdv.logging.logger.logging.getLogger')
@patch('sdv.logging.logger.get_sdv_logger_config')
def test_get_sdv_logger_csv(mock_get_sdv_logger_config, mock_getlogger,
                            mock_filehandler, mock_csvformatter):
    # Setup
    mock_logger_conf = {
        'log_registry': 'local',
        'loggers': {
            'test_logger': {
                'level': 'DEBUG',
                'format': 'CSV',
                'handlers': {
                    'filename': 'logfile.csv',
                    'class': 'logging.FileHandler'
                }
            }
        }
    }
    mock_get_sdv_logger_config.return_value = mock_logger_conf
    mock_logger_instance = Mock()
    mock_logger_instance.handlers = []
    mock_getlogger.return_value = mock_logger_instance
    mock_filehandler_instance = Mock()
    mock_filehandler.return_value = mock_filehandler_instance

    # Run
    get_sdv_logger('test_logger')

    # Assert
    mock_logger_instance.setLevel.assert_called_once_with(logging.DEBUG)
    mock_logger_instance.addHandler.assert_called_once_with(mock_filehandler_instance)
    mock_filehandler.assert_called_once_with('logfile.csv')
    mock_filehandler_instance.setLevel.assert_called_once_with(logging.DEBUG)
    mock_filehandler_instance.setFormatter.assert_called_once_with(mock_csvformatter.return_value)
