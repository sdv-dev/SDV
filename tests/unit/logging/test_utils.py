"""Test ``SDV`` logging utilities."""

from io import StringIO
from unittest.mock import Mock, mock_open, patch

import numpy as np
import pandas as pd

from sdv.logging.utils import (
    disable_single_table_logger,
    get_sdv_logger_config,
    load_logfile_dataframe,
)


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
            'test_logger': {'level': 'DEBUG', 'handlers': {'class': 'logging.StreamHandler'}}
        },
    }


@patch('sdv.logging.utils.platformdirs.user_data_dir')
@patch('sdv.logging.utils.Path.mkdir', side_effect=PermissionError)
@patch('sdv.logging.utils.os.access', return_value=False)
def test_get_logger_config_no_permissions(mock_os_access, mock_mkdir, mock_data_dir):
    """Test get_sdv_logger_config when there is no write or read permission.

    Return empty config when both user and default paths are inaccessible.
    """
    # Setup
    mock_data_dir.return_value = '/no_permission/user/config'

    # Run
    config = get_sdv_logger_config()

    # Assert
    assert config == {}


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


def test_load_logfile_dataframe():
    """Test loading the CSV logfile into a DataFrame"""
    # Setup
    logfile = StringIO(
        'INFO,Instance,2024-05-14 11:29:00.649735,GaussianCopulaSynthesizer,'
        'GaussianCopulaSynthesizer_1.12.1_5387a6e9f4d,,,\n'
        'INFO,Fit,2024-05-14 11:29:00.649735,GaussianCopulaSynthesizer,'
        'GaussianCopulaSynthesizer_1.12.1_5387a6e9f4d,1,500,9\n'
        'INFO,Sample,2024-05-14 11:29:00.649735,GaussianCopulaSynthesizer,'
        'GaussianCopulaSynthesizer_1.12.1_5387a6e9f4d,1,500,6\n'
    )

    # Run
    log_dataframe = load_logfile_dataframe(logfile)

    # Assert
    expected_log = pd.DataFrame({
        'LEVEL': ['INFO'] * 3,
        'EVENT': ['Instance', 'Fit', 'Sample'],
        'TIMESTAMP': ['2024-05-14 11:29:00.649735'] * 3,
        'SYNTHESIZER CLASS NAME': ['GaussianCopulaSynthesizer'] * 3,
        'SYNTHESIZER ID': ['GaussianCopulaSynthesizer_1.12.1_5387a6e9f4d'] * 3,
        'TOTAL NUMBER OF TABLES': [np.nan, 1, 1],
        'TOTAL NUMBER OF ROWS': [np.nan, 500, 500],
        'TOTAL NUMBER OF COLUMNS': [np.nan, 9, 6],
    })
    pd.testing.assert_frame_equal(log_dataframe, expected_log)
