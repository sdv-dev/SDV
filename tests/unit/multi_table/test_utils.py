"""Test the multi-table data utility functions."""

import re
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import call, patch

import pandas as pd
import pytest

from sdv.multi_table.utils import load_from_csv, validate_file_exists


def test_validate_file_exists():
    """Test that no error is raised when the file exists."""
    # Setup
    data = pd.DataFrame({'numbers': [1, 2, 3]})

    # Run
    with TemporaryDirectory() as temp_dir:
        filepath = Path(temp_dir) / 'mydata.csv'
        data.to_csv(filepath)
        validate_file_exists(f'{temp_dir}/mydata.csv', 'table')


def test_validate_file_exists_file_missing():
    """Test that an error is raised when the file does not exist."""
    # Run and assert
    error_message = re.escape(
        "No data found for table 'table'. Please check the filepath ('mydata.csv')."
    )
    with pytest.raises(ValueError, match=error_message):
        validate_file_exists('mydata.csv', 'table')


@patch('sdv.multi_table.utils.validate_file_exists')
@patch('sdv.multi_table.utils.load_data_from_csv')
def test_load_from_csv(load_mock, validate_mock):
    """Test that each dataframe is loaded based on the filepaths."""
    # Setup
    validate_mock.return_value = None
    table1 = pd.DataFrame({'numbers': [1, 2, 3]})
    table2 = pd.DataFrame({'names': ['John', 'Jane']})
    load_mock.side_effect = [table1, table2]

    # Run
    data = load_from_csv({'numbers': 'numbers.csv', 'names': 'names.csv'})

    # Assert
    pd.testing.assert_frame_equal(table1, data['numbers'])
    pd.testing.assert_frame_equal(table2, data['names'])
    validate_mock.assert_has_calls([
        call('numbers.csv', 'numbers'),
        call('names.csv', 'names')
    ])
