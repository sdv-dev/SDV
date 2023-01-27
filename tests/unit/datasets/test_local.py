"""Test the functions for loading locally stored data."""

import json
import os.path as op
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, call, patch

import pandas as pd
import pytest

from sdv.datasets.local import load_csvs


@patch('sdv.datasets.local.warnings')
@patch('sdv.datasets.local.load_data_from_csv')
def test_load_csvs(load_mock, warnings_mock):
    """Test that the function loads only the csv files in a folder.
    If the folder contains files that aren't csvs, they should be ignored.
    """
    # Setup
    users_table = pd.DataFrame({
        'user_id': [1, 2, 3],
        'name': ['a', 'b', 'c']
    })
    orders_table = pd.DataFrame({
        'order_id': [1, 2, 3],
        'user_id': [1, 2, 2]
    })
    fake_json = {'tables': ['orders', 'users']}
    users_mock = Mock()
    orders_mock = Mock()

    load_mock.side_effect = lambda file: orders_mock if 'orders.csv' in file else users_mock

    # Run
    with TemporaryDirectory() as temp_dir:
        orders_path = Path(temp_dir) / 'orders.csv'
        users_path = Path(temp_dir) / 'users.csv'
        users_table.to_csv(users_path)
        orders_table.to_csv(orders_path)
        json_file_path = op.join(temp_dir, 'fake.json')
        with open(json_file_path, 'w') as outfile:
            json.dump(fake_json, outfile)

        csvs = load_csvs(temp_dir)

    # Assert
    assert csvs == {
        'orders': orders_mock,
        'users': users_mock
    }
    assert call(op.join(temp_dir, 'orders.csv')) in load_mock.mock_calls
    assert call(op.join(temp_dir, 'users.csv')) in load_mock.mock_calls
    warnings_mock.warn.assert_called_once_with(
        f"Ignoring incompatible files ['fake.json'] in folder '{temp_dir}'.")


@patch('sdv.datasets.local.load_data_from_csv')
def test_load_csvs_no_csvs(load_mock):
    """Test that the function raises an error if there are no csvs in the folder."""
    # Setup
    fake_json = {'tables': ['orders', 'users']}

    # Run and Assert
    with TemporaryDirectory() as temp_dir:
        json_file_path = op.join(temp_dir, 'fake.json')
        with open(json_file_path, 'w') as outfile:
            json.dump(fake_json, outfile)
        error_message = re.escape(
            f"No CSV files exist in '{temp_dir}'. Please make sure your files end in the "
            "'.csv' suffix."
        )
        with pytest.raises(ValueError, match=error_message):
            load_csvs(temp_dir)


def test_load_csvs_folder_does_not_exist():
    """Test that the function raises an error if the folder does not exist."""
    # Run and Assert
    error_message = re.escape("The folder 'demo/' cannot be found.")
    with pytest.raises(ValueError, match=error_message):
        load_csvs('demo/')
