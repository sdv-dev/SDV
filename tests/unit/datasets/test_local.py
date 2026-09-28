"""Test the functions for loading locally stored data."""

import json
import os.path as op
import re
from unittest.mock import Mock, call, patch

import pandas as pd
import pytest

from sdv.datasets.local import load_csvs, save_csvs


@patch('sdv.datasets.local.warnings')
@patch('sdv.datasets.local._load_data_from_csv')
def test_load_csvs(load_mock, warnings_mock, tmp_path):
    """Test that the function loads only the csv files in a folder.
    If the folder contains files that aren't csvs, they should be ignored.
    """
    # Setup
    users_table = pd.DataFrame({'user_id': [1, 2, 3], 'name': ['a', 'b', 'c']})
    orders_table = pd.DataFrame({'order_id': [1, 2, 3], 'user_id': [1, 2, 2]})
    fake_json = {'tables': ['orders', 'users']}
    users_mock = Mock()
    orders_mock = Mock()

    load_mock.side_effect = lambda file, args: orders_mock if 'orders.csv' in file else users_mock

    # Run
    orders_path = tmp_path / 'orders.csv'
    users_path = tmp_path / 'users.csv'
    users_table.to_csv(users_path)
    orders_table.to_csv(orders_path)
    json_file_path = op.join(tmp_path, 'fake.json')
    with open(json_file_path, 'w') as outfile:
        json.dump(fake_json, outfile)

    csvs = load_csvs(tmp_path)

    # Assert
    assert csvs == {'orders': orders_mock, 'users': users_mock}
    assert call(op.join(tmp_path, 'orders.csv'), None) in load_mock.mock_calls
    assert call(op.join(tmp_path, 'users.csv'), None) in load_mock.mock_calls
    warnings_mock.warn.assert_called_once_with(
        f"Ignoring incompatible files ['fake.json'] in folder '{tmp_path}'."
    )


def test_load_csvs_no_csvs(tmp_path):
    """Test that the function raises an error if there are no csvs in the folder."""
    # Setup
    fake_json = {'tables': ['orders', 'users']}

    # Run and Assert
    json_file_path = tmp_path / 'fake.json'
    with open(json_file_path, 'w') as outfile:
        json.dump(fake_json, outfile)
    error_message = re.escape(
        f"No CSV files exist in '{tmp_path}'. Please make sure your files end in the '.csv' suffix."
    )
    with pytest.raises(ValueError, match=error_message):
        load_csvs(tmp_path)


def test_load_csvs_folder_does_not_exist():
    """Test that the function raises an error if the folder does not exist."""
    # Run and Assert
    error_message = re.escape("The folder 'demo/' cannot be found.")
    with pytest.raises(ValueError, match=error_message):
        load_csvs('demo/')


def test_save_csvs_data_not_dict():
    """Test that ``save_csvs`` raises an error if the data is not a dictionary."""
    # Run and Assert
    expected_message = re.escape(
        "'data' must be a dictionary that maps table names to pandas DataFrames."
    )
    with pytest.raises(ValueError, match=expected_message):
        save_csvs('data', 'folder')


def test_save_csvs_data_not_dict_of_dataframes():
    """Test that ``save_csvs`` raises an error if the data is not a dictionary of dataframes."""
    # Setup
    data = {'parent': 'dataframe', 'child': 'dataframe'}

    # Run and Assert
    expected_message = re.escape(
        "'data' must be a dictionary that maps table names to pandas DataFrames."
    )
    with pytest.raises(ValueError, match=expected_message):
        save_csvs(data, 'folder')


@patch('sdv.datasets.local.os.path.exists')
@patch('sdv.datasets.local.os.makedirs')
def test_save_csvs_folder_does_not_exist(mock_makedirs, mock_exists, tmp_path):
    """Test that ``save_csvs`` creates the folder if it does not exist."""
    # Setup
    mock_exists.return_value = False
    folder = tmp_path / 'data'

    parent_mock = Mock(spec=pd.DataFrame)
    child_mock = Mock(spec=pd.DataFrame)
    data = {'parent': parent_mock, 'child': child_mock}

    # Run
    save_csvs(data, folder)

    # Assert
    mock_makedirs.assert_called_once_with(folder)
    mock_exists.assert_has_calls([
        call(folder),
        call(op.join(folder, 'parent.csv')),
        call(op.join(folder, 'child.csv')),
    ])
    parent_mock.to_csv.assert_called_once_with(op.join(folder, 'parent.csv'))
    child_mock.to_csv.assert_called_once_with(op.join(folder, 'child.csv'))


@patch('sdv.datasets.local.os.path.exists')
def test_save_csvs(mock_exists, tmp_path):
    """Test ``save_csvs``."""
    # Setup
    folder = tmp_path / 'data'
    folder.mkdir()
    mock_exists.side_effect = [True, False, False]

    parent_mock = Mock(spec=pd.DataFrame)
    child_mock = Mock(spec=pd.DataFrame)
    data = {'parent': parent_mock, 'child': child_mock}

    # Run
    save_csvs(data, folder, suffix='-synthetic', to_csv_parameters={'index': False})

    # Assert
    mock_exists.assert_has_calls([
        call(folder),
        call(op.join(folder, 'parent-synthetic.csv')),
        call(op.join(folder, 'child-synthetic.csv')),
    ])
    parent_mock.to_csv.assert_called_once_with(op.join(folder, 'parent-synthetic.csv'), index=False)
    child_mock.to_csv.assert_called_once_with(op.join(folder, 'child-synthetic.csv'), index=False)


def test_save_csvs_existing_files(tmp_path):
    """Test ``save_csvs`` raises an error with the names of the existing files."""
    # Setup
    folder = tmp_path / 'data'
    folder.mkdir()
    (folder / 'parent-synthetic.csv').touch()

    parent = pd.DataFrame(
        data={
            'id': [0, 1, 2, 3, 4],
            'A': [True, True, False, True, False],
            'B': [0.434, 0.312, 0.212, 0.339, 0.491],
        }
    )

    data = {'parent': parent}

    # Run and Assert
    error_message = re.escape(
        f"The following files already exist in '{folder}':\n"
        'parent-synthetic.csv\n'
        'Please remove them or specify a different suffix.'
    )
    with pytest.raises(FileExistsError, match=error_message):
        save_csvs(data, folder, suffix='-synthetic')


def test_save_csvs_existing_files_more_files(tmp_path):
    """Test it errors with a summary of the existing files if more than three existing files."""
    # Setup
    folder = tmp_path / 'data'
    folder.mkdir()
    (folder / 'parent.csv').touch()
    (folder / 'child.csv').touch()
    (folder / 'grandchild.csv').touch()
    (folder / 'grandchild2.csv').touch()

    data = {
        'parent': Mock(spec=pd.DataFrame),
        'child': Mock(spec=pd.DataFrame),
        'grandchild': Mock(spec=pd.DataFrame),
        'grandchild2': Mock(spec=pd.DataFrame),
    }

    # Run and Assert
    error_message = re.escape(
        f"The following files already exist in '{folder}':\n"
        'parent.csv\n'
        'child.csv\n'
        'grandchild.csv\n'
        '+ 1 more files.'
        'Please remove them or specify a different suffix.'
    )
    with pytest.raises(FileExistsError, match=error_message):
        save_csvs(data, folder)
