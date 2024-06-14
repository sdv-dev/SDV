"""Methods to load local datasets."""

import os
import warnings

import pandas as pd

from sdv._utils import _load_data_from_csv


def load_csvs(folder_name, read_csv_parameters=None):
    """Load csv files from specified folder.

    Args:
        folder_name (str):
            The full path of the folder with the data to be loaded.
        read_csv_parameters (dict):
            A python dictionary of with string and value accepted by ``pandas.read_csv``
            function. Defaults to ``None``.
    """
    if not os.path.exists(folder_name):
        raise ValueError(f"The folder '{folder_name}' cannot be found.")

    dirpath, _, filenames = list(os.walk(folder_name))[0]
    csvs = {}
    other_files = []
    for filename in filenames:
        base_name, ext = os.path.splitext(filename)
        if ext == '.csv':
            filepath = os.path.join(dirpath, filename)
            csvs[base_name] = _load_data_from_csv(filepath, read_csv_parameters)
        else:
            other_files.append(filename)

    if other_files:
        warnings.warn(f"Ignoring incompatible files {other_files} in folder '{folder_name}'.")

    if not csvs:
        raise ValueError(
            f"No CSV files exist in '{folder_name}'. Please make sure your files end in the "
            "'.csv' suffix."
        )

    return csvs


def save_csvs(data, folder_name, suffix=None, to_csv_parameters=None):
    """Save dataframes to csv files in a specified folder.

    Args:
        data (dict):
            A dictionary that maps each table name (string) to the data for that
            table (pandas.DataFrame).
        folder_name (str):
            The full path of the folder where the data will be saved.
        suffix (str):
            A string to be appended to the name of the file. Defaults to ``None``.
        to_csv_parameters (dict):
            A python dictionary of with string and value accepted by ``pandas.DataFrame.to_csv``
            function. Defaults to ``None``.
    """
    error_message_data = "'data' must be a dictionary that maps table names to pandas DataFrames."
    if not isinstance(data, dict):
        raise ValueError(error_message_data)

    for table_name, table in data.items():
        if not isinstance(table, pd.DataFrame):
            raise ValueError(error_message_data)

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    table_name_to_filepath = {}
    errors = []
    for table_name in data:
        filename = f'{table_name}{suffix}.csv' if suffix else f'{table_name}.csv'
        filepath = os.path.join(folder_name, filename)
        if os.path.exists(filepath):
            errors.append(filename)

        table_name_to_filepath[table_name] = filepath

    if errors:
        end_message = 'Please remove them or specify a different suffix.'
        filename_to_print = '\n'.join(errors[:3])
        if len(errors) > 3:
            end_message = ''.join([f'+ {len(errors) - 3} more files.', end_message])

        raise FileExistsError(
            f"The following files already exist in '{folder_name}':\n{filename_to_print}"
            f'\n{end_message}'
        )

    to_csv_parameters = to_csv_parameters or {}
    for table_name, filepath in table_name_to_filepath.items():
        table = data[table_name]
        table.to_csv(filepath, **to_csv_parameters)
