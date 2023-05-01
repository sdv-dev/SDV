"""Constraint utility functions."""

from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd


def cast_to_datetime64(value):
    """Cast a given value to a ``numpy.datetime64`` format.

    Args:
        value (pandas.Series, np.ndarray, list, or str):
            Input data to convert to ``numpy.datetime64``.

    Return:
        ``numpy.datetime64`` value or values.
    """
    if isinstance(value, str):
        value = pd.to_datetime(value).to_datetime64()
    elif isinstance(value, pd.Series):
        value.apply(lambda x: pd.to_datetime(x).to_datetime64())
        value = value.astype('datetime64[ns]')
    elif isinstance(value, (np.ndarray, list)):
        value = np.array([
            pd.to_datetime(item).to_datetime64()
            for item in value
        ])

    return value


def matches_datetime_format(value, datetime_format):
    """Check if datetime value matches the provided format.

    Args:
        value (str):
            The datetime value.
        datetime_format (str):
            The datetime format to check for.

    Return:
        True if the value matches the format. Otherwise False.
    """
    try:
        datetime.strptime(value, datetime_format)
    except Exception:
        return False

    return True


def _cast_to_type(data, dtype):
    if isinstance(data, pd.Series):
        data = data.apply(dtype)
    elif isinstance(data, (np.ndarray, list)):
        data = np.array([dtype(value) for value in data])
    else:
        data = dtype(data)

    return data


def logit(data, low, high):
    """Apply a logit function to the data using ``low`` and ``high``.

    Args:
        data (pd.Series, pd.DataFrame, np.array, int, float or datetime):
            Data to apply the logit function to.
        low (pd.Series, np.array, int, float or datetime):
            Low value/s to use when scaling.
        high (pd.Series, np.array, int, float or datetime):
            High value/s to use when scaling.

    Returns:
        Logit scaled version of the input data.
    """
    data = (data - low) / (high - low)
    data = _cast_to_type(data, Decimal)
    data = data * Decimal(0.95) + Decimal(0.025)
    data = _cast_to_type(data, float)
    return np.log(data / (1.0 - data))


def sigmoid(data, low, high):
    """Apply a sigmoid function to the data using ``low`` and ``high``.

    Args:
        data (pd.Series, pd.DataFrame, np.array, int, float or datetime):
            Data to apply the logit function to.
        low (pd.Series, np.array, int, float or datetime):
            Low value/s to use when scaling.
        high (pd.Series, np.array, int, float or datetime):
            High value/s to use when scaling.

    Returns:
        Sigmoid transform of the input data.
    """
    data = 1 / (1 + np.exp(-data))
    data = _cast_to_type(data, Decimal)
    data = (data - Decimal(0.025)) / Decimal(0.95)
    data = _cast_to_type(data, float)
    data = data * (high - low) + low

    return data


def check_nans_row(row):
    """Check for NaNs in a pandas row.

    Outputs a concatenated string of the column names with NaNs.

    Args:
        row (pandas.Series):
            A pandas row.
    """
    columns_with_nans = ''
    for column, value in row.items():
        if pd.isna(value):
            columns_with_nans += f'{column}, '

    columns_with_nans = columns_with_nans.rstrip(', ')
    if columns_with_nans == '':
        return 'None'

    return columns_with_nans


def add_nans_column(table_data, list_column_names):
    """Add a categorical column to the table_data indicating where NaNs are.

    Args:
        table_data (pandas.DataFrame):
            The table data.
        list_column_names (list):
            The list of column names to check for NaNs.
    """
    nan_column_name = '#'.join(list_column_names) + '.nan_component'
    nan_column = table_data[list_column_names].apply(check_nans_row, axis=1)
    if not (nan_column == 'None').all():
        table_data[nan_column_name] = nan_column
        return True
    else:
        return False


def revert_nans_columns(table_data, nan_column_name):
    """Reverts the NaNs in the table_data based on the categorical column.

    Args:
        table_data (pandas.DataFrame):
            The table data.
        nan_column (pandas.Series):
            The categorical columns indicating where the NaNs are.
    """
    combinations = table_data[nan_column_name].unique()
    for combination in combinations:
        if combination != 'None':
            column_names = [column_name.strip() for column_name in combination.split(',')]
            table_data.loc[table_data[nan_column_name] == combination, column_names] = np.nan

    return table_data.drop(columns=nan_column_name)


def get_datetime_diff(high, low, dtype='O'):
    """Calculate the difference between two datetime columns.

    When casting datetimes to float using ``astype``, NaT values are not automatically
    converted to NaN values. This method calculates the difference between the high
    and low column values, preserving missing values as NaNs.

    Args:
        high (numpy.ndarray):
            The high column values.
        low (numpy.ndarray):
            The low column values.

    Returns:
        numpy.ndarray:
            The difference between the high and low column values.
    """
    if dtype == 'O':
        low = cast_to_datetime64(low)
        high = cast_to_datetime64(high)
    diff_column = high - low
    nan_mask = np.isnan(diff_column)
    diff_column = diff_column.astype(np.float64)
    diff_column[nan_mask] = np.nan
    return diff_column
