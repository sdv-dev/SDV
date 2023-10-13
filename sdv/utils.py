"""Miscellaneous utility functions."""
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import pandas as pd
from pandas.core.tools.datetimes import _guess_datetime_format_for_array


def cast_to_iterable(value):
    """Return a ``list`` if the input object is not a ``list`` or ``tuple``."""
    if isinstance(value, (list, tuple)):
        return value

    return [value]


def get_datetime_format(value):
    """Get the ``strftime`` format for a given ``value``.

    This function returns the ``strftime`` format of a given ``value`` when possible.
    If the ``_guess_datetime_format_for_array`` from ``pandas.core.tools.datetimes`` is
    able to detect the ``strftime`` it will return it as a ``string`` if not, a ``None``
    will be returned.

    Args:
        value (pandas.Series, np.ndarray, list, or str):
            Input to attempt detecting the format.

    Return:
        String representing the datetime format in ``strftime`` format or ``None`` if not detected.
    """
    if not isinstance(value, pd.Series):
        value = pd.Series(value)

    value = value[~value.isna()]
    value = value.astype(str).to_numpy()

    return _guess_datetime_format_for_array(value)


def is_datetime_type(value):
    """Determine if the input is a datetime type or not.

    If a ``pandas.Series`` or ``list`` is passed, it will return ``True`` if the first
    thousand values are datetime. Otherwise, it will check if the value is a datetime.

    Note: it will return ``False`` if ``value`` is a string representing
    a date before the year 1677.

    Args:
        value (array-like iterable, int, str or datetime):
            Input to evaluate.

    Returns:
        bool:
            True if the input is a datetime type, False if not.
    """
    if isinstance(value, str) or (not isinstance(value, Iterable)):
        value = cast_to_iterable(value)

    values = pd.Series(value)
    values = values[~values.isna()]
    values = values.head(1000)  # only check 1000 values so this method takes less than 1 second
    for value in values:
        if not (
            bool(get_datetime_format([value]))
            or isinstance(value, pd.Timestamp)
            or isinstance(value, datetime)
        ):
            return False

    return True


def is_numerical_type(value):
    """Determine if the input is numerical or not.

    Args:
        value (int, str, datetime, bool):
            Input to evaluate.

    Returns:
        bool:
            True if the input is numerical, False if not.
    """
    return pd.isna(value) | pd.api.types.is_float(value) | pd.api.types.is_integer(value)


def is_boolean_type(value):
    """Determine if the input is a boolean or not.

    Args:
        value (int, str, datetime, bool):
            Input to evaluate.

    Returns:
        bool:
            True if the input is a boolean, False if not.
    """
    return True if pd.isna(value) | (value is True) | (value is False) else False


def validate_datetime_format(column, datetime_format):
    """Determine the values of the column that match the datetime format.

    Args:
        column (pd.Series):
            Column to evaluate.
        datetime_format (str):
            The datetime format.

    Returns:
        pd.Series:
            Series of booleans, with True if the value matches the format, False if not.
    """
    pandas_datetime_format = datetime_format.replace('%-', '%')
    datetime_column = pd.to_datetime(
        column,
        errors='coerce',
        format=pandas_datetime_format
    )
    valid = pd.isna(column) | ~pd.isna(datetime_column)

    return set(column[~valid])


def convert_to_timedelta(column):
    """Convert a ``pandas.Series`` to one with dtype ``timedelta``.

    ``pd.to_timedelta`` does not handle nans, so this function masks the nans, converts and then
    reinserts them.

    Args:
        column (pandas.Series):
            Column to convert.

    Returns:
        pandas.Series:
            The column converted to timedeltas.
    """
    nan_mask = pd.isna(column)
    column[nan_mask] = 0
    column = pd.to_timedelta(column)
    column[nan_mask] = pd.NaT
    return column


def load_data_from_csv(filepath, pandas_kwargs=None):
    """Load DataFrame from a filepath.

    Args:
        filepath (str):
            String that represents the ``path`` to the ``csv`` file.
        pandas_kwargs (dict):
            A python dictionary of with string and value accepted by ``pandas.read_csv``
            function. Defaults to ``None``.
    """
    filepath = Path(filepath)
    pandas_kwargs = pandas_kwargs or {}
    data = pd.read_csv(filepath, **pandas_kwargs)
    return data


def groupby_list(list_to_check):
    """Return the first element of the list if the length is 1 else the entire list."""
    return list_to_check[0] if len(list_to_check) == 1 else list_to_check


def create_unique_name(name, list_names):
    """Modify the ``name`` parameter if it already exists in the list of names."""
    result = name
    while result in list_names:
        result += '_'

    return result


def format_invalid_values_string(invalid_values, num_values):
    """Convert ``invalid_values`` into a string of invalid values.

    Args:
        invalid_values (pd.DataFrame, set):
            Object of values to be converted into string.
        num_values (int):
            Maximum number of values of the object to show.

    Returns:
        str:
            A stringified version of the object.
    """
    if isinstance(invalid_values, pd.DataFrame):
        if len(invalid_values) > num_values:
            return f'{invalid_values.head(num_values)}\n+{len(invalid_values) - num_values} more'

    if isinstance(invalid_values, set):
        invalid_values = sorted(invalid_values, key=lambda x: str(x))
        if len(invalid_values) > num_values:
            extra_missing_values = [f'+ {len(invalid_values) - num_values} more']
            return f'{invalid_values[:num_values] + extra_missing_values}'

    return f'{invalid_values}'
