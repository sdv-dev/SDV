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


def get_first_non_nan_value(input_value):
    """Return the first not ``nan`` value when possible.

    Convert to ``pandas.Series`` if the ``input_value`` is not already. This helps to detect
    easier the ``nan`` values. We filter the values that are ``nans`` since pandas does not
    detect them properly in their ``_guess_datetime_format_for_array``. Also there is a bug in
    ``pandas`` that does not support ``numpy.str_`` data type, that is why we use
    ``pandas.Series`` and convert the data type to ``string`` and then to ``numpy.ndarray``.

    Args:
       input_value (pandas.Series, np.ndarray, list, or str):
            Input to return the first non ``nan`` value.

    Returns:
        str or ``nan``:
            Returns either the first valid value or ``nan``.
    """
    value = input_value
    if not isinstance(value, pd.Series):
        value = pd.Series(input_value)

    value = value[~value.isna()]
    value = value.astype(str).to_numpy()
    if len(value):
        return value[0]

    if isinstance(input_value, Iterable) and not isinstance(input_value, str):
        return input_value[0]

    return input_value


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

    Args:
        value (pandas.DataFrame, int, str or datetime):
            Input to evaluate.

    Returns:
        bool:
            True if the input is a datetime type, False if not.
    """
    if isinstance(value, Iterable) and not isinstance(value, str):
        value = get_first_non_nan_value(value)

    return (
        pd.api.types.is_datetime64_any_dtype(value)
        or isinstance(value, pd.Timestamp)
        or isinstance(value, datetime)
        or bool(get_datetime_format([value]))
    )


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
