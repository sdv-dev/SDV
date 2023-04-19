"""Miscellaneous utility functions."""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.core.tools.datetimes import _guess_datetime_format_for_array


def cast_to_iterable(value):
    """Return a ``list`` if the input object is not a ``list`` or ``tuple``."""
    if isinstance(value, (list, tuple)):
        return value

    return [value]


def display_tables(tables, max_rows=10, datetime_fmt='%Y-%m-%d %H:%M:%S', row=True):
    """Display mutiple tables side by side on a Jupyter Notebook.

    Args:
        tables (dict[str, DataFrame]):
            ``dict`` containing table names and pandas DataFrames.
        max_rows (int):
            Max rows to show per table. Defaults to 10.
        datetime_fmt (str):
            Format with which to display datetime columns.
    """
    # Import here to avoid making IPython a hard dependency
    from IPython.core.display import HTML

    names = []
    data = []
    for name, table in tables.items():
        table = table.copy()
        for column in table.columns:
            column_data = table[column]
            if column_data.dtype.kind == 'M':
                table[column] = column_data.dt.strftime(datetime_fmt)

        names.append(f'<td style="text-align:left"><b>{name}</b></td>')
        data.append(f'<td>{table.head(max_rows).to_html(index=False)}</td>')

    if row:
        html = f"<table><tr>{''.join(names)}</tr><tr>{''.join(data)}</tr></table>"
    else:
        rows = [
            f'<tr>{name}</tr><tr>{table}</tr>'
            for name, table in zip(names, data)
        ]
        html = f"<table>{''.join(rows)}</table>"

    return HTML(html)


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
    if isinstance(value, pd.Series):
        value = value.astype(str).to_list()
    if not isinstance(value, (list, np.ndarray)):
        value = [value]

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
    if isinstance(value, (np.ndarray, pd.Series, list)):
        value = value[0]

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


def validate_datetime_format(value, datetime_format):
    """Determine that value matches datetime format.

    Args:
        value (int, str, datetime, bool):
            Input to evaluate.
        datetime_format (str):
            The datetime format.

    Returns:
        bool:
            True if the input matches the format, False if not.
    """
    pandas_datetime_format = datetime_format.replace('%-', '%')

    try:
        pd.to_datetime(value, format=pandas_datetime_format)

    except ValueError:
        return False

    return True


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
