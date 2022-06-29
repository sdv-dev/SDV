"""Constraint utility functions."""

from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
from pandas.core.tools.datetimes import _guess_datetime_format_for_array


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
