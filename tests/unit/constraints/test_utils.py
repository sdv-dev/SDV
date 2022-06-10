"""Tests for the sdv.constraints.utils module."""
from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd

from sdv.constraints.utils import _cast_to_type, is_datetime_type, logit, sigmoid


def test_is_datetime_type_with_datetime_series():
    """Test the ``is_datetime_type`` function when a datetime series is passed.

    Expect to return True when a datetime series is passed.

    Input:
    - A pandas.Series of type `datetime64[ns]`
    Output:
    - True
    """
    # Setup
    data = pd.Series([
        pd.to_datetime('2020-01-01'),
        pd.to_datetime('2020-01-02'),
        pd.to_datetime('2020-01-03')],
    )

    # Run
    is_datetime = is_datetime_type(data)

    # Assert
    assert is_datetime


def test_is_datetime_type_with_datetime():
    """Test the ``is_datetime_type`` function when a datetime is passed.

    Expect to return True when a datetime variable is passed.

    Input:
    - datetime.Datetime
    Output:
    - True
    """
    # Setup
    data = datetime(2020, 1, 1)

    # Run
    is_datetime = is_datetime_type(data)

    # Assert
    assert is_datetime


def test_is_datetime_type_with_pandas_datetime():
    """Test the ``is_datetime_type`` function when a pandas.datetime is passed.

    Expect to return True when a datetime variable is passed.

    Input:
    - pandas.Datetime
    Output:
    - True
    """
    # Setup
    data = pd.to_datetime('2020-01-01')

    # Run
    is_datetime = is_datetime_type(data)

    # Assert
    assert is_datetime


def test_is_datetime_type_with_int():
    """Test the ``is_datetime_type`` function when an int is passed.

    Expect to return False when an int variable is passed.

    Input:
    - int
    Output:
    - False
    """
    # Setup
    data = 2

    # Run
    is_datetime = is_datetime_type(data)

    # Assert
    assert is_datetime is False


def test_is_datetime_type_with_string():
    """Test the ``is_datetime_type`` function when a string is passed.

    Expect to return False when a string variable is passed.

    Input:
    - string
    Output:
    - False
    """
    # Setup
    data = 'test'

    # Run
    is_datetime = is_datetime_type(data)

    # Assert
    assert is_datetime is False


def test_is_datetime_type_with_int_series():
    """Test the ``is_datetime_type`` function when an int series is passed.

    Expect to return False when an int series variable is passed.

    Input:
    -  pd.Series of type int
    Output:
    - False
    """
    # Setup
    data = pd.Series([1, 2, 3, 4])

    # Run
    is_datetime = is_datetime_type(data)

    # Assert
    assert is_datetime is False


def test__cast_to_type():
    """Test the ``_cast_to_type`` function.

    Given ``pd.Series``, ``np.array`` or just a numeric value, it should
    cast it to the given ``type``.

    Input:
        - pd.Series
        - np.array
        - numeric
        - Type
    Output:
        The values should be casted to the expected ``type``.
    """
    # Setup
    value = 88
    series = pd.Series([1, 2, 3])
    array = np.array([1, 2, 3])

    # Run
    res_value = _cast_to_type(value, float)
    res_series = _cast_to_type(series, float)
    res_array = _cast_to_type(array, float)

    # Assert
    assert isinstance(res_value, float)
    assert res_series.dtype == float
    assert res_array.dtype == float


def test_logit():
    """Test the ``logit`` function.

    Setup:
        - Compute ``expected_res`` with the ``high`` and ``low`` values.
    Input:
        - ``data`` a number.
        - ``low`` and ``high`` numbers.
    Output:
        The result of the scaled logit.
    """
    # Setup
    high, low = 100, 49
    _data = (88 - low) / (high - low)
    _data = Decimal(_data) * Decimal(0.95) + Decimal(0.025)
    _data = float(_data)
    expected_res = np.log(_data / (1.0 - _data))

    data = 88

    # Run
    res = logit(data, low, high)

    # Assert

    assert res == expected_res


def test_sigmoid():
    """Test the ``sigmoid`` function.

    Setup:
        - Compute ``expected_res`` with the ``high`` and ``low`` values.
    Input:
        - ``data`` a number.
        - ``low`` and ``high`` numbers.
    Output:
        The result of sigmoid.
    """
    # Setup
    high, low = 100, 49
    _data = data = 1.1064708752806303

    _data = 1 / (1 + np.exp(-data))
    _data = (Decimal(_data) - Decimal(0.025)) / Decimal(0.95)
    _data = float(_data)
    expected_res = _data * (high - low) + low

    # Run
    res = sigmoid(data, low, high)

    # Assert
    assert res == expected_res
