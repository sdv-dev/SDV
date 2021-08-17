"""Tests for the sdv.constraints.utils module."""
from datetime import datetime

import pandas as pd

from sdv.constraints.utils import is_datetime_type


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
