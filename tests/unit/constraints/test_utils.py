"""Tests for the sdv.constraints.utils module."""

from decimal import Decimal
from unittest.mock import patch

import numpy as np
import pandas as pd

from sdv.constraints.utils import (
    _cast_to_type,
    cast_to_datetime64,
    compute_nans_column,
    downcast_datetime_to_lower_precision,
    format_datetime_array,
    get_datetime_diff,
    get_datetime_format_precision,
    get_lower_precision_format,
    get_mappable_combination,
    get_nan_component_value,
    logit,
    match_datetime_precision,
    matches_datetime_format,
    revert_nans_columns,
    sigmoid,
)


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


def test_cast_to_datetime64():
    """Test the ``cast_to_datetime64`` function.

    Setup:
        - String value representing a datetime
        - List value with a ``np.nan`` and string values.
        - pd.Series with datetime values.
    Output:
        - A single np.datetime64
        - A list of np.datetime64
        - A series of np.datetime64
    """
    # Setup
    string_value = '2021-02-02'
    list_value = [None, np.nan, '2021-02-02']
    series_value = pd.Series(['2021-02-02', None, pd.NaT])

    # Run
    string_out = cast_to_datetime64(string_value)
    list_out = cast_to_datetime64(list_value)
    series_out = cast_to_datetime64(series_value)

    # Assert
    expected_string_output = np.datetime64('2021-02-02')
    expected_series_output = pd.Series([
        np.datetime64('2021-02-02'),
        np.datetime64('NaT'),
        np.datetime64('NaT'),
    ])
    expected_list_output = np.array(
        [np.datetime64('NaT'), np.datetime64('NaT'), '2021-02-02'], dtype='datetime64[ns]'
    )
    np.testing.assert_array_equal(expected_list_output, list_out)
    pd.testing.assert_series_equal(expected_series_output, series_out)
    assert expected_string_output == string_out


def test_cast_to_datetime64_datetime_format():
    """Test it when `datetime_format` is passed."""
    # Setup
    string_value = '2021-02-02'
    list_value = [None, np.nan, '2021-02-02']
    series_value = pd.Series(['2021-02-02', None, pd.NaT])

    # Run
    string_out = cast_to_datetime64(string_value, datetime_format='%Y-%m-%d')
    list_out = cast_to_datetime64(list_value, datetime_format='%Y-%m-%d')
    series_out = cast_to_datetime64(series_value, datetime_format='%Y-%m-%d')

    # Assert
    expected_string_output = np.datetime64('2021-02-02')
    expected_series_output = pd.Series([
        np.datetime64('2021-02-02'),
        np.datetime64('NaT'),
        np.datetime64('NaT'),
    ])
    expected_list_output = np.array(
        [np.datetime64('NaT'), np.datetime64('NaT'), '2021-02-02'], dtype='datetime64[ns]'
    )
    np.testing.assert_array_equal(expected_list_output, list_out)
    pd.testing.assert_series_equal(expected_series_output, series_out)
    assert expected_string_output == string_out


def test_matches_datetime_format():
    """Test the ``matches_datetime_format`` method.

    If the provided datetime string matches the format, then it should return True.

    Input:
        - Datetime string that matches the format

    Output:
        - True
    """
    # Run
    result = matches_datetime_format('1/1/2020', '%m/%d/%Y')

    # Assert
    assert result is True


def test_matches_datetime_format_does_not_match():
    """Test the ``matches_datetime_format`` method.

    If the provided datetime string does not match the format, then it should return False.

    Input:
        - Datetime string that does not match the format

    Output:
        - False
    """
    # Run
    result = matches_datetime_format('1-1-2020', '%m/%d/%Y')

    # Assert
    assert result is False


def test_matches_datetime_format_bad_value():
    """Test the ``matches_datetime_format`` method.

    If the provided value is not a string, then it should return False.

    Input:
        - int and a datetime format

    Output:
        - False
    """
    # Run
    result = matches_datetime_format(10, '%m/%d/%Y')

    # Assert
    assert result is False


def test_get_nan_component_value():
    """Test the ``get_nan_component_value`` method."""
    # Setup
    row = pd.Series([np.nan, 2, np.nan, 4], index=['a', 'b', 'c', 'd'])

    # Run
    result = get_nan_component_value(row)

    # Assert
    assert result == 'a, c'


def test_compute_nans_columns():
    """Test the ``compute_nans_columns`` method."""
    # Setup
    data = pd.DataFrame({
        'a': [1, np.nan, 3, np.nan],
        'b': [np.nan, 2, 3, np.nan],
        'c': [1, np.nan, 3, np.nan],
    })

    # Run
    output = compute_nans_column(data, ['a', 'b', 'c'])
    expected_output = pd.Series(['b', 'a, c', 'None', 'a, b, c'], name='a#b#c.nan_component')

    # Assert
    pd.testing.assert_series_equal(output, expected_output)


def test_compute_nans_columns_without_nan():
    """Test the ``compute_nans_columns`` method when there are no nans."""
    # Setup
    data = pd.DataFrame({'a': [1, 2, 3, 2], 'b': [2.5, 2, 3, 2.5], 'c': [1, 2, 3, 2]})

    # Run
    output = compute_nans_column(data, ['a', 'b', 'c'])

    # Assert
    assert output is None


def test_revert_nans_columns():
    """Test the ``revert_nans_columns`` method."""
    # Setup
    data = pd.DataFrame({
        'a': [1, 2, 3, 2],
        'b': [2.5, 2, 3, 2.5],
        'c': [1, 2, 3, 2],
        'a#b#c.nan_component': ['b', 'a, c', 'None', 'a, b, c'],
    })
    nan_column_name = 'a#b#c.nan_component'

    # Run
    result = revert_nans_columns(data, nan_column_name)

    expected_data = pd.DataFrame({
        'a': [1, np.nan, 3, np.nan],
        'b': [np.nan, 2, 3, np.nan],
        'c': [1, np.nan, 3, np.nan],
    })

    # Assert
    pd.testing.assert_frame_equal(result, expected_data)


def test_get_datetime_diff():
    """Test the ``_get_datetime_diff`` method.

    The method is expected to compute the difference between the high and low
    datetime columns, treating missing values as NaN.
    """
    # Setup
    high = pd.Series(['2022-02-02', '', '2023-01-02']).to_numpy()
    low = pd.Series(['2022-02-01', '2022-02-02', '2023-01-01']).to_numpy()
    expected = np.array([8.64e13, np.nan, 8.64e13])

    # Run
    diff = get_datetime_diff(high, low, dtype='O')

    # Assert
    assert np.array_equal(expected, diff, equal_nan=True)


def test_get_datetime_diff_with_format_precision_missmatch():
    """Test `get_datetime_diff` with miss matching datetime formats."""
    # Setup
    high = np.array(['2024-11-13 12:00:00.123', '2024-11-13 13:00:00.456'], dtype='O')
    low = np.array(['2024-11-13 12:00:00', '2024-11-13 13:00:00'], dtype='O')
    high_format = '%Y-%m-%d %H:%M:%S.%f'
    low_format = '%Y-%m-%d %H:%M:%S'
    expected_diff = np.array([0.0, 0.0], dtype=np.float64)

    # Run
    result = get_datetime_diff(
        high, low, high_datetime_format=high_format, low_datetime_format=low_format
    )

    # Assert
    np.testing.assert_array_almost_equal(result, expected_diff)


def test_get_mappable_combination():
    """Test the ``get_mappable_combination`` method."""
    # Setup
    already_mappable = ('a', 1, 1.2, 'b')
    not_mappable = ('a', 1, np.nan, 'b')

    # Run
    result_already_mappable = get_mappable_combination(already_mappable)
    result_not_mappable = get_mappable_combination(not_mappable)

    # Assert
    expected_result_not_mappable = ('a', 1, None, 'b')
    assert result_already_mappable == already_mappable
    assert result_not_mappable == expected_result_not_mappable


def test_get_datetime_format_precision_seconds():
    """Test `get_datetime_format_precision` with second-level precision."""
    # Setup
    format_str = '%Y-%m-%d %H:%M:%S'
    expected_precision = 6

    # Run
    result = get_datetime_format_precision(format_str)

    # Assert
    assert result == expected_precision


def test_get_datetime_format_precision_microseconds():
    """Test `get_datetime_format_precision` with microsecond-level precision."""
    # Setup
    format_str = '%Y-%m-%d %H:%M:%S.%f'
    expected_precision = 7

    # Run
    result = get_datetime_format_precision(format_str)

    # Assert
    assert result == expected_precision


def test_get_datetime_format_precision_minutes():
    """Test `get_datetime_format_precision` with minute-level precision."""
    # Setup
    format_str = '%Y-%m-%d %H:%M'
    expected_precision = 5

    # Run
    result = get_datetime_format_precision(format_str)

    # Assert
    assert result == expected_precision


def test_get_datetime_format_precision_days():
    """Test `get_datetime_format_precision` with day-level precision."""
    # Setup
    format_str = '%Y-%m-%d'
    expected_precision = 3

    # Run
    result = get_datetime_format_precision(format_str)

    # Assert
    assert result == expected_precision


def test_get_datetime_format_precision_no_precision():
    """Test `get_datetime_format_precision` with no precision format."""
    # Setup
    format_str = '%Y'
    expected_precision = 1

    # Run
    result = get_datetime_format_precision(format_str)

    # Assert
    assert result == expected_precision


def test_get_datetime_format_precision_mixed_format_higher_precision():
    """Test `get_datetime_format_precision` with mixed higher-precision format."""
    # Setup
    format_str = '%Y-%m-%d %H:%M:%S.%f %z'
    expected_precision = 7

    # Run
    result = get_datetime_format_precision(format_str)

    # Assert
    assert result == expected_precision


def test_get_lower_precision_format_with_different_precision():
    """Test `get_lower_precision_format` with different precision levels."""
    # Setup
    primary_format = '%Y-%m-%d %H:%M:%S'
    secondary_format = '%Y-%m-%d %H:%M:%S.%f'

    # Run
    result = get_lower_precision_format(primary_format, secondary_format)

    # Assert
    assert result == primary_format


def test_get_lower_precision_format_with_equal_precision():
    """Test `get_lower_precision_format` when both formats have the same precision."""
    # Setup
    primary_format = '%Y-%m-%d %H:%M:%S'
    secondary_format = '%Y-%m-%d %H:%M:%S'

    # Run
    result = get_lower_precision_format(primary_format, secondary_format)

    # Assert
    assert result == secondary_format == primary_format


def test_get_lower_precision_format_with_date_only():
    """Test `get_lower_precision_format` with date-only formats."""
    # Setup
    primary_format = '%Y-%m-%d'
    secondary_format = '%Y-%m'

    # Run
    result = get_lower_precision_format(primary_format, secondary_format)

    # Assert
    assert result == secondary_format


def test_get_lower_precision_format_with_week_and_day_formats():
    """Test `get_lower_precision_format` with week and day level formats."""
    # Setup
    primary_format = '%Y-%W'
    secondary_format = '%Y-%m-%d'

    # Run
    result = get_lower_precision_format(primary_format, secondary_format)

    # Assert
    assert result == secondary_format


def test_downcast_datetime_to_lower_precision():
    """Test `downcast_datetime_to_lower_precision` to ensure datetime downcasting."""
    # Setup
    data = np.array(
        ['2024-11-13 12:30:45.123456789', '2024-11-13 13:45:30.987654321'], dtype='datetime64[ns]'
    )
    target_format = '%Y-%m-%d %H:%M:%S'
    expected_result = np.array(['2024-11-13 12:30:45', '2024-11-13 13:45:30'], dtype='O')

    # Run
    result = downcast_datetime_to_lower_precision(data, target_format)

    # Assert
    np.testing.assert_array_equal(result, cast_to_datetime64(expected_result))


def test_downcast_datetime_to_lower_precision_to_day():
    """Test `downcast_datetime_to_lower_precision` to downcast datetime to day precision."""
    # Setup
    data = np.array(
        ['2024-11-13 12:30:45.123456789', '2024-11-14 13:45:30.987654321'], dtype='datetime64[ns]'
    )
    target_format = '%Y-%m-%d'  # Downcasting to day precision
    expected_result = np.array(['2024-11-13', '2024-11-14'], dtype='O')

    # Run
    result = downcast_datetime_to_lower_precision(data, target_format)

    # Assert
    np.testing.assert_array_equal(result, cast_to_datetime64(expected_result))


def test_format_datetime_array_with_lower_precision_format():
    """Test `format_datetime_array` formatting datetime array to a lower-precision format."""
    # Setup
    datetime_array = np.array(
        ['2024-11-13 12:30:45.123456789', '2024-11-13 13:45:30.987654321'], dtype='datetime64[ns]'
    )
    target_format = '%Y-%m-%d %H:%M:%S'
    expected_result = np.array(['2024-11-13 12:30:45', '2024-11-13 13:45:30'], dtype='O')

    # Run
    result = format_datetime_array(datetime_array, target_format)

    # Assert
    np.testing.assert_array_equal(result, expected_result)


@patch('sdv.constraints.utils.downcast_datetime_to_lower_precision')
def test_match_datetime_precision_low_has_higher_precision(mock_downcast):
    """Test `match_datetime_precision` when `low` has higher precision than `high`.

    This test checks that if the `low` array has a more precise format than `high`,
    `low` is downcasted to match the `high` format.
    """
    # Setup
    low = np.array(['2024-11-13 10:34:45.123456', '2024-11-14 12:20:10.654321'], dtype='O')
    high = np.array(['2024-11-13 10:34:45', '2024-11-14 12:20:10'], dtype='O')
    low_format = '%Y-%m-%d %H:%M:%S.%f'
    high_format = '%Y-%m-%d %H:%M:%S'
    expected_low = np.array(['2024-11-13 10:34:45', '2024-11-14 12:20:10'], dtype='O')

    # Set the return value of the mock to simulate downcasting
    mock_downcast.return_value = expected_low

    # Run
    result_low, result_high = match_datetime_precision(low, high, low_format, high_format)

    # Assert
    mock_downcast.assert_called_once_with(low, high_format)
    np.testing.assert_array_equal(result_low, expected_low)
    np.testing.assert_array_equal(result_high, high)


@patch('sdv.constraints.utils.downcast_datetime_to_lower_precision')
def test_match_datetime_precision_high_has_higher_precision(mock_downcast):
    """Test `match_datetime_precision` when `high` has higher precision than `low`.

    This test checks that if the `high` array has a more precise format than `low`,
    `high` is downcasted to match the `low` format.
    """
    # Setup
    low = np.array(['2024-11-13 10:34:45', '2024-11-14 12:20:10'], dtype='O')
    high = np.array(['2024-11-13 10:34:45.123456', '2024-11-14 12:20:10.654321'], dtype='O')
    low_format = '%Y-%m-%d %H:%M:%S'
    high_format = '%Y-%m-%d %H:%M:%S.%f'
    expected_high = np.array(['2024-11-13 10:34:45', '2024-11-14 12:20:10'], dtype='O')

    # Set the return value of the mock to simulate downcasting
    mock_downcast.return_value = expected_high

    # Run
    result_low, result_high = match_datetime_precision(low, high, low_format, high_format)

    # Assert
    mock_downcast.assert_called_once_with(high, low_format)
    np.testing.assert_array_equal(result_low, low)
    np.testing.assert_array_equal(result_high, expected_high)
