"""Constraint utility functions."""

import re
import warnings
from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd

PRECISION_LEVELS = {
    '%Y': 1,  # Year
    '%y': 1,  # Year without century (same precision as %Y)
    '%B': 2,  # Full month name
    '%b': 2,  # Abbreviated month name
    '%m': 2,  # Month as a number
    '%d': 3,  # Day of the month
    '%j': 3,  # Day of the year
    '%U': 3,  # Week number (Sunday-starting)
    '%W': 3,  # Week number (Monday-starting)
    '%A': 3,  # Full weekday name
    '%a': 3,  # Abbreviated weekday name
    '%w': 3,  # Weekday as a decimal
    '%H': 4,  # Hour (24-hour clock)
    '%I': 4,  # Hour (12-hour clock)
    '%M': 5,  # Minute
    '%S': 6,  # Second
    '%f': 7,  # Microsecond
    # Formats that don't add precision
    '%p': 0,  # AM/PM
    '%z': 0,  # UTC offset
    '%Z': 0,  # Time zone name
    '%c': 0,  # Locale-based date/time
    '%x': 0,  # Locale-based date
    '%X': 0,  # Locale-based time
}


def cast_to_datetime64(value, datetime_format=None, ignore_timezone=True):
    """Cast a given value to a ``numpy.datetime64`` format.

    Args:
        value (pandas.Series, np.ndarray, list, or str):
            Input data to convert.
        datetime_format (str, optional):
            Datetime format of the `value`.
        ignore_timezone (bool):
            If True, strips `%z` or `%Z` from the format and removes tzinfo.

    Returns:
        numpy.datetime64, pandas.Series, or numpy.ndarray of datetime64
    """
    if datetime_format:
        datetime_format = datetime_format.replace('%#', '%').replace('%-', '%')

    if isinstance(value, str):
        return _parse_datetime64_value(value, datetime_format, ignore_timezone)

    elif isinstance(value, pd.Series):
        dt_series = _parse_datetime(value, datetime_format, ignore_timezone)
        return dt_series.astype('datetime64[ns]')

    elif isinstance(value, (np.ndarray, list)):
        return np.array([
            _parse_datetime64_value(val, datetime_format, ignore_timezone) for val in value
        ])


def _parse_datetime64_value(value, datetime_format=None, ignore_timezone=True):
    """Parse a single value into `datetime64`, optionally ignoring timezone."""
    if pd.isna(value):
        return pd.NaT.to_datetime64()

    return _parse_datetime(value, datetime_format, ignore_timezone).to_datetime64()


def _parse_datetime(value, datetime_format, ignore_timezone):
    is_series = isinstance(value, pd.Series)
    parsed_value = pd.to_datetime(value, format=datetime_format, errors='coerce')

    if is_series and ignore_timezone and hasattr(parsed_value, 'dt'):
        if hasattr(parsed_value.dt, 'tz_localize'):
            parsed_value = parsed_value.dt.tz_localize(None)

    elif ignore_timezone and hasattr(parsed_value, 'tz_localize'):
        if isinstance(parsed_value, (list, tuple, pd.Series, np.ndarray)):
            parsed_value = [
                new_value.replace(tzinfo=None)
                if isinstance(new_value, datetime)
                else new_value.tz_localize(None)
                for new_value in parsed_value
            ]

        else:
            parsed_value = parsed_value.tz_localize(None)

    if is_series and not isinstance(parsed_value, pd.Series):
        return pd.Series(parsed_value)

    return parsed_value


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


def get_nan_component_value(row):
    """Check for NaNs in a pandas row.

    Outputs a concatenated string of the column names with NaNs.

    Args:
        row (pandas.Series):
            A pandas row.

    Returns:
        A concatenated string of the column names with NaNs.
    """
    columns_with_nans = []
    for column, value in row.items():
        if pd.isna(value):
            columns_with_nans.append(column)

    if columns_with_nans:
        return ', '.join(columns_with_nans)

    return 'None'


def compute_nans_column(table_data, list_column_names):
    """Compute a categorical column to the table_data indicating where NaNs are.

    Args:
        table_data (pandas.DataFrame):
            The table data.
        list_column_names (list):
            The list of column names to check for NaNs.

    Returns:
        A dict with the column name as key and the column indicating where NaNs are as value.
        Empty dict if there are no NaNs.
    """
    nan_column_name = '#'.join(list_column_names) + '.nan_component'
    column = table_data[list_column_names].apply(get_nan_component_value, axis=1)
    if not (column == 'None').all():
        return pd.Series(column, name=nan_column_name)

    return None


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
        if not pd.isna(combination) and combination != 'None':
            column_names = [column_name.strip() for column_name in combination.split(',')]
            table_data.loc[table_data[nan_column_name] == combination, column_names] = np.nan

    return table_data.drop(columns=nan_column_name)


def get_datetime_diff(high, low, high_datetime_format=None, low_datetime_format=None, dtype='O'):
    """Calculate the difference between two datetime columns.

    When casting datetimes to float using ``astype``, NaT values are not automatically
    converted to NaN values. This method calculates the difference between the high
    and low column values, preserving missing values as NaNs.

    Args:
        high (numpy.ndarray):
            The high column values.
        low (numpy.ndarray):
            The low column values.
        high_datetime_format (str):
            Datetime format of the `high` column.
        low_datetime_format (str):
            Datetime format of the `low` column.

    Returns:
        numpy.ndarray:
            The difference between the high and low column values.
    """
    if dtype == 'O':
        low = cast_to_datetime64(low, low_datetime_format)
        high = cast_to_datetime64(high, high_datetime_format)

        if low_datetime_format != high_datetime_format:
            low, high = match_datetime_precision(
                low=low,
                high=high,
                low_datetime_format=low_datetime_format,
                high_datetime_format=high_datetime_format,
            )

    diff_column = high - low
    nan_mask = pd.isna(diff_column)
    diff_column = diff_column.astype(np.float64)
    diff_column[nan_mask] = np.nan
    return diff_column


def get_mappable_combination(combination):
    """Get a mappable combination of values.

    This function replaces NaN values with None inside the tuple
    to ensure consistent comparisons when using mapping.

    Args:
        combination (tuple):
            A combination of values.

    Returns:
        tuple:
            A mappable combination of values.
    """
    return tuple(None if pd.isna(x) else x for x in combination)


def match_datetime_precision(low, high, low_datetime_format, high_datetime_format):
    """Match `low` or `high` datetime array to the lower precision format.

    Args:
        low (np.ndarray):
            Array of datetime values for the low column.
        high (np.ndarray):
            Array of datetime values for the high column.
        low_datetime_format (str):
            The datetime format of the `low` column.
        high_datetime_format (str):
            The datetime format of the `high` column.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            Adjusted `low` and `high` arrays where the higher precision format is
            downcasted to the lower precision format.
    """
    lower_precision_format = get_lower_precision_format(low_datetime_format, high_datetime_format)
    if lower_precision_format == high_datetime_format:
        low = downcast_datetime_to_lower_precision(low, lower_precision_format)
    else:
        high = downcast_datetime_to_lower_precision(high, lower_precision_format)

    return low, high


def get_datetime_format_precision(format_str):
    """Return the precision level of a datetime format string."""
    # Find all format codes in the format string
    found_formats = re.findall(r'%[A-Za-z]', format_str)
    found_levels = (
        PRECISION_LEVELS.get(found_format)
        for found_format in found_formats
        if found_format in PRECISION_LEVELS
    )

    return max(found_levels, default=0)


def get_lower_precision_format(primary_format, secondary_format):
    """Compare two datetime format strings and return the one with lower precision.

    Args:
        primary_format (str):
            The first datetime format string to compare.
        low_precision_format (str):
            The second datetime format string to compare.

    Returns:
        str:
            The datetime format string with the lower precision level.
    """
    primary_level = get_datetime_format_precision(primary_format)
    secondary_level = get_datetime_format_precision(secondary_format)
    if primary_level >= secondary_level:
        return secondary_format

    return primary_format


def downcast_datetime_to_lower_precision(data, target_format):
    """Convert a datetime string from a higher-precision format to a lower-precision format.

    Args:
        data (np.array):
            The data to cast to the `target_format`.
        target_format (str):
            The datetime string to downcast.

    Returns:
        str: The datetime string in the lower precision format.
    """
    downcasted_data = format_datetime_array(data, target_format)
    return cast_to_datetime64(downcasted_data, target_format)


def format_datetime_array(datetime_array, target_format):
    """Format each element in a numpy datetime64 array to a specified string format.

    Args:
        datetime_array (np.ndarray):
            Array of datetime64[ns] elements.
        target_format (str):
            The datetime format to cast each element to.

    Returns:
        np.ndarray: Array of formatted datetime strings.
    """
    return np.array([
        pd.to_datetime(date).strftime(target_format) if not pd.isna(date) else pd.NaT
        for date in datetime_array
    ])


def _warn_if_timezone_aware_formats(formats):
    if any(dt_format and ('%z' in dt_format or '%Z' in dt_format) for dt_format in formats):
        warnings.warn(
            'Timezone information in datetime formats will be ignored when evaluating '
            'constraints. All datetime values will be treated as naive (timezone-unaware). '
            'Support for timezone-aware constraints will be added in a future release.',
            UserWarning,
        )
