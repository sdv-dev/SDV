import importlib
import json
import re
import traceback
import warnings
from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd

from sdv._utils import _cast_to_iterable
from sdv.cag._errors import ConstraintNotMetError
from sdv.errors import RefitWarning, SynthesizerInputError, TableNameError
from sdv.metadata import Metadata

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


def _validate_columns_not_primary_key(table_name, columns, metadata):
    """Validate that none of the columns are in the primary key for the table."""
    primary_key = metadata.tables[table_name].primary_key
    if metadata.tables[table_name]._primary_key_is_composite:
        key_columns = set(primary_key).intersection(set(columns))
        if key_columns:
            pk_columns = "', '".join(sorted(key_columns))
            raise ConstraintNotMetError(
                f"Cannot apply constraint because ['{pk_columns}'] are "
                f"part of the primary key for table '{table_name}'."
            )
    elif primary_key in columns:
        raise ConstraintNotMetError(
            f"Cannot apply constraint because '{primary_key}' is the "
            f"primary key of table '{table_name}'."
        )


def _validate_columns_in_metadata(table_name, columns, metadata):
    """Validates that the columns are in the metadata.

    Args:
        table_name (str):
            The name of the table in the metadata.

        columns (list[str])
            The column names to check.

        metadata (sdv.metadata.Metadata):
            The Metadata to check.
    """
    if not set(columns).issubset(set(metadata.tables[table_name].columns)):
        missing_columns = set(columns) - set(metadata.tables[table_name].columns)
        missing_columns = "', '".join(sorted(missing_columns))
        raise ConstraintNotMetError(f"Table '{table_name}' is missing columns '{missing_columns}'.")


def _validate_table_and_column_names(table_name, columns, metadata):
    """Validate the table name and columns against the metadata.

    It checks the following:
        - If the table name is None, the metadata should only contain a single table.
        - The table name is in the metadata.
        - The columns are in the metadata.

    Args:
        table_name (str):
            The name of the table in the metadata to validate.

        columns (list[str])
            The column names to check.

        metadata (sdv.metadata.Metadata):
            The Metadata to check.
    """
    if table_name is None and len(metadata.tables) > 1:
        raise ConstraintNotMetError(
            'Metadata contains more than 1 table but no ``table_name`` provided.'
        )
    if table_name is None:
        table_name = metadata._get_single_table_name()
    elif table_name not in metadata.tables:
        raise ConstraintNotMetError(f"Table '{table_name}' missing from metadata.")

    _validate_columns_in_metadata(table_name, columns, metadata)


def _validate_table_name_if_defined(table_name):
    """Validate if the table name is defined, it is a string."""
    if table_name and not isinstance(table_name, str):
        raise TableNameError


def _is_list_of_type(values, type_to_check=str):
    """Checks that 'values' is a list and all elements are of type 'type_to_check'."""
    return isinstance(values, list) and all(isinstance(value, type_to_check) for value in values)


def _get_invalid_rows(valid):
    """Determine the indices of the rows where value is False.

    Args:
        valid (pd.Series):
            The input data to check for False values.

    Returns:
        (str): A string that describes the indices where the value is False.
            If there are more than 5 indices, the rest are described as 'more'.
    """
    invalid_rows = np.where(~valid)[0]
    if len(invalid_rows) <= 5:
        invalid_rows_str = ', '.join(str(i) for i in invalid_rows)
    else:
        first_five = ', '.join(str(i) for i in invalid_rows[:5])
        remaining = len(invalid_rows) - 5
        invalid_rows_str = f'{first_five}, +{remaining} more'
    return invalid_rows_str


def _get_is_valid_dict(data, table_name):
    """Create a dictionary of True values for each table besides table_name.

    Besides table_name, all rows of every other table are considered valid,
    so the boolean Series will be True for all rows of every other table.

    Args:
        data (dict):
            The data.
        table_name (str):
            The name of the table to exclude from the dictionary.

    Returns:
        dict:
            Dictionary of table names to boolean Series of True values.
    """
    return {
        table: pd.Series(True, index=table_data.index)
        for table, table_data in data.items()
        if table != table_name or table_name is None
    }


def _convert_to_snake_case(string):
    """Convert a string to snake case (words separated by underscores, all lowercase)."""
    return re.sub(r'([a-z])([A-Z])', r'\1_\2', string).lower()


def _remove_columns_from_metadata(metadata, table_name, columns_to_drop):
    """Remove columns from metadata, including column relationships.

        Will raise an error if the primary key is being dropped.

    Args:
        metadata (dict, sdv.metadata.Metadata): The Metadata which contains
            the columns to drop.
        table_name (str): Name of the table in the metadata, where the column(s)
            are located.
        columns_to_drop (list[str]): The list of column names to drop from the
            Metadata.

    Returns:
        (sdv.metadata.Metadata): The new Metadata, with the columns removed.
    """
    if isinstance(metadata, Metadata):
        metadata = metadata.to_dict()
    column_set = set(columns_to_drop)
    primary_key = _cast_to_iterable(metadata['tables'][table_name].get('primary_key'))
    for column in column_set:
        if primary_key and column in primary_key:
            raise ValueError('Cannot remove primary key from Metadata')
        del metadata['tables'][table_name]['columns'][column]

    metadata['tables'][table_name]['column_relationships'] = [
        rel
        for rel in metadata['tables'][table_name].get('column_relationships', [])
        if set(rel['column_names']).isdisjoint(column_set)
    ]
    return Metadata.load_from_dict(metadata)


def _validate_constraints(constraints, synthesizer_fitted):
    """Validate the constraints.

    Args:
        constraints (list[sdv.cag.BaseConstraint]):
            The list of constraints to validate.
        synthesizer_fitted (bool):
            Whether the synthesizer has been fitted or not.

    Raises:
        ValueError: If the constraints are not valid.
    """
    if not isinstance(constraints, list):
        raise ValueError('Constraints must be a list of sdv.cag objects.')

    if synthesizer_fitted:
        warnings.warn(
            "For these constraints to take effect, please refit the synthesizer using 'fit'.",
            RefitWarning,
        )

    return constraints


def _validate_constraints_single_table(constraints, synthesizer_fitted):
    """Check if the constraints are single table.

    Args:
        constraints (list):
            A list of constraints to check.
    """
    constraints = _validate_constraints(constraints, synthesizer_fitted)
    for constraint in constraints:
        if constraint._is_single_table is False:
            raise SynthesizerInputError(
                f'Constraint `{constraint.__class__.__name__}` is not compatible with the '
                'single table synthesizers.'
            )

    return constraints


def load_constraint_from_dict(constraint_dict):
    """Load a constraint from a constraint dictionary.

    Args:
        constraint_dict (dict):
            A constraint dictionary containing the following keys:
            - `class_name` (str): The constraint class name.
            - `parameters` (dict): Dictionary of the parameters used to instantiate the constraint.

    Returns:
        Instance of `class_name` constraint instantiated with the given `parameters`.
    """
    expected_keys = {'class_name', 'parameters'}
    if not isinstance(constraint_dict, dict) or set(constraint_dict.keys()) != expected_keys:
        raise ValueError(
            'Invalid `constraint_dict`. Expected dictionary with keys `class_name` and '
            f' `parameters`, got {constraint_dict}.'
        )

    class_name = constraint_dict['class_name']
    parameters = constraint_dict['parameters']
    if not isinstance(class_name, str):
        raise ValueError('`class_name` must be a string.')

    if not isinstance(parameters, dict):
        raise ValueError('`parameters` must be a dict.')

    cag_module = importlib.import_module('sdv.cag')
    try:
        sandbox_module = importlib.import_module('sdv.cag.sandbox')
        sandbox_constraint = getattr(sandbox_module, class_name, None)
    except ModuleNotFoundError:
        sandbox_constraint = None

    constraint_class = getattr(cag_module, class_name, sandbox_constraint)
    if constraint_class is None:
        raise ValueError(f"Unknown `constraint_class` '{class_name}'.")

    return constraint_class.load_constraint_from_dict(parameters=parameters)


def _load_constraints_from_file(filepath):
    """Load constraints from a file (JSON).

    Args:
        filepath (str):
            The string path to the file containing the constraints to load.

    Returns:
        list[BaseConstraint]:
            A list of constraint objects.
    """
    with open(filepath, 'r') as f:
        constraints_json = json.load(f)

    constraint_list = []
    for constraint_dict in constraints_json:
        try:
            constraint_list.append(load_constraint_from_dict(constraint_dict))
        except Exception as e:
            warnings.warn(
                f'Could not load constraint ({constraint_dict}):\n'
                f'    {traceback.format_exception_only(type(e), e)[0]}'
            )
    return constraint_list
