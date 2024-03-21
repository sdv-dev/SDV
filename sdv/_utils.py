"""Miscellaneous utility functions."""
import warnings
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd
from pandas.core.tools.datetimes import _guess_datetime_format_for_array

from sdv import version
from sdv.errors import SDVVersionWarning, SynthesizerInputError


def _cast_to_iterable(value):
    """Return a ``list`` if the input object is not a ``list`` or ``tuple``."""
    if isinstance(value, (list, tuple)):
        return value

    return [value]


def _get_datetime_format(value):
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


def _is_datetime_type(value):
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
        value = _cast_to_iterable(value)

    values = pd.Series(value)
    values = values[~values.isna()]
    values = values.head(1000)  # only check 1000 values so this method takes less than 1 second
    for value in values:
        if not (
            bool(_get_datetime_format([value]))
            or isinstance(value, pd.Timestamp)
            or isinstance(value, datetime)
        ):
            return False

    return True


def _is_numerical_type(value):
    """Determine if the input is numerical or not.

    Args:
        value (int, str, datetime, bool):
            Input to evaluate.

    Returns:
        bool:
            True if the input is numerical, False if not.
    """
    return pd.isna(value) | pd.api.types.is_float(value) | pd.api.types.is_integer(value)


def _is_boolean_type(value):
    """Determine if the input is a boolean or not.

    Args:
        value (int, str, datetime, bool):
            Input to evaluate.

    Returns:
        bool:
            True if the input is a boolean, False if not.
    """
    return True if pd.isna(value) | (value is True) | (value is False) else False


def _validate_datetime_format(column, datetime_format):
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


def _convert_to_timedelta(column):
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


def _load_data_from_csv(filepath, read_csv_parameters=None):
    """Load DataFrame from a filepath.

    Args:
        filepath (str):
            String that represents the ``path`` to the ``csv`` file.
        read_csv_parameters (dict):
            A python dictionary of with string and value accepted by ``pandas.read_csv``
            function. Defaults to ``None``.
    """
    filepath = Path(filepath)
    read_csv_parameters = read_csv_parameters or {}
    data = pd.read_csv(filepath, **read_csv_parameters)
    return data


def _groupby_list(list_to_check):
    """Return the first element of the list if the length is 1 else the entire list."""
    return list_to_check[0] if len(list_to_check) == 1 else list_to_check


def _create_unique_name(name, list_names):
    """Modify the ``name`` parameter if it already exists in the list of names."""
    result = name
    while result in list_names:
        result += '_'

    return result


def _format_invalid_values_string(invalid_values, num_values):
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


def _get_root_tables(relationships):
    parent_tables = {rel['parent_table_name'] for rel in relationships}
    child_tables = {rel['child_table_name'] for rel in relationships}
    return parent_tables - child_tables


def _get_relationship_for_child(relationships, child_table):
    return [rel for rel in relationships if rel['child_table_name'] == child_table]


def _get_relationship_for_parent(relationships, parent_table):
    return [rel for rel in relationships if rel['parent_table_name'] == parent_table]


def _get_rows_to_drop(metadata, data):
    """Get the rows to drop to ensure referential integrity.

    The logic of this function is to start at the root tables, look at invalid references
    and then save the index of the rows to drop. Then, we looked at the relationships that
    we didn't check and repeat the process until there are no more relationships to check.
    This ensures that we preserve the referential integrity between all the relationships.

    Args:
        metadata (MultiTableMetadata):
            Metadata of the datasets.
        data (dict):
            Dictionary that maps each table name (string) to the data for that
            table (pandas.DataFrame).

    Returns:
        dict:
            Dictionary with the table names as keys and the indexes of the rows to drop as values.
    """
    table_to_idx_to_drop = defaultdict(set)
    relationships = deepcopy(metadata.relationships)
    while relationships:
        current_roots = _get_root_tables(relationships)
        for root in current_roots:
            parent_table = root
            relationships_parent = _get_relationship_for_parent(relationships, parent_table)
            parent_column = metadata.tables[parent_table].primary_key
            valid_parent_idx = [
                idx for idx in data[parent_table].index
                if idx not in table_to_idx_to_drop[parent_table]
            ]
            valid_parent_values = set(data[parent_table].loc[valid_parent_idx, parent_column])
            for relationship in relationships_parent:
                child_table = relationship['child_table_name']
                child_column = relationship['child_foreign_key']

                is_nan = data[child_table][child_column].isna()
                invalid_values = set(
                    data[child_table].loc[~is_nan, child_column]
                ) - valid_parent_values
                invalid_rows = data[child_table][
                    data[child_table][child_column].isin(invalid_values)
                ]
                idx_to_drop = set(invalid_rows.index)

                if idx_to_drop:
                    table_to_idx_to_drop[child_table] = table_to_idx_to_drop[
                        child_table
                    ].union(idx_to_drop)

            relationships = [rel for rel in relationships if rel not in relationships_parent]

    return table_to_idx_to_drop


def _validate_foreign_keys_not_null(metadata, data):
    """Validate that the foreign keys in the data don't have null values."""
    invalid_tables = defaultdict(list)
    for table_name, table_data in data.items():
        for foreign_key in metadata._get_all_foreign_keys(table_name):
            if table_data[foreign_key].isna().any():
                invalid_tables[table_name].append(foreign_key)

    if invalid_tables:
        err_msg = (
            'The data contains null values in foreign key columns. '
            'This feature is currently unsupported. Please remove '
            'null values to fit the synthesizer.\n'
            '\n'
            'Affected columns:\n'
        )
        for table_name, invalid_columns in invalid_tables.items():
            err_msg += f"Table '{table_name}', column(s) {invalid_columns}\n"

        raise SynthesizerInputError(err_msg)


def check_sdv_versions_and_warn(synthesizer):
    """Check if the current SDV and SDV Enterprise versions mismatch.

    Args:
        synthesizer (BaseSingleTableSynthesizer or BaseMultiTableSynthesizer):
            An SDV model instance to check versions against.

    Raises:
        SDVVersionWarning:
            If the current SDV or SDV Enterprise version does not match the version used to fit
            the synthesizer.
    """
    current_public_version = getattr(version, 'public', None)
    current_enterprise_version = getattr(version, 'enterprise', None)
    if synthesizer._fitted:
        fitted_public_version = getattr(synthesizer, '_fitted_sdv_version', None)
        fitted_enterprise_version = getattr(synthesizer, '_fitted_sdv_enterprise_version', None)
        public_missmatch = current_public_version != fitted_public_version
        enterprise_missmatch = current_enterprise_version != fitted_enterprise_version

        if (public_missmatch or enterprise_missmatch):
            static_message = (
                'The latest bug fixes and features may not be available for this synthesizer. '
                'To see these enhancements, create and train a new synthesizer on this version.'
            )
            if public_missmatch and enterprise_missmatch:
                message = (
                    'You are currently on SDV version '
                    f'{current_public_version} and SDV Enterprise version '
                    f'{current_enterprise_version} but this synthesizer was created on '
                    f'SDV version {synthesizer._fitted_sdv_version} and SDV Enterprise version '
                    f'{synthesizer._fitted_sdv_enterprise_version}.'
                )

            elif public_missmatch:
                message = (
                    'You are currently on SDV version '
                    f'{current_public_version} but this synthesizer was created on '
                    f'version {synthesizer._fitted_sdv_version}.'
                )
            elif enterprise_missmatch:
                message = (
                    'You are currently on SDV Enterprise version '
                    f'{current_enterprise_version} but this synthesizer was created on '
                    f'version {synthesizer._fitted_sdv_enterprise_version}.'
                )

            message = f'{message} {static_message}'
            warnings.warn(message, SDVVersionWarning)
