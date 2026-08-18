"""Utility methods to compare the real and synthetic data."""

import sys
import warnings

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from sdv._utils import _cast_to_iterable, _check_is_dict_of_dataframes
from sdv.metadata import Metadata

MISSING_VALUE_PLACEHOLDER = '__sdv_missing_value__'


def _validate_referential_integrity_inputs(
    metadata, synthetic_data, table_name, foreign_key_name, num_rows
):
    """Validate the inputs of the ``print_referential_integrity`` method."""
    if not isinstance(metadata, Metadata):
        raise TypeError('metadata must be of Metadata type.')

    _check_is_dict_of_dataframes(synthetic_data, 'synthetic_data')

    if not isinstance(table_name, str):
        raise TypeError('table_name must be a string.')

    foreign_key_names = _cast_to_iterable(foreign_key_name)
    if not all(isinstance(name, str) for name in foreign_key_names):
        raise TypeError('foreign_key_name must be a string or a tuple of strings.')

    if isinstance(num_rows, bool) or not isinstance(num_rows, int):
        raise TypeError("'num_rows' must be an integer greater than 0.")

    if num_rows <= 0:
        raise ValueError("'num_rows' must be an integer greater than 0.")

    if table_name not in metadata.tables:
        raise ValueError(f"table_name: '{table_name}' not found in metadata.")

    if table_name not in synthetic_data:
        raise ValueError(f"table_name: '{table_name}' not found in synthetic_data.")

    for name in foreign_key_names:
        if name not in metadata.tables[table_name].columns:
            raise ValueError(
                f"foreign_key_name: '{name}' not in Metadata for table_name: '{table_name}'."
            )

        if name not in synthetic_data[table_name].columns:
            raise ValueError(f"foreign_key_name: '{name}' not found in synthetic_data.")

    return foreign_key_names


def _get_parent_relationship(metadata, table_name, foreign_key_names):
    """Get the parent table and primary key linked to the given foreign key.

    Args:
        metadata (Metadata):
            The metadata object describing the synthetic data.
        table_name (str):
            The name of the table that contains the foreign key.
        foreign_key_names (list[str]):
            The columns making up the foreign key to look up.

    Returns:
        tuple[str, list[str], list[str]]:
            The parent table name, the columns making up its primary key, and the foreign key
            columns in the order the relationship defines them.
    """
    for relationship in metadata.relationships:
        child_foreign_key = _cast_to_iterable(relationship['child_foreign_key'])
        if table_name == relationship['child_table_name'] and set(child_foreign_key) == set(
            foreign_key_names
        ):
            return (
                relationship['parent_table_name'],
                _cast_to_iterable(relationship['parent_primary_key']),
                child_foreign_key,
            )

    foreign_key = "', '".join(foreign_key_names)
    raise ValueError(
        f"Unable to find a relationship in metadata given table_name: '{table_name}' "
        f"and foreign_key_name: '{foreign_key}'."
    )


def _format_key(key_names, key_values):
    """Format a set of key columns and their values as ``name: value`` pairs."""
    return ', '.join(f'{name}: {value}' for name, value in zip(key_names, key_values))


def print_referential_integrity(
    metadata, synthetic_data, table_name, foreign_key_name, num_rows=10
):
    """Check that referential integrity is met by looking up a few rows.

    A random selection of rows is taken from the table containing the foreign key. For each
    one, the linked row is looked up in the parent table and the outcome is printed.

    Args:
        metadata (Metadata):
            The metadata object describing the synthetic data.
        synthetic_data (dict):
            A dictionary mapping each table name to a pandas DataFrame containing the
            synthetic data for it.
        table_name (str):
            The name of the table that contains the foreign key to check.
        foreign_key_name (str or tuple[str]):
            The column of the foreign key to check. For composite keys, this is a tuple of
            strings.
        num_rows (int):
            The number of rows to check. Defaults to 10.

    Raises:
        TypeError:
            If any of the inputs is not of the expected type.
        ValueError:
            If the table, the columns or the relationship is missing, or if ``num_rows`` is
            not greater than 0.
    """
    foreign_key_names = _validate_referential_integrity_inputs(
        metadata, synthetic_data, table_name, foreign_key_name, num_rows
    )
    parent_table_name, parent_primary_keys, foreign_key_names = _get_parent_relationship(
        metadata, table_name, foreign_key_names
    )

    child_data = synthetic_data[table_name]
    if len(child_data) < num_rows:
        warnings.warn(
            f"The synthetic data contains '{len(child_data)}' rows which is less than "
            f"num_rows: '{num_rows}'. Changing num_rows to '{len(child_data)}'."
        )
        num_rows = len(child_data)

    parent_data = synthetic_data[parent_table_name]
    parent_keys = set(parent_data[parent_primary_keys].itertuples(index=False, name=None))
    child_primary_keys = _cast_to_iterable(metadata.tables[table_name].primary_key or [])

    for _, child_row in child_data.sample(n=num_rows, replace=False).iterrows():
        heading = f'Picking random {table_name} row'
        if child_primary_keys:
            key_values = ', '.join(str(child_row[name]) for name in child_primary_keys)
            heading += f': {key_values}'

        foreign_key_values = tuple(child_row[name] for name in foreign_key_names)
        if any(pd.isna(value) for value in foreign_key_values):
            result = '✅ Foreign key is null; no linked parent row expected'
        elif foreign_key_values in parent_keys:
            found = _format_key(parent_primary_keys, foreign_key_values)
            result = f'✅ Found {parent_table_name} row! {found}'
        else:
            result = f'❌ Unable to find the linked {parent_table_name} row'

        sys.stdout.write(f'{heading}\n{result}\n\n')


def _validate_data(real_data, synthetic_data, table_name, column_names):
    """Validate that both datasets contain the table and columns to check."""
    if not isinstance(table_name, str):
        raise TypeError(f"'table_name' must be a string, got {type(table_name).__name__}.")

    if not isinstance(column_names, list) or not all(
        isinstance(column_name, str) for column_name in column_names
    ):
        raise TypeError("'column_names' must be a list of strings.")

    if not column_names:
        raise ValueError("'column_names' must contain at least one column name.")

    for argument_name, data in [('real_data', real_data), ('synthetic_data', synthetic_data)]:
        if table_name not in data:
            raise ValueError(f"Table '{table_name}' is not present in '{argument_name}'.")

        missing = [column for column in column_names if column not in data[table_name].columns]
        if missing:
            missing_columns = "', '".join(missing)
            raise ValueError(
                f"The columns '{missing_columns}' are not present in table '{table_name}' "
                f"of '{argument_name}'."
            )


def _align_dtypes(real_column, synthetic_column):
    """Make sure data types of columns being evaluated match.

    Args:
        real_column (pd.Series):
            The column of real data.
        synthetic_column (pd.Series):
            The column of synthetic data.

    Returns:
        tuple[pd.Series, pd.Series]:
            The real and synthetic column, cast to a comparable dtype.
    """
    if real_column.dtype == synthetic_column.dtype:
        return real_column, synthetic_column

    if is_numeric_dtype(real_column) and is_numeric_dtype(synthetic_column):
        return real_column.astype('float64'), synthetic_column.astype('float64')

    if is_datetime64_any_dtype(real_column) or is_datetime64_any_dtype(synthetic_column):
        return (
            pd.to_datetime(real_column, errors='coerce'),
            pd.to_datetime(synthetic_column, errors='coerce'),
        )

    return real_column.astype(str), synthetic_column.astype(str)


def _get_combinations(data):
    """Get the set of unique combinations of values in the data."""
    combinations = data.astype('object')
    combinations = combinations.where(combinations.notna(), MISSING_VALUE_PLACEHOLDER)

    return set(combinations.itertuples(index=False, name=None))


def _compute_overlap(real_data, synthetic_data, table_name, column_names):
    """Get the number of combinations shared by both datasets and the percentage they represent.

    Args:
        real_data (dict):
            A dictionary mapping a table name to a pandas DataFrame containing real data.
        synthetic_data (dict):
            A dictionary mapping a table name to a pandas DataFrame containing synthetic data.
        table_name (str):
            The name of the table that contains the columns to check.
        column_names (list[str]):
            The column names to combine.

    Returns:
        tuple[int, float]:
            The number of shared combinations and their percentage of all combinations.
    """
    real_values = real_data[table_name][column_names].copy()
    synthetic_values = synthetic_data[table_name][column_names].copy()
    for column_name in column_names:
        real_values[column_name], synthetic_values[column_name] = _align_dtypes(
            real_values[column_name], synthetic_values[column_name]
        )

    real_combinations = _get_combinations(real_values)
    synthetic_combinations = _get_combinations(synthetic_values)

    num_common = len(real_combinations & synthetic_combinations)
    num_total = len(real_combinations | synthetic_combinations)
    percent = round(num_common / num_total * 100, 2) if num_total else 0.0

    return num_common, percent


def get_combination_overlap(real_data, synthetic_data, table_name, column_names, verbose=True):
    """Calculate the overlap of combinations of column values between real and synthetic data.

    Args:
        real_data (dict):
            A dictionary mapping a table name to a pandas DataFrame containing real data.
        synthetic_data (dict):
            A dictionary mapping a table name to a pandas DataFrame containing synthetic data.
        table_name (str):
            The name of the table that contains the columns to check.
        column_names (list[str]):
            A list of strings representing the column names to check. Combinations of these
            columns will be checked.
        verbose (bool):
            Whether to print out the interpretation of the results. Defaults to ``True``.

    Returns:
        int:
            The number of unique combinations that appear in both the real and synthetic data.

    Raises:
        TypeError:
            If ``table_name`` is not a string or ``column_names`` is not a list of strings.
        ValueError:
            If the table or any of the columns is missing from the data.
    """
    _validate_data(real_data, synthetic_data, table_name, column_names)
    num_common, percent = _compute_overlap(real_data, synthetic_data, table_name, column_names)

    if verbose:
        sys.stdout.write(f'Number of common combinations: {num_common} ({percent}%)\n')
        if num_common == 0:
            sys.stdout.write(
                '✅ The synthetic data does not contain any of the same combinations from the '
                'real data\n'
            )
        elif percent <= 2:
            sys.stdout.write(
                '⚠️ The synthetic data contains a few of the same combinations as the real '
                'data. This might be due to random chance.\n'
            )
        else:
            sys.stdout.write(
                '❌ The synthetic data contains a significant number of the same combinations '
                'as the real data. This might be due to a small number of possible '
                'combinations, a large sample of synthetic data, or a misconfiguration in your '
                'synthesizer.\n'
            )

    return num_common


def get_pii_overlap(real_data, synthetic_data, table_name, pii_column_name, verbose=True):
    """Calculate the overlap of PII values between the real and synthetic data.

    Args:
        real_data (dict):
            A dictionary mapping a table name to a pandas DataFrame containing real data.
        synthetic_data (dict):
            A dictionary mapping a table name to a pandas DataFrame containing synthetic data.
        table_name (str):
            The name of the table that contains the PII column to check.
        pii_column_name (str):
            The name of the column that contains PII values to check.
        verbose (bool):
            Whether to print out the interpretation of the results. Defaults to ``True``.

    Returns:
        int:
            The number of unique PII values that appear in both the real and synthetic data.

    Raises:
        TypeError:
            If ``table_name`` or ``pii_column_name`` is not a string.
        ValueError:
            If the table or the column is missing from the data.
    """
    if not isinstance(pii_column_name, str):
        raise TypeError(
            f"'pii_column_name' must be a string, got {type(pii_column_name).__name__}."
        )

    column_names = [pii_column_name]
    _validate_data(real_data, synthetic_data, table_name, column_names)
    num_common, percent = _compute_overlap(real_data, synthetic_data, table_name, column_names)

    if verbose:
        sys.stdout.write(f'Number of common data points: {num_common} ({percent}%)\n')
        if num_common == 0:
            sys.stdout.write(
                '✅ The synthetic data does not contain any PII values from the real data\n'
            )
        elif percent <= 2:
            sys.stdout.write(
                '⚠️ The synthetic data contains a few PII values from the real data. '
                'This might be due to random chance.\n'
            )
        else:
            sys.stdout.write(
                '❌ The synthetic data contains a significant number of the same PII values of '
                'as the real data. This might be due to a small number of possible PII values, '
                'a large sample of synthetic data, or a misconfiguration in your synthesizer.\n'
            )

    return num_common
