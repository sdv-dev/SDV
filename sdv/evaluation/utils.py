"""Utility methods to compare the real and synthetic data."""

import sys
import warnings

import pandas as pd

from sdv._utils import _cast_to_iterable, _check_is_dict_of_dataframes
from sdv.metadata import Metadata


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
