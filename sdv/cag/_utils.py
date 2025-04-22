import numpy as np
import pandas as pd

from sdv.cag._errors import PatternNotMetError
from sdv.metadata import Metadata


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
        raise PatternNotMetError(f"Table '{table_name}' is missing columns '{missing_columns}'.")


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
        raise PatternNotMetError(
            'Metadata contains more than 1 table but no ``table_name`` provided.'
        )
    if table_name is None:
        table_name = metadata._get_single_table_name()
    elif table_name not in metadata.tables:
        raise PatternNotMetError(f"Table '{table_name}' missing from metadata.")

    _validate_columns_in_metadata(table_name, columns, metadata)


def _validate_table_name_if_defined(table_name):
    """Validate if the table name is defined, it is a string."""
    if table_name and not isinstance(table_name, str):
        raise ValueError('`table_name` must be a string or None.')


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
    for column in columns_to_drop:
        primary_key = metadata['tables'][table_name].get('primary_key')
        if primary_key and primary_key == column:
            raise ValueError('Cannot remove primary key from Metadata')
        del metadata['tables'][table_name]['columns'][column]

    metadata['tables'][table_name]['column_relationships'] = [
        rel
        for rel in metadata['tables'][table_name].get('column_relationships', [])
        if set(rel['column_names']).isdisjoint(columns_to_drop)
    ]
    return Metadata.load_from_dict(metadata)


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
        if table != table_name
    }
