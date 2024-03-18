"""Utility functions."""
import sys

import pandas as pd

from sdv._utils import (
    _get_relationship_for_child, _get_rows_to_drop, _validate_foreign_keys_not_null)
from sdv.errors import InvalidDataError, SynthesizerInputError


def drop_unknown_references(metadata, data, drop_missing_values=True, verbose=True):
    """Drop rows with unknown foreign keys.

    Args:
        metadata (MultiTableMetadata):
            Metadata of the datasets.
        data (dict):
            Dictionary that maps each table name (string) to the data for that
            table (pandas.DataFrame).
        drop_missing_values (bool):
            Boolean describing whether or not to also drop foreign keys with missing values
            If True, drop rows with missing values in the foreign keys.
            Defaults to True.
        verbose (bool):
            If True, print information about the rows that are dropped.
            Defaults to True.

    Returns:
        dict:
            Dictionary with the dataframes ensuring referential integrity.
    """
    success_message = 'Success! All foreign keys have referential integrity.'
    metadata.validate()
    try:
        metadata.validate_data(data)
        if drop_missing_values:
            _validate_foreign_keys_not_null(metadata, data)

        if verbose:
            message = [success_message, 'No rows were dropped.']
            sys.stdout.write('\n'.join(message))

        return data
    except (InvalidDataError, SynthesizerInputError):
        result = data.copy()
        table_to_idx_to_drop = _get_rows_to_drop(metadata, result)
        for table in metadata.tables:
            idx_to_drop = table_to_idx_to_drop[table]
            result[table] = result[table].drop(idx_to_drop)
            if drop_missing_values:
                relationships = _get_relationship_for_child(metadata.relationships, table)
                for relationship in relationships:
                    child_column = relationship['child_foreign_key']
                    result[table] = result[table].dropna(subset=[child_column])

            if result[table].empty:
                raise InvalidDataError([
                    f"All references in table '{table}' are unknown and must be dropped."
                    'Try providing different data for this table.'
                ])

        if verbose:
            table_names = sorted(metadata.tables)
            summary_table = pd.DataFrame({
                'Table Name': table_names,
                '# Rows (Original)': [len(data[table]) for table in table_names],
                '# Invalid Rows': [
                    len(data[table]) - len(result[table]) for table in table_names
                ],
                '# Rows (New)': [len(result[table]) for table in table_names]
            })
            message = [success_message, 'Summary of the number of rows dropped:']
            message.append(summary_table.to_string(index=False))
            sys.stdout.write('\n'.join(message))

        return result
