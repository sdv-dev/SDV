"""Utility functions."""
import sys
from copy import deepcopy

import pandas as pd

from sdv._utils import _validate_foreign_keys_not_null
from sdv.errors import InvalidDataError, SynthesizerInputError
from sdv.metadata.errors import InvalidMetadataError
from sdv.multi_table.hma import MAX_NUMBER_OF_COLUMNS
from sdv.multi_table.utils import (
    _drop_rows, _get_total_estimated_columns, _print_simplified_schema_summary, _simplify_data,
    _simplify_metadata)


def drop_unknown_references(data, metadata, drop_missing_values=True, verbose=True):
    """Drop rows with unknown foreign keys.

    Args:
        data (dict):
            Dictionary that maps each table name (string) to the data for that
            table (pandas.DataFrame).
        metadata (MultiTableMetadata):
            Metadata of the datasets.
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
    table_names = sorted(metadata.tables)
    summary_table = pd.DataFrame({
        'Table Name': table_names,
        '# Rows (Original)': [len(data[table]) for table in table_names],
        '# Invalid Rows': [0] * len(table_names),
        '# Rows (New)': [len(data[table]) for table in table_names]
    })
    metadata.validate()
    try:
        metadata.validate_data(data)
        if drop_missing_values:
            _validate_foreign_keys_not_null(metadata, data)

        if verbose:
            sys.stdout.write(
                '\n'.join([success_message, '', summary_table.to_string(index=False)])
            )

        return data
    except (InvalidDataError, SynthesizerInputError):
        result = deepcopy(data)
        _drop_rows(metadata, result, drop_missing_values)
        if verbose:
            summary_table['# Invalid Rows'] = [
                len(data[table]) - len(result[table]) for table in table_names
            ]
            summary_table['# Rows (New)'] = [len(result[table]) for table in table_names]
            sys.stdout.write('\n'.join([
                success_message, '', summary_table.to_string(index=False)
            ]))

        return result


def simplify_schema(data, metadata, verbose=True):
    """Simplify the schema of the data and metadata.

    This function simplifies the schema of the data and metadata by:
    - Removing tables that are not child or grandchild of the main root table.
    - Removing all modelable columns for grandchild tables.
    - Removing some modelable columns for child tables.
    - Removing all relationships that are not between the main root table and its children
    or grandchildren.

    Args:
        data (dict):
            Dictionary that maps each table name (string) to the data for that
            table (pandas.DataFrame).
        metadata (MultiTableMetadata):
            Metadata of the datasets.
        verbose (bool):
            If True, print information about the simplification process.
            Defaults to True.

    Returns:
        tuple:
            dict:
                Dictionary with the simplified dataframes.
            MultiTableMetadata:
                Simplified metadata.
    """
    try:
        error_message = (
            'The provided data/metadata combination is not valid.'
            ' Please make sure that the data/metadata combination is valid'
            ' before trying to simplify the schema.'
        )
        metadata.validate()
        metadata.validate_data(data)
    except InvalidMetadataError as error:
        raise InvalidMetadataError(error_message) from error
    except InvalidDataError as error:
        raise InvalidDataError([error_message]) from error

    total_estimated_columns = _get_total_estimated_columns(metadata)
    if total_estimated_columns <= MAX_NUMBER_OF_COLUMNS:
        _print_simplified_schema_summary(data, data)
        return data, metadata

    simple_metadata = _simplify_metadata(metadata)
    simple_data = _simplify_data(data, simple_metadata)
    if verbose:
        _print_simplified_schema_summary(data, simple_data)

    return simple_data, simple_metadata
