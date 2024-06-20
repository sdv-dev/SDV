"""POC functions to use HMASynthesizer succesfully."""

import warnings

from sdv.errors import InvalidDataError
from sdv.metadata.errors import InvalidMetadataError
from sdv.multi_table.hma import MAX_NUMBER_OF_COLUMNS
from sdv.multi_table.utils import (
    _get_total_estimated_columns,
    _print_simplified_schema_summary,
    _print_subsample_summary,
    _simplify_data,
    _simplify_metadata,
    _subsample_data,
)
from sdv.utils.utils import drop_unknown_references as utils_drop_unknown_references


def drop_unknown_references(data, metadata, drop_missing_values=False, verbose=True):
    """Wrap the drop_unknown_references function from the utils module."""
    warnings.warn(
        "Please access the 'drop_unknown_references' function directly from the sdv.utils module"
        'instead of sdv.utils.poc.',
        FutureWarning,
    )
    return utils_drop_unknown_references(data, metadata, drop_missing_values, verbose)


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


def get_random_subset(data, metadata, main_table_name, num_rows, verbose=True):
    """Subsample multi-table table based on a table and a number of rows.

    The strategy is to:
    - Subsample the disconnected roots tables by keeping a similar proportion of data
      than the main table. Ensure referential integrity.
    - Subsample the main table and its descendants to ensure referential integrity.
    - Subsample the ancestors of the main table by removing primary key rows that are no longer
      referenced by the descendants and drop also some unreferenced rows.

    Args:
        data (dict):
            Dictionary that maps each table name (string) to the data for that
            table (pandas.DataFrame).
        metadata (MultiTableMetadata):
            Metadata of the datasets.
        main_table_name (str):
            Name of the main table.
        num_rows (int):
            Number of rows to keep in the main table.
        verbose (bool):
            If True, print information about the subsampling process.
            Defaults to True.

    Returns:
        dict:
            Dictionary with the subsampled dataframes.
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

    error_message_num_rows = '``num_rows`` must be a positive integer.'
    if not isinstance(num_rows, (int, float)) or num_rows != int(num_rows):
        raise ValueError(error_message_num_rows)

    if num_rows <= 0:
        raise ValueError(error_message_num_rows)

    if len(data[main_table_name]) <= num_rows:
        if verbose:
            _print_subsample_summary(data, data)

        return data

    result = _subsample_data(data, metadata, main_table_name, num_rows)
    if verbose:
        _print_subsample_summary(data, result)

    metadata.validate_data(result)
    return result
