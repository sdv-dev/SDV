"""POC functions to use HMASynthesizer succesfully."""

import warnings

import numpy as np
import pandas as pd

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


def drop_unknown_references(data, metadata):
    """Wrap the drop_unknown_references function from the utils module."""
    warnings.warn(
        "Please access the 'drop_unknown_references' function directly from the sdv.utils module"
        'instead of sdv.utils.poc.',
        FutureWarning,
    )
    return utils_drop_unknown_references(data, metadata)


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


def get_random_sequence_subset(
    data,
    metadata,
    num_sequences,
    max_sequence_length=None,
    long_sequence_subsampling_method='first_rows',
):
    """Subsample sequential data based on a number of sequences.

    Args:
        data (pandas.DataFrame):
            The sequential data.
        metadata (SingleTableMetadata):
            A SingleTableMetadata object describing the data.
        num_sequences (int):
            The number of sequences to subsample.
        max_sequence_length (int):
            The maximum length each subsampled sequence is allowed to be. Defaults to None. If
            None, do not enforce any max length, meaning that entire sequences will be sampled.
            If provided all subsampled sequences must be <= the provided length.
        long_sequence_subsampling_method (str):
            The method to use when a selected sequence is too long. Options are:
            - (default) first_rows: Keep the first n rows of the sequence, where n is the max
            sequence length.
            - last_rows: Keep the last n rows of the sequence, where n is the max sequence length.
            - random: Randomly choose n rows to keep within the sequence. It is important to keep
            the randomly chosen rows in the same order as they appear in the original data.
    """
    if long_sequence_subsampling_method not in ['first_rows', 'last_rows', 'random']:
        raise ValueError(
            'long_sequence_subsampling_method must be one of "first_rows", "last_rows" or "random"'
        )

    sequence_key = metadata.sequence_key
    if not sequence_key:
        raise ValueError(
            'Your metadata does not include a sequence key. A sequence key must be provided to '
            'subset the sequential data.'
        )

    selected_sequences = np.random.permutation(data[sequence_key])[:num_sequences]
    subset = data[data[sequence_key].isin(selected_sequences)].reset_index(drop=True)
    if max_sequence_length:
        grouped_sequences = subset.groupby(sequence_key)
        if long_sequence_subsampling_method == 'first_rows':
            return grouped_sequences.head(max_sequence_length).reset_index(drop=True)
        elif long_sequence_subsampling_method == 'last_rows':
            return grouped_sequences.tail(max_sequence_length).reset_index(drop=True)
        else:
            subsetted_sequences = []
            for _, group in grouped_sequences:
                if len(group) > max_sequence_length:
                    idx = np.random.permutation(len(group))[:max_sequence_length]
                    idx.sort()
                    subsetted_sequences.append(group.iloc[idx])
                else:
                    subsetted_sequences.append(group)

            return pd.concat(subsetted_sequences, ignore_index=True)

    return subset
