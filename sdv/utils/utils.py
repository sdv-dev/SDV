"""Utils module."""

import datetime
import sys
import warnings
from copy import deepcopy

import cloudpickle
import numpy as np
import pandas as pd

from sdv._utils import (
    _validate_foreign_keys_not_null,
    check_sdv_versions_and_warn,
    check_synthesizer_version,
    generate_synthesizer_id,
)
from sdv.errors import InvalidDataError, SamplingError, SynthesizerInputError
from sdv.logging import get_sdv_logger
from sdv.metadata.metadata import Metadata
from sdv.multi_table.base import BaseMultiTableSynthesizer
from sdv.multi_table.utils import _drop_rows

SINGLE_TABLE_SYNTHESIZER_LOGGER = get_sdv_logger('SingleTableSynthesizer')
MULTI_TABLE_SYNTHESIZER_LOGGER = get_sdv_logger('MultiTableSynthesizer')


def drop_unknown_references(data, metadata, drop_missing_values=False, verbose=True):
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
            Defaults to False.
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
        '# Rows (New)': [len(data[table]) for table in table_names],
    })
    metadata.validate()
    try:
        # Suppress duplicate datetime_format warnings during referential integrity validation.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message=r"No 'datetime_format' is present.*",
                category=UserWarning,
            )
            metadata.validate_data(data)
        if drop_missing_values:
            _validate_foreign_keys_not_null(metadata, data)

        if verbose:
            sys.stdout.write('\n'.join([success_message, '', summary_table.to_string(index=False)]))

        return data
    except (InvalidDataError, SynthesizerInputError):
        result = deepcopy(data)
        _drop_rows(result, metadata, drop_missing_values)
        if verbose:
            summary_table['# Invalid Rows'] = [
                len(data[table]) - len(result[table]) for table in table_names
            ]
            summary_table['# Rows (New)'] = [len(result[table]) for table in table_names]
            sys.stdout.write('\n'.join([success_message, '', summary_table.to_string(index=False)]))

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
        metadata (Metadata):
            A Metadata object describing the data.
        num_sequences (int):
            The number of sequences to subsample.
        max_sequence_length (int):
            The maximum length each subsampled sequence is allowed to be. Defaults to None. If
            None, do not enforce any max length, meaning that entire sequences will be sampled.
            If provided all subsampled sequences must be <= the provided length.
        long_sequence_subsampling_method (str):
            The method to use when a selected sequence is too long. Options are:
            - first_rows (default): Keep the first n rows of the sequence, where n is the max
            sequence length.
            - last_rows: Keep the last n rows of the sequence, where n is the max sequence length.
            - random: Randomly choose n rows to keep within the sequence. It is important to keep
            the randomly chosen rows in the same order as they appear in the original data.
    """
    if isinstance(metadata, Metadata):
        metadata = metadata._convert_to_single_table()

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

    if sequence_key not in data.columns:
        raise ValueError(
            'Your provided sequence key is not in the data. This is required to get a subset.'
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


def load_synthesizer(filepath):
    """Load a synthesizer from a file.

    Args:
        filepath (str):
            The path to the file containing the synthesizer.
    """
    with open(filepath, 'rb') as f:
        try:
            synthesizer = cloudpickle.load(f)
        except RuntimeError as e:
            err_msg = (
                'Attempting to deserialize object on a CUDA device but '
                'torch.cuda.is_available() is False. If you are running on a CPU-only machine,'
                " please use torch.load with map_location=torch.device('cpu') "
                'to map your storages to the CPU.'
            )
            if str(e) == err_msg:
                raise SamplingError(
                    'This synthesizer was created on a machine with GPU but the current '
                    'machine is CPU-only. This feature is currently unsupported. We recommend'
                    ' sampling on the same GPU-enabled machine.'
                )
            raise e

    check_synthesizer_version(synthesizer)
    check_sdv_versions_and_warn(synthesizer)
    if getattr(synthesizer, '_synthesizer_id', None) is None:
        synthesizer._synthesizer_id = generate_synthesizer_id(synthesizer)

    logger = (
        MULTI_TABLE_SYNTHESIZER_LOGGER
        if isinstance(synthesizer, BaseMultiTableSynthesizer)
        else SINGLE_TABLE_SYNTHESIZER_LOGGER
    )
    logger.info({
        'EVENT': 'Load',
        'TIMESTAMP': datetime.datetime.now(),
        'SYNTHESIZER CLASS NAME': synthesizer.__class__.__name__,
        'SYNTHESIZER ID': synthesizer._synthesizer_id,
    })

    return synthesizer
