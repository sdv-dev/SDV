"""Utility functions for tabular models."""

import os
import warnings

import numpy as np

from sdv.errors import SynthesizerInputError

TMP_FILE_NAME = '.sample.csv.temp'
DISABLE_TMP_FILE = 'disable'
IGNORED_DICT_KEYS = ['fitted', 'distribution', 'type']


def detect_discrete_columns(metadata, data):
    """Detect the discrete columns in a dataset.

    Args:
        metadata (sdv.metadata.SingleTableMetadata):
            Metadata that belongs to the given ``data``.

        data (pandas.DataFrame):
            ``pandas.DataFrame`` that matches the ``metadata``.

    Returns:
        discrete_columns (list):
            A list of discrete columns to be used with some of ``sdv`` synthesizers.
    """
    discrete_columns = []

    for column in data.columns:
        if column in metadata.columns:
            if metadata.columns[column]['sdtype'] not in ['numerical', 'datetime']:
                discrete_columns.append(column)

        else:
            column_data = data[column].dropna()
            if set(column_data.unique()) == {0.0, 1.0}:
                column_data = column_data.astype(bool)

            try:
                dtype = column_data.infer_objects().dtype.kind
                if dtype in ['O', 'b']:
                    discrete_columns.append(column)

            except Exception:
                discrete_columns.append(column)

    return discrete_columns


def handle_sampling_error(is_tmp_file, output_file_path, sampling_error):
    """Handle sampling errors by printing a user-legible error and then raising.

    Args:
        is_tmp_file (bool):
            Whether or not the output file is a temp file.
        output_file_path (str):
            The output file path.
        sampling_error:
            The error to raise.

    Side Effects:
        The error will be raised.
    """
    if 'Unable to sample any rows for the given conditions' in str(sampling_error):
        raise sampling_error

    error_msg = None
    if is_tmp_file:
        error_msg = (
            'Error: Sampling terminated. Partial results are stored in a temporary file: '
            f'{output_file_path}. This file will be overridden the next time you sample. '
            'Please rename the file if you wish to save these results.'
        )
    elif output_file_path is not None:
        error_msg = (
            f'Error: Sampling terminated. Partial results are stored in {output_file_path}.'
        )

    if error_msg:
        raise type(sampling_error)(error_msg + '\n' + str(sampling_error))

    raise sampling_error


def check_num_rows(num_rows, expected_num_rows, is_reject_sampling, max_tries_per_batch):
    """Check the number of sampled rows against the expected number of rows.

    If the number of sampled rows is zero, throw a ValueError.
    If the number of sampled rows is less than the expected number of rows,
    raise a warning.

    Args:
        num_rows (int):
            The number of sampled rows.
        expected_num_rows (int):
            The expected number of rows.
        is_reject_sampling (bool):
            If reject sampling is used or not.
        max_tries_per_batch (int):
            Number of times to retry sampling until the batch size is met.

    Side Effects:
        ValueError or warning.
    """
    if num_rows < expected_num_rows:
        if num_rows == 0:
            user_msg = ('Unable to sample any rows for the given conditions. ')
            if is_reject_sampling:
                user_msg = user_msg + (
                    f'Try increasing `max_tries_per_batch` (currently: {max_tries_per_batch}). '
                    'Note that increasing this value will also increase the sampling time.'
                )
            else:
                user_msg = user_msg + (
                    'This may be because the provided values are out-of-bounds in the '
                    'current model. \nPlease try again with a different set of values.'
                )
            raise ValueError(user_msg)

        else:
            # This case should only happen with reject sampling.
            user_msg = (
                f'Only able to sample {num_rows} rows for the given conditions. '
                'To sample more rows, try increasing `max_tries_per_batch` '
                f'(currently: {max_tries_per_batch}). Note that increasing this value '
                'will also increase the sampling time.'
            )
            warnings.warn(user_msg)


def validate_file_path(output_file_path):
    """Validate the user-passed output file arg, and create the file."""
    output_path = None
    if output_file_path == DISABLE_TMP_FILE:
        # Temporary way of disabling the output file feature, used by HMA1.
        return output_path

    elif output_file_path:
        output_path = os.path.abspath(output_file_path)
        if os.path.exists(output_path):
            raise AssertionError(f'{output_path} already exists.')

    else:
        if os.path.exists(TMP_FILE_NAME):
            os.remove(TMP_FILE_NAME)

        output_path = TMP_FILE_NAME

    # Create the file.
    with open(output_path, 'w+'):
        pass

    return output_path


def flatten_array(nested, prefix=''):
    """Flatten an array as a dict.

    Args:
        nested (list, numpy.array):
            Iterable to flatten.
        prefix (str):
            Name to append to the array indices. Defaults to ``''``.

    Returns:
        dict:
            Flattened array.
    """
    result = {}
    for index in range(len(nested)):
        prefix_key = '__'.join([prefix, str(index)]) if len(prefix) else str(index)

        value = nested[index]
        if isinstance(value, (list, np.ndarray)):
            result.update(flatten_array(value, prefix=prefix_key))

        elif isinstance(value, dict):
            result.update(flatten_dict(value, prefix=prefix_key))

        else:
            result[prefix_key] = value

    return result


def flatten_dict(nested, prefix=''):
    """Flatten a dictionary.

    This method returns a flatten version of a dictionary, concatenating key names with
    double underscores.

    Args:
        nested (dict):
            Original dictionary to flatten.
        prefix (str):
            Prefix to append to key name. Defaults to ``''``.

    Returns:
        dict:
            Flattened dictionary.
    """
    result = {}

    for key, value in nested.items():
        prefix_key = '__'.join([prefix, str(key)]) if len(prefix) else key

        if key in IGNORED_DICT_KEYS and not isinstance(value, (dict, list)):
            continue

        elif isinstance(value, dict):
            result.update(flatten_dict(value, prefix_key))

        elif isinstance(value, (np.ndarray, list)):
            result.update(flatten_array(value, prefix_key))

        else:
            result[prefix_key] = value

    return result


def _key_order(key_value):
    parts = []
    for part in key_value[0].split('__'):
        if part.isdigit():
            part = int(part)

        parts.append(part)

    return parts


def unflatten_dict(flat):
    """Transform a flattened dict into its original form.

    Args:
        flat (dict):
            Flattened dict.

    Returns:
        dict:
            Nested dict (if corresponds)
    """
    unflattened = {}

    for key, value in sorted(flat.items(), key=_key_order):
        if '__' in key:
            key, subkey = key.split('__', 1)
            subkey, name = subkey.rsplit('__', 1)

            if name.isdigit():
                column_index = int(name)
                row_index = int(subkey)

                array = unflattened.setdefault(key, [])

                if len(array) == row_index:
                    row = []
                    array.append(row)
                elif len(array) == row_index + 1:
                    row = array[row_index]
                else:
                    # This should never happen
                    raise ValueError('There was an error unflattening the extension.')

                if len(row) == column_index:
                    row.append(value)
                else:
                    # This should never happen
                    raise ValueError('There was an error unflattening the extension.')

            else:
                subdict = unflattened.setdefault(key, {})
                if subkey.isdigit():
                    subkey = int(subkey)

                inner = subdict.setdefault(subkey, {})
                inner[name] = value

        else:
            unflattened[key] = value

    return unflattened


def validate_numerical_distributions(numerical_distributions, metadata_columns):
    """Validate ``numerical_distributions``.

    Raise an error if it's not None or dict, or if its columns are not present in the metadata.

    Args:
        numerical_distributions (dict):
            Dictionary that maps field names from the table that is being modeled with
            the distribution that needs to be used.
        metadata_columns (list):
            Columns present in the metadata.
    """
    if numerical_distributions:
        if not isinstance(numerical_distributions, dict):
            raise TypeError('numerical_distributions can only be None or a dict instance.')

        invalid_columns = numerical_distributions.keys() - set(metadata_columns)
        if invalid_columns:
            raise SynthesizerInputError(
                'Invalid column names found in the numerical_distributions dictionary '
                f'{invalid_columns}. The column names you provide must be present '
                'in the metadata.'
            )


def log_numerical_distributions_error(numerical_distributions, processed_data_columns, logger):
    """Log error when numerical distributions columns don't exist anymore."""
    unseen_columns = numerical_distributions.keys() - set(processed_data_columns)
    for column in unseen_columns:
        logger.info(
            f"Requested distribution '{numerical_distributions[column]}' "
            f"cannot be applied to column '{column}' because it no longer "
            'exists after preprocessing.'
        )
