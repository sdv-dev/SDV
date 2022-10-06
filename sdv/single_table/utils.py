"""Utility functions for tabular models."""

import os
import warnings

TMP_FILE_NAME = '.sample.csv.temp'
DISABLE_TMP_FILE = 'disable'


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
        if column in metadata._columns:
            if metadata._columns[column]['sdtype'] not in ['numerical', 'datetime']:
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


@staticmethod
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
