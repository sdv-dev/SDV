"""Utility functions for tabular models."""

import warnings

import numpy as np
import tqdm

IGNORED_DICT_KEYS = ['fitted', 'distribution', 'type']


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
    result = dict()
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
    result = dict()

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
    parts = list()
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
    unflattened = dict()

    for key, value in sorted(flat.items(), key=_key_order):
        if '__' in key:
            key, subkey = key.split('__', 1)
            subkey, name = subkey.rsplit('__', 1)

            if name.isdigit():
                column_index = int(name)
                row_index = int(subkey)

                array = unflattened.setdefault(key, list())

                if len(array) == row_index:
                    row = list()
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
                subdict = unflattened.setdefault(key, dict())
                if subkey.isdigit():
                    subkey = int(subkey)

                inner = subdict.setdefault(subkey, dict())
                inner[name] = value

        else:
            unflattened[key] = value

    return unflattened


def progress_bar_wrapper(function, pb_total, pb_description):
    """Enclose given function with a progress bar.

    Args:
        function (function):
            The function to execute.
        pb_total (int):
            The total to use in the progress bar.
        pb_description (str):
            The description of the progress bar.

    Returns:
        The function return value.
    """
    with tqdm.tqdm(total=pb_total) as progress_bar:
        progress_bar.set_description(pb_description)
        return function(progress_bar)


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

    if is_tmp_file:
        print('Error: Sampling terminated. Partial results are stored in a temporary file: '
              f'{output_file_path}. This file will be overridden the next time you sample. '
              'Please rename the file if you wish to save these results.')
    elif output_file_path is not None:
        print('Error: Sampling terminated. '
              f'Partial results are stored in {output_file_path}.')

    raise sampling_error


def check_num_rows(num_rows, expected_num_rows, is_reject_sampling, max_tries, batch_size_per_try):
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
        max_tries (int):
            The maximum number of tries in reject sampling.
        batch_size_per_try (int):
            The batch size per try in reject sampling.

    Side Effects:
        ValueError or warning.
    """
    if num_rows < expected_num_rows:
        if num_rows == 0:
            user_msg = ('Unable to sample any rows for the given conditions. ')
            if is_reject_sampling:
                user_msg = user_msg + (
                    f'Try increasing `max_tries` (currently: {max_tries}) or increasing '
                    f'`batch_size_per_try` (currently: {batch_size_per_try}). Note that '
                    'increasing these values will also increase the sampling time.'
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
                'To sample more rows, try increasing `max_tries` '
                f'(currently: {max_tries}) or increasing `batch_size_per_try` '
                f'(currently: {batch_size_per_try}. Note that increasing these values '
                f'will also increase the sampling time.'
            )
            warnings.warn(user_msg)
