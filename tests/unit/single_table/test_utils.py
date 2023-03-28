
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sdv.metadata.single_table import SingleTableMetadata
from sdv.single_table.utils import (
    _key_order, check_num_rows, detect_discrete_columns, flatten_array, flatten_dict,
    handle_sampling_error, unflatten_dict)


def test_detect_discrete_columns():
    """Test that the detect discrete columns returns a list columns that are not continuum."""
    # Setup
    metadata = SingleTableMetadata()
    metadata.columns = {
        'name': {
            'sdtype': 'categorical',
        },
        'age': {
            'sdtype': 'numerical',
        },
        'subscribed': {
            'sdtype': 'boolean',
        },
        'join_date': {
            'sdtype': 'datetime'
        }
    }
    data = pd.DataFrame({
        'name': ['John', 'Doe', 'John Doe', 'John Doe Doe'],
        'age': [1, 2, 3, 4],
        'subscribed': [None, True, False, np.nan],
        'join_date': ['2021-02-02', '2022-03-04', '2015-05-06', '2018-09-30'],
        'uses_synthetic': [np.nan, True, False, False],
        'surname': [object(), object(), object(), object()],
        'bool': [0., 0., 1., np.nan]
    })

    # Run
    result = detect_discrete_columns(metadata, data)

    # Assert
    assert result == ['name', 'subscribed', 'uses_synthetic', 'surname', 'bool']


def test_flatten_array_default():
    """Test get flatten array."""
    # Run
    result = flatten_array([['foo', 'bar'], 'tar'])

    # Asserts
    expected = {
        '0__0': 'foo',
        '0__1': 'bar',
        '1': 'tar'
    }
    assert result == expected


def test_flatten_array_with_prefix():
    """Test get flatten array."""
    # Run
    result = flatten_array([['foo', 'bar'], 'tar'], prefix='test')

    # Asserts
    expected = {
        'test__0__0': 'foo',
        'test__0__1': 'bar',
        'test__1': 'tar'
    }
    assert result == expected


def test_flatten_dict_default():
    """Test get flatten dict with some result."""
    # Run
    nested = {
        'foo': 'value',
        'bar': {'bar_dict': 'value_bar_dict'},
        'tar': ['value_tar_list_0', 'value_tar_list_1'],
        'fitted': 'value_1',
        'distribution': 'value_2',
        'type': 'value_3'
    }
    result = flatten_dict(nested)

    # Asserts
    expected = {
        'foo': 'value',
        'bar__bar_dict': 'value_bar_dict',
        'tar__0': 'value_tar_list_0',
        'tar__1': 'value_tar_list_1'
    }
    assert result == expected


def test_flatten_dict_with_prefix():
    """Test get flatten dict with some result."""
    # Run
    nested = {
        'foo': 'value',
        'bar': {'bar_dict': 'value_bar_dict'},
        'tar': ['value_tar_list_0', 'value_tar_list_1'],
        'fitted': 'value_1',
        'distribution': 'value_2',
        'type': 'value_3'
    }
    result = flatten_dict(nested, prefix='test')

    # Asserts
    expected = {
        'test__foo': 'value',
        'test__bar__bar_dict': 'value_bar_dict',
        'test__tar__0': 'value_tar_list_0',
        'test__tar__1': 'value_tar_list_1'
    }
    assert result == expected


def test__key_order():
    """Test key order."""
    # Run
    key_value = ['foo__0__1']
    result = _key_order(key_value)

    # Asserts
    assert result == ['foo', 0, 1]


def test_unflatten_dict_raises_error_row_index():
    """Test unflatten dict raises error row_index."""
    # Run
    flat = {
        'foo__0__1': 'some value'
    }

    err_msg = 'There was an error unflattening the extension.'
    with pytest.raises(ValueError, match=err_msg):
        unflatten_dict(flat)


def test_unflatten_dict_raises_error_column_index():
    """Test unflatten dict raises error column_index."""
    # Run
    flat = {
        'foo__1__0': 'some value'
    }

    err_msg = 'There was an error unflattening the extension.'
    with pytest.raises(ValueError, match=err_msg):
        unflatten_dict(flat)


def test_unflatten_dict():
    """Test unflatten_dict."""
    # Run
    flat = {
        'foo__0__foo': 'foo value',
        'bar__0__0': 'bar value',
        'tar': 'tar value'
    }
    result = unflatten_dict(flat)

    # Asserts
    expected = {
        'foo': {0: {'foo': 'foo value'}},
        'bar': [['bar value']],
        'tar': 'tar value',
    }
    assert result == expected


def test_handle_sampling_error():
    """Test when an error is raised at the end of the function when temp dir is ``True``."""
    # Run and Assert
    error_msg = (
        'Error: Sampling terminated. Partial results are stored in a temporary file: test.csv. '
        'This file will be overridden the next time you sample. Please rename the file if you '
        'wish to save these results.'
        '\n'
        'Test error'
    )
    with pytest.raises(ValueError, match=error_msg):
        handle_sampling_error(True, 'test.csv', ValueError('Test error'))


def test_handle_sampling_error_false_temp_file():
    """Test that an error is raised when temp dir is ``False``."""
    # Run and Assert
    error_msg = (
        'Error: Sampling terminated. Partial results are stored in test.csv.'
        '\n'
        'Test error'
    )
    with pytest.raises(ValueError, match=error_msg):
        handle_sampling_error(False, 'test.csv', ValueError('Test error'))


def test_handle_sampling_error_false_temp_file_none_output_file():
    """Test the ``handle_sampling_error`` function.

    Expect that only the passed error message is raised when ``is_tmp_file`` and
    ``output_file_path`` are ``False`` or ``None``.
    """
    # Run and Assert
    error_msg = 'Test error'
    with pytest.raises(ValueError, match=error_msg):
        handle_sampling_error(False, 'test.csv', ValueError('Test error'))


def test_handle_sampling_error_ignore():
    """Test that the error is raised if the error is the no rows error."""
    # Run and assert
    error_msg = 'Unable to sample any rows for the given conditions.'
    with pytest.raises(ValueError, match=error_msg):
        handle_sampling_error(True, 'test.csv', ValueError(error_msg))


def test_check_num_rows_reject_sampling_error():
    """Test that the error for reject sampling is raised if there are no sampled rows."""
    # Setup
    num_rows = 0
    expected_num_rows = 5
    is_reject_sampling = True
    max_tries_per_batch = 1

    # Run and Assert
    error_msg = (
        'Unable to sample any rows for the given conditions. '
        r'Try increasing `max_tries_per_batch` \(currently: 1\).'
    )
    with pytest.raises(ValueError, match=error_msg):
        check_num_rows(num_rows, expected_num_rows, is_reject_sampling, max_tries_per_batch)


def test_check_num_rows_non_reject_sampling_error():
    """Test that the error for non reject sampling is raised if there are no sampled rows."""
    # Setup
    num_rows = 0
    expected_num_rows = 5
    is_reject_sampling = False
    max_tries = 1
    error_msg = (
        r'Unable to sample any rows for the given conditions. '
        'This may be because the provided values are out-of-bounds in the current model.'
    )

    # Run and assert
    with pytest.raises(ValueError, match=error_msg):
        check_num_rows(num_rows, expected_num_rows, is_reject_sampling, max_tries)


@patch('sdv.single_table.utils.warnings')
def test_check_num_rows_non_reject_sampling_warning(warning_mock):
    """Test the ``check_num_rows`` function.

    Expect that no error is raised if there are valid sampled rows.
    Expect that a warning is raised if there are fewer than the expected number of rows.
    """
    # Setup
    num_rows = 2
    expected_num_rows = 5
    is_reject_sampling = True
    max_tries = 1
    error_msg = (
        'Unable to sample any rows for the given conditions. '
        'Try increasing `max_tries` (currently: 1).'
    )

    # Run
    check_num_rows(num_rows, expected_num_rows, is_reject_sampling, max_tries)

    # Assert
    warning_mock.warn.called_once_with(error_msg)


@patch('sdv.single_table.utils.warnings')
def test_check_num_rows_valid(warning_mock):
    """Test the ``check_num_rows`` passes withnout calling warnings."""
    # Setup
    num_rows = 5
    expected_num_rows = 5
    is_reject_sampling = True
    max_tries = 1

    # Run
    check_num_rows(num_rows, expected_num_rows, is_reject_sampling, max_tries)

    # Assert
    assert warning_mock.warn.call_count == 0
