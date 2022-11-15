
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import scipy

from sdv.metadata.single_table import SingleTableMetadata
from sdv.single_table.utils import (
    _key_order, check_num_rows, detect_discrete_columns, flatten_array, flatten_dict,
    get_nearest_correlation_matrix, handle_sampling_error, rebuild_correlation_matrix,
    unflatten_dict)


def test_detect_discrete_columns():
    """Test that the detect discrete columns returns a list columns that are not continuum."""
    # Setup
    metadata = SingleTableMetadata()
    metadata._columns = {
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
        'surname.value': [object(), object(), object(), object()],
        'bool.value': [0., 0., 1., np.nan]
    })

    # Run
    result = detect_discrete_columns(metadata, data)

    # Assert
    assert result == ['name', 'subscribed', 'uses_synthetic', 'surname.value', 'bool.value']


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
    """Test the ``handle_sampling_error`` function.

    Expect that the error is raised at the end of the function.

    Input:
        - True
        - a temp file
        - the sampling error

    Side Effects:
        - the error is raised.
    """
    # Setup
    error_msg = (
        'Error: Sampling terminated. Partial results are stored in a temporary file: test.csv. '
        'This file will be overridden the next time you sample. Please rename the file if you '
        'wish to save these results.'
        '\n'
        'Test error'
    )

    # Run and assert
    with pytest.raises(ValueError, match=error_msg):
        handle_sampling_error(True, 'test.csv', ValueError('Test error'))


def test_handle_sampling_error_false_temp_file():
    """Test the ``handle_sampling_error`` function.

    Expect that the error is raised at the end of the function.

    Input:
        - False
        - a temp file
        - the sampling error

    Side Effects:
        - the error is raised.
    """
    # Setup
    error_msg = (
        'Error: Sampling terminated. Partial results are stored in test.csv.'
        '\n'
        'Test error'
    )

    # Run and assert
    with pytest.raises(ValueError, match=error_msg):
        handle_sampling_error(False, 'test.csv', ValueError('Test error'))


def test_handle_sampling_error_false_temp_file_none_output_file():
    """Test the ``handle_sampling_error`` function.

    Expect that only the passed error message is raised when ``is_tmp_file`` and
    ``output_file_path`` are False/None.

    Input:
        - False
        - None
        - the sampling error

    Side Effects:
        - the samlping error is raised
    """
    # Setup
    error_msg = 'Test error'

    # Run and assert
    with pytest.raises(ValueError, match=error_msg):
        handle_sampling_error(False, 'test.csv', ValueError('Test error'))


def test_handle_sampling_error_ignore():
    """Test the ``handle_sampling_error`` function.

    Expect that the error is raised if the error is the no rows error.

    Input:
        - a temp file
        - the sampling error

    Side Effects:
        - the error is raised.
    """
    # Setup
    error_msg = 'Unable to sample any rows for the given conditions.'

    # Run and assert
    with pytest.raises(ValueError, match=error_msg):
        handle_sampling_error(True, 'test.csv', ValueError(error_msg))


def test_check_num_rows_reject_sampling_error():
    """Test the ``check_num_rows`` function.

    Expect that the error for reject sampling is raised if there are no sampled rows.

    Input:
        - no sampled rows
        - is_reject_sampling is True

    Side Effects:
        - the error is raised.
    """
    # Setup
    num_rows = 0
    expected_num_rows = 5
    is_reject_sampling = True
    max_tries_per_batch = 1
    error_msg = (
        'Unable to sample any rows for the given conditions. '
        r'Try increasing `max_tries_per_batch` \(currently: 1\).'
    )

    # Run and assert
    with pytest.raises(ValueError, match=error_msg):
        check_num_rows(num_rows, expected_num_rows, is_reject_sampling, max_tries_per_batch)


def test_check_num_rows_non_reject_sampling_error():
    """Test the ``check_num_rows`` function.

    Expect that the error for non reject sampling is raised if there are no sampled rows.

    Input:
        - no sampled rows
        - is_reject_sampling is False

    Side Effects:
        - the error is raised.
    """
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


@patch('sdv.tabular.utils.warnings')
def test_check_num_rows_non_reject_sampling_warning(warning_mock):
    """Test the ``check_num_rows`` function.

    Expect that no error is raised if there are valid sampled rows.
    Expect that a warning is raised if there are fewer than the expected number of rows.

    Input:
        - no sampled rows
        - is_reject_sampling is False

    Side Effects:
        - the error is raised.
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


@patch('sdv.tabular.utils.warnings')
def test_check_num_rows_valid(warning_mock):
    """Test the ``check_num_rows`` function.

    Expect that no error is raised if there are valid sampled rows.
    Expect that a warning is raised if there are fewer than the expected number of rows.

    Input:
        - no sampled rows
        - is_reject_sampling is False

    Side Effects:
        - the error is raised.
    """
    # Setup
    num_rows = 5
    expected_num_rows = 5
    is_reject_sampling = True
    max_tries = 1

    # Run
    check_num_rows(num_rows, expected_num_rows, is_reject_sampling, max_tries)

    # Assert
    assert warning_mock.warn.call_count == 0


def test_get_nearest_correlation_matrix_valid():
    """Test ``get_nearest_correlation_matrix`` with a psd input.

    If the matrix is positive semi-definite, do nothing.

    Input:
        - matrix which is positive semi-definite.

    Expected Output:
        - the input, unmodified.
    """
    # Run
    correlation_matrix = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    output = get_nearest_correlation_matrix(correlation_matrix)

    # Assert
    expected = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    assert expected == output.tolist()
    assert output is correlation_matrix


def test_get_nearest_correlation_matrix_invalid():
    """Test ``get_nearest_correlation_matrix`` with a non psd input.

    If the matrix is not positive semi-definite, modify it to make it PSD.

    Input:
        - matrix which is not positive semi-definite.

    Expected Output:
        - modified matrix which is positive semi-definite.
    """
    # Run
    not_psd_matrix = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, -1],
    ])
    output = get_nearest_correlation_matrix(not_psd_matrix)

    # Assert
    expected = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    assert expected == output.tolist()

    not_psd_eigenvalues = scipy.linalg.eigh(not_psd_matrix)[0]
    output_eigenvalues = scipy.linalg.eigh(output)[0]
    assert (not_psd_eigenvalues < 0).any()
    assert (output_eigenvalues >= 0).all()


def test_rebuild_correlation_matrix_valid():
    """Test ``rebuild_correlation_matrix`` with a valid correlation input.

    If the input contains values between -1 and 1, the method is expected
    to simply rebuild the square matrix with the same values.

    Input:
        - list of lists with values between -1 and 1

    Expected Output:
        - numpy array with the square correlation matrix
    """
    # Run
    triangular_covariance = [
        [0.1],
        [0.2, 0.3]
    ]
    correlation = rebuild_correlation_matrix(triangular_covariance)

    # Assert
    expected = [
        [1.0, 0.1, 0.2],
        [0.1, 1.0, 0.3],
        [0.2, 0.3, 1.0]
    ]
    assert expected == correlation


def test_rebuild_correlation_matrix_outside():
    """Test ``_rebuild_correlation_matrix`` with an invalid correlation input.

    If the input contains values outside -1 and 1, the method is expected
    to scale them down to the valid range.

    Input:
        - list of lists with values outside of -1 and 1

    Expected Output:
        - numpy array with the square correlation matrix
    """
    # Run
    triangular_covariance = [
        [1.0],
        [2.0, 1.0]
    ]
    correlation = rebuild_correlation_matrix(triangular_covariance)

    # Assert
    expected = [
        [1.0, 0.5, 1.0],
        [0.5, 1.0, 0.5],
        [1.0, 0.5, 1.0]
    ]
    assert expected == correlation
