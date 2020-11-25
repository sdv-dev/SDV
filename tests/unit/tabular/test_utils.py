"""Tests for the sdv.models.utils module."""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sdv.tabular.utils import (
    _key_order, check_matrix_symmetric_positive_definite, flatten_array, flatten_dict, impute,
    make_positive_definite, square_matrix, unflatten_dict)


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


def test_impute():
    """Test _impute data."""
    # Run
    data = pd.DataFrame({'foo': [0, None, 1], 'bar': ['a', None, 'b']})
    result = impute(data)

    # Asserts
    expected = pd.DataFrame({'foo': [0, 0.5, 1], 'bar': ['a', 'a', 'b']})
    pd.testing.assert_frame_equal(result, expected)


def test_square_matrix():
    """Test fill zeros a triangular matrix."""
    # Run
    matrix = [[0.1, 0.5], [0.3]]
    result = square_matrix(matrix)

    # Asserts
    expected = [[0.1, 0.5], [0.3, 0.0]]
    assert result == expected


@patch('sdv.tabular.utils.check_matrix_symmetric_positive_definite')
def test_make_positive_definite(mock_check):
    """Test find the nearest positive-definite matrix."""
    # Setup
    mock_check.return_value = True

    # Run
    matrix = np.array([[0, 1], [1, 0]])
    result = make_positive_definite(matrix)

    # Asserts
    expected = np.array([[0.5, 0.5], [0.5, 0.5]])
    np.testing.assert_equal(result, expected)

    assert mock_check.call_count == 1


@patch('sdv.tabular.utils.check_matrix_symmetric_positive_definite')
def test_make_positive_definite_iterate(mock_check):
    """Test find the nearest positive-definite matrix iterating."""
    # Setup
    mock_check.side_effect = [False, False, True]

    # Run
    matrix = np.array([[-1, -5], [-3, -7]])
    result = make_positive_definite(matrix)

    # Asserts
    expected = np.array([[0.8, -0.4], [-0.4, 0.2]])
    np.testing.assert_array_almost_equal(result, expected)

    assert mock_check.call_count == 3


def test_check_matrix_symmetric_positive_definite_shape_error():
    """Test check matrix shape error."""
    # Run
    matrix = np.array([])
    result = check_matrix_symmetric_positive_definite(matrix)

    # Asserts
    assert not result


def test_check_matrix_symmetric_positive_definite_np_error():
    """Test check matrix numpy raise error."""
    # Run
    matrix = np.array([[-1, 0], [0, 0]])
    result = check_matrix_symmetric_positive_definite(matrix)

    # Asserts
    assert not result


def test_check_matrix_symmetric_positive_definite():
    """Test check matrix numpy."""
    # Run
    matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
    result = check_matrix_symmetric_positive_definite(matrix)

    # Asserts
    assert result


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

    with pytest.raises(ValueError):
        unflatten_dict(flat)


def test_unflatten_dict_raises_error_column_index():
    """Test unflatten dict raises error column_index."""
    # Run
    flat = {
        'foo__1__0': 'some value'
    }

    with pytest.raises(ValueError):
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
