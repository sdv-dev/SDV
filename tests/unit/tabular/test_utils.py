"""Tests for the sdv.models.utils module."""

from unittest.mock import call, patch

import pytest
import tqdm

from sdv.tabular.utils import (
    _key_order, flatten_array, flatten_dict, handle_sampling_error, progress_bar_wrapper,
    unflatten_dict)


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


@patch('sdv.tabular.utils.tqdm.tqdm', spec=tqdm.tqdm)
def test_progress_bar_wrapper(tqdm_mock):
    """Test the ``progress_bar_wrapper`` function.

    Expect that it wraps the given function with a tqdm progress bar.

    Input:
        - test function
        - total=100
        - progress bar description
    Output:
        - test function output
    Side Effects:
        - the progress bar is created.
    """
    # Setup
    total = 100
    description = 'test description'

    def test_fn(pbar):
        return 'hello'

    # Run
    output = progress_bar_wrapper(test_fn, total, description)

    # Assert
    assert output == 'hello'
    tqdm_mock.assert_has_calls([
        call(total=total),
        call().__enter__(),
        call().__enter__().set_description(description),
        call().__exit__(None, None, None),
    ])


def test_handle_sampling_error():
    """Test the ``handle_sampling_error`` function.

    Expect that the error is raised at the end of the function.

    Input:
        - a temp file
        - the sampling error

    Side Effects:
        - the error is raised.
    """
    # Setup
    error_msg = 'Test error'

    # Run and assert
    with pytest.raises(ValueError, match=error_msg):
        handle_sampling_error(True, 'test.csv', ValueError(error_msg))
