"""CAG _utils unit tests."""

import re
from unittest.mock import Mock

import pytest

from sdv.cag._errors import PatternNotMetError
from sdv.cag._utils import (
    _convert_to_snake_case,
    _is_list_of_type,
    _validate_table_and_column_names,
    _validate_table_name_if_defined,
)


def test__validate_table_and_column_names():
    """Test `_validate_table_and_column_names` method."""
    # Setup
    columns_correct = {'parent_1', 'parent_2'}
    wrong_columns = {'wrong_column_1', 'wrong_column_2'}
    metadata = Mock()
    metadata.tables = {'parent': Mock(), 'child': Mock()}
    metadata.tables['parent'].columns = columns_correct

    expected_not_single_table = re.escape(
        'Metadata contains more than 1 table but no ``table_name`` provided.'
    )

    expected_error_wrong_table = re.escape("Table 'wrong_table' missing from metadata.")
    expected_error_wrong_columns = re.escape(
        "Table 'parent' is missing columns 'wrong_column_1', 'wrong_column_2'."
    )

    # Run and Assert
    _validate_table_and_column_names('parent', columns_correct, metadata)
    with pytest.raises(PatternNotMetError, match=expected_not_single_table):
        _validate_table_and_column_names(None, columns_correct, metadata)

    with pytest.raises(PatternNotMetError, match=expected_error_wrong_table):
        _validate_table_and_column_names('wrong_table', columns_correct, metadata)

    with pytest.raises(PatternNotMetError, match=expected_error_wrong_columns):
        _validate_table_and_column_names('parent', wrong_columns, metadata)


def test__validate_table_and_column_names_single_table():
    """Test `_validate_table_and_column_names` method with only a single table."""
    # Setup
    columns_correct = {'parent_1', 'parent_2'}
    metadata = Mock()
    metadata.tables = {'parent': Mock()}
    metadata.tables['parent'].columns = columns_correct
    metadata._get_single_table_name.return_value = 'parent'

    # Run
    _validate_table_and_column_names('parent', columns_correct, metadata)
    _validate_table_and_column_names(None, columns_correct, metadata)

    # Assert
    metadata._get_single_table_name.assert_called_once()


def test__validate_table_name_if_defined():
    """Test `_validate_table_name_if_defined` method works with None or string"""
    _validate_table_name_if_defined(table_name='child')
    _validate_table_name_if_defined(table_name=None)


def test__validate_table_name_if_defined_raises():
    """Test `_validate_table_name_if_defined` method raises an error with wrong type"""
    expected_table_name_str_or_none = '`table_name` must be a string or None.'
    with pytest.raises(ValueError, match=expected_table_name_str_or_none):
        _validate_table_name_if_defined(table_name=1)


def test__is_list_of_type():
    """Test `_is_list_of_type` method"""
    assert _is_list_of_type(['a', 'b'])
    assert not _is_list_of_type(['a', 1])
    assert not _is_list_of_type([1, 2])
    assert not _is_list_of_type(1)
    assert not _is_list_of_type('a')


def test__convert_to_snake_case():
    """Test `_convert_to_snake_case` method"""
    assert _convert_to_snake_case('camelCaseString') == 'camel_case_string'
    assert _convert_to_snake_case('PascalCaseString') == 'pascal_case_string'
