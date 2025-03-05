"""CAG _utils unit tests."""

import re
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from sdv.cag._errors import PatternNotMetError
from sdv.cag._utils import (
    _is_list_of_strings,
    _remove_columns_from_metadata,
    _validate_table_and_column_names,
    _validate_table_name_if_defined,
)
from sdv.metadata.metadata import Metadata


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


def test__remove_columns_from_metadata_single():
    """Test `_remove_columns_from_metadata` method removes columns from metadata (single-table)"""
    # Setup
    original_metadata = Metadata.load_from_dict({
        'tables': {
            'table': {
                'columns': {
                    'country_column': {'sdtype': 'categorical'},
                    'city_column': {'sdtype': 'categorical'},
                },
                'column_relationships': [
                    {'type': 'address', 'column_names': ['country_column', 'city_column']}
                ],
            }
        },
        'relationships': [],
        'METADATA_SPEC_VERSION': 'V1',
    })

    # Run
    column_to_drop = 'country_column'
    new_metadata = _remove_columns_from_metadata(
        metadata=original_metadata,
        table_name='table',
        columns_to_drop=[column_to_drop],
    )

    # Assert
    assert isinstance(new_metadata, Metadata)
    assert column_to_drop in original_metadata.tables['table'].columns
    assert (
        column_to_drop in original_metadata.tables['table'].column_relationships[0]['column_names']
    )
    assert column_to_drop not in new_metadata.tables['table'].columns
    assert 'city_column' in new_metadata.tables['table'].columns
    assert len(new_metadata.tables['table'].column_relationships) == 0


def test__remove_columns_from_metadata_multi():
    """Test `_remove_columns_from_metadata` method removes columns from metadata (multi-table)"""
    # Setup
    original_metadata = Metadata.load_from_dict({
        'tables': {
            'parent': {
                'primary_key': 'id',
                'columns': {
                    'id': {'sdtype': 'id'},
                    'A': {'sdtype': 'numerical'},
                    'B': {'sdtype': 'numerical'},
                },
                'column_relationships': [{'type': 'gps', 'column_names': ['A', 'B']}],
            },
            'child': {
                'primary_key': 'id',
                'columns': {
                    'id': {'sdtype': 'id'},
                },
            },
        },
        'relationships': [
            {
                'parent_table_name': 'parent',
                'parent_primary_key': 'id',
                'child_table_name': 'child',
                'child_foreign_key': 'id',
            },
        ],
    })
    columns_to_drop = ['A', 'B']

    # Run
    new_metadata = _remove_columns_from_metadata(
        metadata=original_metadata,
        table_name='parent',
        columns_to_drop=columns_to_drop,
    )

    # Assert
    assert isinstance(new_metadata, Metadata)
    for column in columns_to_drop:
        assert column in original_metadata.tables['parent'].columns
        assert column in original_metadata.tables['parent'].column_relationships[0]['column_names']

        assert column not in new_metadata.tables['parent'].columns
        assert len(new_metadata.tables['parent'].column_relationships) == 0


def test__remove_columns_from_metadata_raises_pk():
    """Test `_remove_columns_from_metadata` method raises an error if primary key is dropped"""
    # Setup
    original_metadata = Metadata.load_from_dict({
        'tables': {
            'parent': {
                'primary_key': 'id',
                'columns': {'id': {'sdtype': 'id'}},
            },
        },
        'relationships': [
            {
                'parent_table_name': 'parent',
                'parent_primary_key': 'id',
                'child_table_name': 'child',
                'child_foreign_key': 'id',
            },
        ],
    })

    # Run and Assert
    cannot_remove_pk = 'Cannot remove primary key from Metadata'
    with pytest.raises(ValueError, match=cannot_remove_pk):
        _remove_columns_from_metadata(
            metadata=original_metadata,
            table_name='parent',
            columns_to_drop=['id'],
        )


def test__is_list_of_strings():
    """Test `_is_list_of_strings` method"""
    assert _is_list_of_strings(['a', 'b'])
    assert not _is_list_of_strings(['a', 1])
    assert not _is_list_of_strings([1, 2])
    assert not _is_list_of_strings(1)
    assert not _is_list_of_strings('a')
