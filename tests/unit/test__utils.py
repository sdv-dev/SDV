from collections import defaultdict
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from sdv._utils import (
    _convert_to_timedelta, _create_unique_name, _get_datetime_format, _get_relationship_for_child,
    _get_relationship_for_parent, _get_root_tables, _get_rows_to_drop, _is_datetime_type)
from tests.utils import SeriesMatcher


@patch('sdv._utils.pd.to_timedelta')
def test__convert_to_timedelta(to_timedelta_mock):
    """Test that nans and values are properly converted to timedeltas."""
    # Setup
    column = pd.Series([7200, 3600, np.nan])
    to_timedelta_mock.return_value = pd.Series([
        pd.Timedelta(hours=1),
        pd.Timedelta(hours=2),
        pd.Timedelta(hours=0)
    ], dtype='timedelta64[ns]')

    # Run
    converted_column = _convert_to_timedelta(column)

    # Assert
    to_timedelta_mock.assert_called_with(SeriesMatcher(pd.Series([7200, 3600, 0.0])))
    expected_column = pd.Series([
        pd.Timedelta(hours=1),
        pd.Timedelta(hours=2),
        pd.NaT
    ], dtype='timedelta64[ns]')
    pd.testing.assert_series_equal(converted_column, expected_column)


def test__get_datetime_format():
    """Test the ``_get_datetime_format``.

    Setup:
        - string value representing datetime.
        - list of values with a datetime.
        - series with a datetime.

    Output:
        - The expected output is the format of the datetime representation.
    """
    # Setup
    string_value = '2021-02-02'
    list_value = [np.nan, '2021-02-02']
    series_value = pd.Series(['2021-02-02T12:10:59'])

    # Run
    string_out = _get_datetime_format(string_value)
    list_out = _get_datetime_format(list_value)
    series_out = _get_datetime_format(series_value)

    # Assert
    expected_output = '%Y-%m-%d'
    assert string_out == expected_output
    assert list_out == expected_output
    assert series_out == '%Y-%m-%dT%H:%M:%S'


def test__is_datetime_type_with_datetime_series():
    """Test the ``_is_datetime_type`` function when a datetime series is passed.

    Expect to return True when a datetime series is passed.

    Input:
    - A pandas.Series of type `datetime64[ns]`
    Output:
    - True
    """
    # Setup
    data = pd.Series([
        pd.to_datetime('2020-01-01'),
        pd.to_datetime('2020-01-02'),
        pd.to_datetime('2020-01-03')
    ],
    )

    # Run
    is_datetime = _is_datetime_type(data)

    # Assert
    assert is_datetime


def test__is_datetime_type_with_mixed_array():
    """Test the ``_is_datetime_type`` function with a list of mixed datetime types."""
    # Setup
    data = [
        pd.to_datetime('2020-01-01'),
        '1890-03-05',
        pd.Timestamp('01-01-01'),
        datetime(2020, 1, 1),
        np.nan
    ]

    # Run
    is_datetime = _is_datetime_type(data)

    # Assert
    assert is_datetime


def test__is_datetime_type_with_invalid_strings_in_list():
    """Test the ``_is_datetime_type`` function with a invalid datetime in a list."""
    # Setup
    data = [
        pd.to_datetime('2020-01-01'),
        '1890-03-05',
        pd.Timestamp('01-01-01'),
        datetime(2020, 1, 1),
        'invalid',
        np.nan
    ]

    # Run
    is_datetime = _is_datetime_type(data)

    # Assert
    assert is_datetime is False


def test__is_datetime_type_with_datetime():
    """Test the ``_is_datetime_type`` function when a datetime is passed.

    Expect to return True when a datetime variable is passed.

    Input:
    - datetime.Datetime
    Output:
    - True
    """
    # Setup
    data = datetime(2020, 1, 1)

    # Run
    is_datetime = _is_datetime_type(data)

    # Assert
    assert is_datetime


def test__is_datetime_type_with_timestamp():
    """Test the ``_is_datetime_type`` function when a Timestamp is passed.

    Expect to return True when a datetime variable is passed.

    Input:
    - datetime.Datetime
    Output:
    - True
    """
    # Setup
    data = pd.Timestamp('2020-01-10')
    is_datetime = _is_datetime_type(data)

    # Assert
    assert is_datetime


def test__is_datetime_type_with_pandas_datetime():
    """Test the ``_is_datetime_type`` function when a pandas.datetime is passed.

    Expect to return True when a datetime variable is passed.

    Input:
    - pandas.Datetime
    Output:
    - True
    """
    # Setup
    data = pd.to_datetime('2020-01-01')

    # Run
    is_datetime = _is_datetime_type(data)

    # Assert
    assert is_datetime


def test__is_datetime_type_with_int():
    """Test the ``_is_datetime_type`` function when an int is passed.

    Expect to return False when an int variable is passed.

    Input:
    - int
    Output:
    - False
    """
    # Setup
    data = 2

    # Run
    is_datetime = _is_datetime_type(data)

    # Assert
    assert is_datetime is False


def test__is_datetime_type_with_datetime_str():
    """Test the ``_is_datetime_type`` function when an valid datetime string is passed.

    Expect to return True when a valid string representing datetime is passed.

    Input:
    - string
    Output:
    - True
    """
    # Setup
    value = '2021-02-02'

    # Run
    is_datetime = _is_datetime_type(value)

    # Assert
    assert is_datetime


def test__is_datetime_type_with_invalid_str():
    """Test the ``_is_datetime_type`` function when an invalid string is passed.

    Expect to return False when an invalid string is passed.

    Input:
    - string
    Output:
    - False
    """
    # Setup
    value = 'abcd'

    # Run
    is_datetime = _is_datetime_type(value)

    # Assert
    assert is_datetime is False


def test__is_datetime_type_with_int_series():
    """Test the ``_is_datetime_type`` function when an int series is passed.

    Expect to return False when an int series variable is passed.

    Input:
    -  pd.Series of type int
    Output:
    - False
    """
    # Setup
    data = pd.Series([1, 2, 3, 4])

    # Run
    is_datetime = _is_datetime_type(data)

    # Assert
    assert is_datetime is False


def test__create_unique_name():
    """Test the ``_create_unique_name`` method."""
    # Setup
    name = 'name'
    existing_names = ['name', 'name_', 'name__']

    # Run
    result = _create_unique_name(name, existing_names)

    # Assert
    assert result == 'name___'


def test__get_root_tables():
    """Test the ``_get_root_tables`` method."""
    # Setup
    relationships = [
        {'parent_table_name': 'parent', 'child_table_name': 'child'},
        {'parent_table_name': 'child', 'child_table_name': 'grandchild'},
        {'parent_table_name': 'parent', 'child_table_name': 'grandchild'}
    ]

    # Run
    result = _get_root_tables(relationships)

    # Assert
    assert result == {'parent'}


def test__get_relationship_for_child():
    """Test the ``_get_relationship_for_child`` method."""
    # Setup
    relationships = [
        {'parent_table_name': 'parent', 'child_table_name': 'child'},
        {'parent_table_name': 'child', 'child_table_name': 'grandchild'},
        {'parent_table_name': 'parent', 'child_table_name': 'grandchild'}
    ]

    # Run
    result = _get_relationship_for_child(relationships, 'grandchild')

    # Assert
    expected_result = [
        {'parent_table_name': 'child', 'child_table_name': 'grandchild'},
        {'parent_table_name': 'parent', 'child_table_name': 'grandchild'}
    ]
    assert result == expected_result


def test__get_relationship_for_parent():
    """Test the ``_get_relationship_for_parent`` method."""
    # Setup
    relationships = [
        {'parent_table_name': 'parent', 'child_table_name': 'child'},
        {'parent_table_name': 'child', 'child_table_name': 'grandchild'},
        {'parent_table_name': 'parent', 'child_table_name': 'grandchild'}
    ]

    # Run
    result = _get_relationship_for_parent(relationships, 'parent')

    # Assert
    expected_result = [
        {'parent_table_name': 'parent', 'child_table_name': 'child'},
        {'parent_table_name': 'parent', 'child_table_name': 'grandchild'}
    ]
    assert result == expected_result


def test__get_rows_to_drop():
    """Test the ``_get_rows_to_drop`` method.

    In the child table:
        - Index 4 is removed because its foreign key doesn't match any primary key in the parent
        table.

    In the grandchild table:
        - Index 2 is removed because its foreign key doesn't match any primary key in the child
        table.
        - Index 4 is removed due to its foreign key not aligning with any parent table primary key.
        - Index 0 is removed following the removal of index 4 from the child table, which
        invalidates the foreign key set to 9 in the grandchild table.
    """
    # Setup
    relationships = [
        {
            'parent_table_name': 'parent',
            'child_table_name': 'child',
            'parent_primary_key': 'id_parent',
            'child_foreign_key': 'parent_foreign_key'
        },
        {
            'parent_table_name': 'child',
            'child_table_name': 'grandchild',
            'parent_primary_key': 'id_child',
            'child_foreign_key': 'child_foreign_key'
        },
        {
            'parent_table_name': 'parent',
            'child_table_name': 'grandchild',
            'parent_primary_key': 'id_parent',
            'child_foreign_key': 'parent_foreign_key'
        }
    ]

    metadata = Mock()
    metadata.relationships = relationships
    metadata.tables = {
        'parent': Mock(primary_key='id_parent'),
        'child': Mock(primary_key='id_child'),
        'grandchild': Mock(primary_key='id_grandchild')
    }

    data = {
        'parent': pd.DataFrame({
            'id_parent': [0, 1, 2, 3, 4],
            'A': [True, True, False, True, False],
        }),
        'child': pd.DataFrame({
            'parent_foreign_key': [0, 1, 2, 2, 5],
            'id_child': [5, 6, 7, 8, 9],
            'B': ['Yes', 'No', 'No', 'No', 'No']
        }),
        'grandchild': pd.DataFrame({
            'parent_foreign_key': [0, 1, 2, 2, 6],
            'child_foreign_key': [9, 5, 11, 6, 4],
            'C': ['Yes', 'No', 'No', 'No', 'No']
        })
    }

    # Run
    result = _get_rows_to_drop(metadata, data)

    # Assert
    expected_result = defaultdict(set, {
        'child': {4},
        'grandchild': {0, 2, 4},
        'parent': set()
    })
    assert result == expected_result