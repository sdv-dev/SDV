"""CAG _utils unit tests."""

import re
import warnings
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from sdv.cag._errors import ConstraintNotMetError
from sdv.cag._utils import (
    _convert_to_snake_case,
    _filter_old_style_constraints,
    _get_datetime_unit,
    _is_list_of_type,
    _load_constraints_from_file,
    _remove_columns_from_metadata,
    _validate_columns_not_primary_key,
    _validate_constraints,
    _validate_constraints_single_table,
    _validate_table_and_column_names,
    _validate_table_name_if_defined,
    chained_reverse_transform,
    load_constraint_from_dict,
    match_chained_datetime_precision,
    upcast_datetime_to_higher_precision,
)
from sdv.cag.base import BaseConstraint
from sdv.errors import SynthesizerInputError
from sdv.metadata.metadata import Metadata


def test__validate_columns_not_primary_key():
    """Test validating columns do not appear in primary key."""
    # Setup
    metadata = Metadata.load_from_dict({
        'tables': {
            'table': {'primary_key': 'col1'},
            'composite_table': {
                'primary_key': ['col1', 'col2'],
            },
        }
    })
    columns = ['col1', 'col2', 'col3']
    expected_single_key_error = re.escape(
        "Cannot apply constraint because 'col1' is the primary key of table 'table'."
    )
    expected_composite_key_error = re.escape(
        "Cannot apply constraint because ['col1', 'col2'] are "
        "part of the primary key for table 'composite_table'."
    )

    # Run and Assert
    with pytest.raises(ConstraintNotMetError, match=expected_single_key_error):
        _validate_columns_not_primary_key('table', columns, metadata)

    with pytest.raises(ConstraintNotMetError, match=expected_composite_key_error):
        _validate_columns_not_primary_key('composite_table', columns, metadata)


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
    with pytest.raises(ConstraintNotMetError, match=expected_not_single_table):
        _validate_table_and_column_names(None, columns_correct, metadata)

    with pytest.raises(ConstraintNotMetError, match=expected_error_wrong_table):
        _validate_table_and_column_names('wrong_table', columns_correct, metadata)

    with pytest.raises(ConstraintNotMetError, match=expected_error_wrong_columns):
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
            'child': {
                'primary_key': ['pk1', 'pk2'],
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
    with pytest.raises(ValueError, match=cannot_remove_pk):
        _remove_columns_from_metadata(
            metadata=original_metadata,
            table_name='child',
            columns_to_drop=['pk1'],
        )


def test__remove_columns_from_metadata_multiple_duplicate_columns():
    """Test `_remove_columns_from_metadata` method raises an error if primary key is dropped"""
    # Setup
    original_metadata = Metadata.load_from_dict({
        'tables': {
            'table': {
                'primary_key': 'id',
                'columns': {
                    'id': {'sdtype': 'id'},
                    'A': {'sdtype': 'numerical'},
                    'B': {'sdtype': 'numerical'},
                },
            },
        },
        'relationships': [],
    })
    columns_to_drop = ['A', 'A']

    # Run
    new_metadata = _remove_columns_from_metadata(
        metadata=original_metadata,
        table_name='table',
        columns_to_drop=columns_to_drop,
    )

    # Assert
    assert isinstance(new_metadata, Metadata)
    assert 'A' in original_metadata.tables['table'].columns
    assert 'A' not in new_metadata.tables['table'].columns
    assert 'B' in new_metadata.tables['table'].columns


def test__filter_old_style_constraints():
    """Test `_filter_old_style_constraints` method"""

    # Setup
    class DummyConstraint(BaseConstraint):
        pass

    constraint_1 = DummyConstraint()
    constraint_2 = DummyConstraint()
    old_style_constraint = {}
    expected_warning = re.escape(
        'The `add_constraints` function no longer supports constraints using the older '
        'dictionary-style definition. Such constraints will be ignored. Please supply '
        'objects from `sdv.cag` instead.'
    )

    # Run
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        result_1 = _filter_old_style_constraints([constraint_1, constraint_2])
        assert len(record) == 0

    with pytest.warns(FutureWarning, match=expected_warning):
        result_2 = _filter_old_style_constraints([constraint_1, constraint_2, old_style_constraint])

    # Assert
    assert result_1 == [constraint_1, constraint_2]
    assert result_2 == result_1


@patch('sdv.cag._utils._filter_old_style_constraints')
def test__validate_constraints(mock_filter_old_style_constraints):
    """Test `_validate_constraints` method"""
    # Setup
    constraint_1 = Mock()
    constraint_2 = Mock()
    expected_error = re.escape('Constraints must be a list of sdv.cag objects.')
    expected_warning = re.escape(
        "For these constraints to take effect, please refit the synthesizer using 'fit'."
    )

    # Run and Assert
    _validate_constraints(constraints=[constraint_1, constraint_2], synthesizer_fitted=False)
    mock_filter_old_style_constraints.assert_called_once_with([constraint_1, constraint_2])
    with pytest.raises(ValueError, match=expected_error):
        _validate_constraints(constraints=constraint_1, synthesizer_fitted=True)

    with pytest.warns(UserWarning, match=expected_warning):
        _validate_constraints(constraints=[constraint_1], synthesizer_fitted=True)


@patch('sdv.cag._utils._validate_constraints')
def test__validate_constraints_single_table(mock_validate_constraints):
    """Test the ``_validate_constraints_single_table`` method"""
    # Setup
    constraint_1 = Mock()
    constraint_1._is_single_table = True
    constraint_2 = Mock()
    constraint_2.__class__.__name__ = 'Constraint_Name'
    constraint_2._is_single_table = False
    expected_err_multi_table = re.escape(
        'Constraint `Constraint_Name` is not compatible with the single table synthesizers.'
    )
    mock_validate_constraints.side_effect = lambda constraints, _fitted: constraints

    # Run
    result = _validate_constraints_single_table(constraints=[constraint_1], synthesizer_fitted=True)
    with pytest.raises(SynthesizerInputError, match=expected_err_multi_table):
        _validate_constraints_single_table(
            constraints=[constraint_1, constraint_2], synthesizer_fitted=False
        )

    # Assert
    assert result == [constraint_1]
    mock_validate_constraints.assert_has_calls([
        call([constraint_1], True),
        call([constraint_1, constraint_2], False),
    ])


@pytest.mark.parametrize(
    ['constraint_dict', 'expected_msg'],
    [
        (
            'Constraint',
            (
                'Invalid `constraint_dict`. Expected dictionary with keys `class_name` and '
                ' `parameters`, got Constraint.'
            ),
        ),
        ({'class_name': 0, 'parameters': {}}, '`class_name` must be a string.'),
        ({'class_name': 'MockConstraint', 'parameters': 'param'}, '`parameters` must be a dict.'),
        ({'class_name': 'Unknown', 'parameters': {}}, "Unknown `constraint_class` 'Unknown'."),
    ],
)
@patch('sdv.cag._utils.importlib')
def test_load_constraints_from_dict_validates_dict(importlib_mock, constraint_dict, expected_msg):
    """Test constraint dictionary is validated when loading a constraint."""
    # Setup
    importlib_mock.import_module.return_value = Mock(spec=['MockConstraint'])

    # Run and Assert
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        load_constraint_from_dict(constraint_dict)


@patch('sdv.cag._utils.importlib')
def test_load_constraints_from_dict(importlib_mock):
    """Test utility method for loading constraints from file."""
    # Setup
    cag_mock = Mock(spec=['mock_constraint'])
    sandbox_mock = Mock(spec=['mock_sandbox_constraint', 'mock_constraint'])
    modules = {'sdv.cag': cag_mock, 'sdv.cag.sandbox': sandbox_mock}
    importlib_mock.import_module.side_effect = lambda module: modules[module]

    cag_dict = {
        'class_name': 'mock_constraint',
        'parameters': {'param1': 0, 'param2': 'a', 'param3': ['a', 'b', 'c']},
    }
    sandbox_dict = {
        'class_name': 'mock_sandbox_constraint',
        'parameters': {'param1': 'value', 'param2': 100, 'other': {'x': 'y', 'a': 'z'}},
    }

    # Run
    cag_constraint = load_constraint_from_dict(cag_dict)
    sandbox_constraint = load_constraint_from_dict(sandbox_dict)

    # Assert
    cag_mock.mock_constraint.load_constraint_from_dict.assert_called_once_with(
        parameters={'param1': 0, 'param2': 'a', 'param3': ['a', 'b', 'c']}
    )
    assert cag_constraint == cag_mock.mock_constraint.load_constraint_from_dict.return_value
    sandbox_mock.mock_sandbox_constraint.load_constraint_from_dict.assert_called_once_with(
        parameters={'param1': 'value', 'param2': 100, 'other': {'x': 'y', 'a': 'z'}}
    )
    assert sandbox_constraint == (
        sandbox_mock.mock_sandbox_constraint.load_constraint_from_dict.return_value
    )


@patch('sdv.cag._utils.open')
@patch('sdv.cag._utils.json')
@patch('sdv.cag._utils.load_constraint_from_dict')
def test__load_constraints_from_file(
    mock_load_constraint_from_dict,
    mock_json,
    mock_open,
):
    """Test loading a list of constraints from a JSON file."""
    # Setup
    constraint_dict1 = {'class_name': 'ConstraintClass1', 'parameters': {}}
    invalid_constraint_dict = {'class_name': 'UnknownConstraint', 'parameters': {}}
    constraint_dict2 = {'class_name': 'ConstraintClass2', 'parameters': {}}
    mock_json.load.return_value = [constraint_dict1, invalid_constraint_dict, constraint_dict2]

    mock_constraint1 = Mock()
    mock_constraint2 = Mock()
    mock_constraints = {
        'ConstraintClass1': mock_constraint1,
        'ConstraintClass2': mock_constraint2,
    }

    def load_constraint_from_dict_mock(constraint_dict):
        if constraint_dict['class_name'] == 'UnknownConstraint':
            raise ValueError("Unknown `constraint_class` 'UnknownConstraint'.")

        return mock_constraints[constraint_dict['class_name']]

    mock_load_constraint_from_dict.side_effect = load_constraint_from_dict_mock
    filepath = 'path/to/constraints.json'
    expected_warning = re.escape(
        "Could not load constraint ({'class_name': 'UnknownConstraint', 'parameters': {}}):\n"
        "    ValueError: Unknown `constraint_class` 'UnknownConstraint'."
    )

    # Run
    with pytest.warns(UserWarning, match=expected_warning):
        result = _load_constraints_from_file(filepath)

    # Assert
    assert result == [mock_constraint1, mock_constraint2]
    mock_open.assert_called_once_with(filepath, 'r')
    mock_json.load.assert_called_once()
    mock_load_constraint_from_dict.assert_has_calls([
        call(constraint_dict1),
        call(invalid_constraint_dict),
        call(constraint_dict2),
    ])


@pytest.mark.parametrize(
    'datetime_format, expected_unit',
    [
        ('%Y', 'Y'),
        ('%Y-%m', 'M'),
        ('%Y-%m-%d', 'D'),
        ('%Y-%m-%d %H', 'h'),
        ('%Y-%m-%d %H:%M', 'm'),
        ('%Y-%m-%d %H:%M:%S', 's'),
        ('%Y-%m-%d %H:%M:%S.%f', 'us'),
        ('%Y-%j', 'D'),
        ('%d/%b/%Y', 'D'),
        ('%B %Y', 'M'),
        ('%y', 'Y'),
    ],
)
def test_get_datetime_unit(datetime_format, expected_unit):
    """Test the `_get_datetime_unit` method."""
    # Run
    result = _get_datetime_unit(datetime_format)

    # Assert
    assert result == expected_unit


def test_get_datetime_unit_unsupported_format():
    """Test `_get_datetime_unit` raises an error for unsupported formats."""
    # Setup
    expected_error = re.escape('Unsupported datetime format')

    # Run and Assert
    with pytest.raises(ValueError, match=expected_error):
        _get_datetime_unit('unsupported')


@pytest.mark.parametrize(
    'source_format, target_format, data, use_highest, expected',
    [
        pytest.param(
            '%Y-%m-%d',
            '%Y-%m-%d %H:%M:%S',
            ['2024-11-13', '2024-11-14'],
            False,
            ['2024-11-13 00:00:00', '2024-11-14 00:00:00'],
            id='day-to-seconds-lowest',
        ),
        pytest.param(
            '%Y-%m-%d',
            '%Y-%m-%d %H:%M:%S',
            ['2024-11-13', '2024-11-14'],
            True,
            ['2024-11-13 23:59:59', '2024-11-14 23:59:59'],
            id='day-to-seconds-highest',
        ),
        pytest.param(
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f',
            ['2024-11-13 12:00:00', '2024-11-14 13:30:00'],
            False,
            ['2024-11-13 12:00:00.000000', '2024-11-14 13:30:00.000000'],
            id='seconds-to-microseconds-lowest',
        ),
        pytest.param(
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f',
            ['2024-11-13 12:00:00', '2024-11-14 13:30:00'],
            True,
            ['2024-11-13 12:00:00.999999', '2024-11-14 13:30:00.999999'],
            id='seconds-to-microseconds-highest',
        ),
        pytest.param(
            '%Y-%m',
            '%Y-%m-%d %H:%M:%S',
            ['2024-02', '2023-02'],
            True,
            ['2024-02-29 23:59:59', '2023-02-28 23:59:59'],
            id='month-to-seconds-leap-year',
        ),
        pytest.param(
            '%Y-%m',
            '%Y-%m-%d %H:%M:%S',
            ['2024-02', '2023-02'],
            False,
            ['2024-02-01 00:00:00', '2023-02-01 00:00:00'],
            id='month-to-seconds-lowest',
        ),
        pytest.param(
            '%Y',
            '%Y-%m-%d',
            ['2024', '2023'],
            True,
            ['2024-12-31', '2023-12-31'],
            id='year-to-day-highest',
        ),
        pytest.param(
            '%Y',
            '%Y-%m-%d',
            ['2024', '2023'],
            False,
            ['2024-01-01', '2023-01-01'],
            id='year-to-day-lowest',
        ),
    ],
)
def test_upcast_datetime_to_higher_precision(
    source_format, target_format, data, use_highest, expected
):
    """Test the `upcast_datetime_to_higher_precision` method."""
    # Setup
    data = np.array(data, dtype='O')
    target_unit = _get_datetime_unit(target_format)
    expected = np.array(expected, dtype=f'datetime64[{target_unit}]')

    # Run
    result = upcast_datetime_to_higher_precision(
        data, source_format, target_format, use_highest=use_highest
    )

    # Assert
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == expected.dtype


@pytest.mark.parametrize('use_highest', [True, False])
def test_upcast_datetime_to_higher_precision_with_nat(use_highest):
    """Test that upcasting preserves missing datetime values."""
    # Setup
    data = np.array(['2024-11-13', 'NaT'], dtype='datetime64[D]')

    # Run
    result = upcast_datetime_to_higher_precision(
        data,
        '%Y-%m-%d',
        '%Y-%m-%d %H:%M:%S',
        use_highest=use_highest,
    )

    # Assert
    expected_datetime = '2024-11-13 23:59:59' if use_highest else '2024-11-13 00:00:00'
    expected = np.array([expected_datetime, 'NaT'], dtype='datetime64[s]')
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    'strict_boundaries, expected_middle',
    [
        (True, '2024-11-13 00:00:00'),
        (False, '2024-11-13 23:59:59'),
    ],
)
@patch(
    'sdv.cag._utils.upcast_datetime_to_higher_precision',
    wraps=upcast_datetime_to_higher_precision,
)
def test_match_chained_datetime_precision_with_lower_precision_middle(
    mock_upcast, strict_boundaries, expected_middle
):
    """Test `match_chained_datetime_precision` with a lower-precision middle column.

    Verify that the middle column is upcasted only once for both comparisons.
    """
    # Setup
    values = [
        np.array(['2024-11-12 12:00:00'], dtype='O'),
        np.array(['2024-11-13'], dtype='O'),
        np.array(['2024-11-14 12:00:00'], dtype='O'),
    ]
    datetime_formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d',
        '%Y-%m-%d %H:%M:%S',
    ]
    expected = [
        np.array(['2024-11-12 12:00:00'], dtype='datetime64[ns]'),
        np.array([expected_middle], dtype='datetime64[ns]'),
        np.array(['2024-11-14 12:00:00'], dtype='datetime64[ns]'),
    ]

    # Run
    result = match_chained_datetime_precision(
        values, datetime_formats, strict_boundaries=strict_boundaries
    )

    # Assert
    for actual, expected_value in zip(result, expected):
        np.testing.assert_array_equal(actual, expected_value)

    mock_upcast.assert_called_once()
    args, kwargs = mock_upcast.call_args
    assert args[1:] == (datetime_formats[1], datetime_formats[0])
    assert kwargs == {'use_highest': not strict_boundaries}
    low_diff = result[1] - result[0]
    high_diff = result[2] - result[1]
    np.testing.assert_array_equal(low_diff + high_diff, result[2] - result[0])


@pytest.mark.parametrize(
    'strict_boundaries, expected_low, expected_high',
    [
        (
            True,
            '2024-11-12 23:59:59',
            '2024-11-14 00:00:00',
        ),
        (
            False,
            '2024-11-12 00:00:00',
            '2024-11-14 23:59:59',
        ),
    ],
)
def test_match_chained_datetime_precision_with_lower_precision_boundaries(
    strict_boundaries, expected_low, expected_high
):
    """Test `match_chained_datetime_precision` with the middle column having higher precision.

    Verify that the low and high columns are upcasted to match the middle column's precision.
    """
    # Setup
    values = [
        np.array(['2024-11-12'], dtype='O'),
        np.array(['2024-11-13 12:00:00'], dtype='O'),
        np.array(['2024-11-14'], dtype='O'),
    ]
    datetime_formats = [
        '%Y-%m-%d',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d',
    ]
    expected = [
        np.array([expected_low], dtype='datetime64[ns]'),
        np.array(['2024-11-13 12:00:00'], dtype='datetime64[ns]'),
        np.array([expected_high], dtype='datetime64[ns]'),
    ]

    # Run
    result = match_chained_datetime_precision(
        values, datetime_formats, strict_boundaries=strict_boundaries
    )

    # Assert
    for actual, expected_value in zip(result, expected):
        np.testing.assert_array_equal(actual, expected_value)


@pytest.mark.parametrize(
    'strict_boundaries, expected_low, expected_middle',
    [
        (
            True,
            '2024-11-12 23:59:59.999999',
            '2024-11-13 12:00:00.000000',
        ),
        (
            False,
            '2024-11-12 00:00:00.000000',
            '2024-11-13 12:00:00.999999',
        ),
    ],
)
def test_match_chained_datetime_precision_with_three_precisions(
    strict_boundaries, expected_low, expected_middle
):
    """Test `match_chained_datetime_precision` with three different precisions.

    Verify that all columns are upcasted to the highest precision.
    """
    # Setup
    values = [
        np.array(['2024-11-12'], dtype='O'),
        np.array(['2024-11-13 12:00:00'], dtype='O'),
        np.array(['2024-11-14 12:00:00.123456'], dtype='O'),
    ]
    datetime_formats = [
        '%Y-%m-%d',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M:%S.%f',
    ]
    expected = [
        np.array([expected_low], dtype='datetime64[ns]'),
        np.array([expected_middle], dtype='datetime64[ns]'),
        np.array(['2024-11-14 12:00:00.123456'], dtype='datetime64[ns]'),
    ]

    # Run
    result = match_chained_datetime_precision(
        values, datetime_formats, strict_boundaries=strict_boundaries
    )

    # Assert
    for actual, expected_value in zip(result, expected):
        np.testing.assert_array_equal(actual, expected_value)
        assert actual.dtype == np.dtype('datetime64[ns]')


@patch('sdv.cag._utils.upcast_datetime_to_higher_precision')
def test_match_chained_datetime_precision_with_same_precision(mock_upcast):
    """Test `match_chained_datetime_precision` with matching precisions.

    Verify that no unnecessary upcasting occurs.
    """
    # Setup
    values = [
        np.array(['2024-11-12 12:00:00'], dtype='O'),
        np.array(['2024-11-13 12:00:00'], dtype='O'),
        np.array(['2024-11-14 12:00:00'], dtype='O'),
    ]
    datetime_formats = ['%Y-%m-%d %H:%M:%S'] * 3
    expected = [np.array(value, dtype='datetime64[ns]') for value in values]

    # Run
    result = match_chained_datetime_precision(values, datetime_formats)

    # Assert
    mock_upcast.assert_not_called()
    for actual, expected_value in zip(result, expected):
        np.testing.assert_array_equal(actual, expected_value)


@pytest.mark.parametrize(
    'strict_boundaries, expected_low, expected_high',
    [
        (
            True,
            '2024-11-12 23:59:59.999999999',
            '2024-11-14 12:00:00.000000000',
        ),
        (
            False,
            '2024-11-12 00:00:00.000000000',
            '2024-11-14 12:00:00.999999999',
        ),
    ],
)
def test_match_chained_datetime_precision_with_missing_format(
    strict_boundaries, expected_low, expected_high
):
    """Test `match_chained_datetime_precision` with a missing format."""
    # Setup
    values = [
        np.array(['2024-11-12'], dtype='O'),
        np.array(['2024-11-13 12:00:00.123456'], dtype='O'),
        np.array(['2024-11-14 12:00:00'], dtype='O'),
    ]
    formats = [
        '%Y-%m-%d',
        None,
        '%Y-%m-%d %H:%M:%S',
    ]
    expected = [
        np.array([expected_low], dtype='datetime64[ns]'),
        np.array(['2024-11-13 12:00:00.123456'], dtype='datetime64[ns]'),
        np.array([expected_high], dtype='datetime64[ns]'),
    ]

    # Run
    result = match_chained_datetime_precision(values, formats, strict_boundaries=strict_boundaries)

    # Assert
    for actual, expected_value in zip(result, expected):
        np.testing.assert_array_equal(actual, expected_value)


def test_match_chained_datetime_precision_without_formats():
    """Test `match_chained_datetime_precision` without datetime formats."""
    # Setup
    values = [
        np.array(['2024-11-12 12:30:00.123456'], dtype='O'),
        np.array(['2024-11-13 13:45:00.654321'], dtype='O'),
    ]
    expected = [np.asarray(value, dtype='datetime64[ns]') for value in values]

    # Run
    result = match_chained_datetime_precision(values, [None, None])

    # Assert
    for actual, expected_value in zip(result, expected):
        np.testing.assert_array_equal(actual, expected_value)


@pytest.mark.parametrize(
    'strict_boundaries, low_diff, high_diff, expected_middle',
    [
        (
            True,
            43_200,
            129_600,
            '2024-11-13 00:00:00',
        ),
        (
            False,
            129_599,
            43_201,
            '2024-11-13 23:59:59',
        ),
    ],
)
def test_chained_reverse_transform_with_lower_precision_middle(
    strict_boundaries, low_diff, high_diff, expected_middle
):
    """Test `chained_reverse_transform` with a lower-precision middle column.

    Verify that the same reconstructed middle value is used throughout the chain.
    """
    # Setup
    index = pd.Index([10, 20])
    low = np.array(
        [
            '2024-11-12 12:00:00',
            '2024-11-12 12:00:00',
        ],
        dtype='O',
    )
    diff_columns = [
        pd.to_timedelta([low_diff, low_diff], unit='s'),
        pd.to_timedelta([high_diff, high_diff], unit='s'),
    ]
    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d',
        '%Y-%m-%d %H:%M:%S',
    ]

    # Run
    middle, high = chained_reverse_transform(
        low,
        diff_columns,
        formats,
        dtype=np.dtype('O'),
        strict_boundaries=strict_boundaries,
        index=index,
    )

    # Assert
    expected_middle = pd.Series(pd.to_datetime([expected_middle] * 2), index=index)
    expected_high = pd.Series(pd.to_datetime(['2024-11-14 12:00:00'] * 2), index=index)

    pd.testing.assert_series_equal(pd.to_datetime(middle), expected_middle)
    pd.testing.assert_series_equal(pd.to_datetime(high), expected_high)
