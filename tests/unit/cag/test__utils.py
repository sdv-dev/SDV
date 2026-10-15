"""CAG _utils unit tests."""

import re
from decimal import Decimal
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from sdv._utils import _cast_to_datetime64
from sdv.cag._errors import ConstraintNotMetError
from sdv.cag._utils import (
    _cast_to_type,
    _convert_to_snake_case,
    _is_list_of_type,
    _load_constraints_from_file,
    _remove_columns_from_metadata,
    _validate_columns_not_primary_key,
    _validate_constraints,
    _validate_constraints_single_table,
    _validate_table_and_column_names,
    _validate_table_name_if_defined,
    _warn_if_timezone_aware_formats,
    compute_nans_column,
    downcast_datetime_to_lower_precision,
    format_datetime_array,
    get_datetime_diff,
    get_datetime_format_precision,
    get_lower_precision_format,
    get_mappable_combination,
    get_nan_component_value,
    load_constraint_from_dict,
    logit,
    match_datetime_precision,
    matches_datetime_format,
    revert_nans_columns,
    sigmoid,
)
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
        'METADATA_SPEC_VERSION': 'V2',
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


def test__validate_constraints():
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


def test__cast_to_type():
    """Test the ``_cast_to_type`` function.

    Given ``pd.Series``, ``np.array`` or just a numeric value, it should
    cast it to the given ``type``.

    Input:
        - pd.Series
        - np.array
        - numeric
        - Type
    Output:
        The values should be casted to the expected ``type``.
    """
    # Setup
    value = 88
    series = pd.Series([1, 2, 3])
    array = np.array([1, 2, 3])

    # Run
    res_value = _cast_to_type(value, float)
    res_series = _cast_to_type(series, float)
    res_array = _cast_to_type(array, float)

    # Assert
    assert isinstance(res_value, float)
    assert res_series.dtype == float
    assert res_array.dtype == float


def test_logit():
    """Test the ``logit`` function.

    Setup:
        - Compute ``expected_res`` with the ``high`` and ``low`` values.
    Input:
        - ``data`` a number.
        - ``low`` and ``high`` numbers.
    Output:
        The result of the scaled logit.
    """
    # Setup
    high, low = 100, 49
    _data = (88 - low) / (high - low)
    _data = Decimal(_data) * Decimal(0.95) + Decimal(0.025)
    _data = float(_data)
    expected_res = np.log(_data / (1.0 - _data))

    data = 88

    # Run
    res = logit(data, low, high)

    # Assert

    assert res == expected_res


def test_sigmoid():
    """Test the ``sigmoid`` function.

    Setup:
        - Compute ``expected_res`` with the ``high`` and ``low`` values.
    Input:
        - ``data`` a number.
        - ``low`` and ``high`` numbers.
    Output:
        The result of sigmoid.
    """
    # Setup
    high, low = 100, 49
    _data = data = 1.1064708752806303

    _data = 1 / (1 + np.exp(-data))
    _data = (Decimal(_data) - Decimal(0.025)) / Decimal(0.95)
    _data = float(_data)
    expected_res = _data * (high - low) + low

    # Run
    res = sigmoid(data, low, high)

    # Assert
    assert res == expected_res


def test_matches_datetime_format():
    """Test the ``matches_datetime_format`` method.

    If the provided datetime string matches the format, then it should return True.

    Input:
        - Datetime string that matches the format

    Output:
        - True
    """
    # Run
    result = matches_datetime_format('1/1/2020', '%m/%d/%Y')

    # Assert
    assert result is True


def test_matches_datetime_format_does_not_match():
    """Test the ``matches_datetime_format`` method.

    If the provided datetime string does not match the format, then it should return False.

    Input:
        - Datetime string that does not match the format

    Output:
        - False
    """
    # Run
    result = matches_datetime_format('1-1-2020', '%m/%d/%Y')

    # Assert
    assert result is False


def test_matches_datetime_format_bad_value():
    """Test the ``matches_datetime_format`` method.

    If the provided value is not a string, then it should return False.

    Input:
        - int and a datetime format

    Output:
        - False
    """
    # Run
    result = matches_datetime_format(10, '%m/%d/%Y')

    # Assert
    assert result is False


def test_get_nan_component_value():
    """Test the ``get_nan_component_value`` method."""
    # Setup
    row = pd.Series([np.nan, 2, np.nan, 4], index=['a', 'b', 'c', 'd'])

    # Run
    result = get_nan_component_value(row)

    # Assert
    assert result == 'a, c'


def test_compute_nans_columns():
    """Test the ``compute_nans_columns`` method."""
    # Setup
    data = pd.DataFrame({
        'a': [1, np.nan, 3, np.nan],
        'b': [np.nan, 2, 3, np.nan],
        'c': [1, np.nan, 3, np.nan],
    })

    # Run
    output = compute_nans_column(data, ['a', 'b', 'c'])
    expected_output = pd.Series(['b', 'a, c', 'None', 'a, b, c'], name='a#b#c.nan_component')

    # Assert
    pd.testing.assert_series_equal(output, expected_output)


def test_compute_nans_columns_without_nan():
    """Test the ``compute_nans_columns`` method when there are no nans."""
    # Setup
    data = pd.DataFrame({'a': [1, 2, 3, 2], 'b': [2.5, 2, 3, 2.5], 'c': [1, 2, 3, 2]})

    # Run
    output = compute_nans_column(data, ['a', 'b', 'c'])

    # Assert
    assert output is None


def test_revert_nans_columns():
    """Test the ``revert_nans_columns`` method."""
    # Setup
    data = pd.DataFrame({
        'a': [1, 2, 3, 2],
        'b': [2.5, 2, 3, 2.5],
        'c': [1, 2, 3, 2],
        'a#b#c.nan_component': ['b', 'a, c', 'None', 'a, b, c'],
    })
    nan_column_name = 'a#b#c.nan_component'

    # Run
    result = revert_nans_columns(data, nan_column_name)

    expected_data = pd.DataFrame({
        'a': [1, np.nan, 3, np.nan],
        'b': [np.nan, 2, 3, np.nan],
        'c': [1, np.nan, 3, np.nan],
    })

    # Assert
    pd.testing.assert_frame_equal(result, expected_data)


def test_get_datetime_diff():
    """Test the ``_get_datetime_diff`` method.

    The method is expected to compute the difference between the high and low
    datetime columns, treating missing values as NaN.
    """
    # Setup
    high = pd.Series(['2022-02-02', '', '2023-01-02']).to_numpy()
    low = pd.Series(['2022-02-01', '2022-02-02', '2023-01-01']).to_numpy()
    expected = np.array([8.64e13, np.nan, 8.64e13])

    # Run
    diff = get_datetime_diff(high, low, dtype='O')

    # Assert
    assert np.array_equal(expected, diff, equal_nan=True)


def test_get_datetime_diff_with_format_precision_mismatch():
    """Test `get_datetime_diff` with miss matching datetime formats."""
    # Setup
    high = np.array(['2024-11-13 12:00:00.123', '2024-11-13 13:00:00.456'], dtype='O')
    low = np.array(['2024-11-13 12:00:00', '2024-11-13 13:00:00'], dtype='O')
    high_format = '%Y-%m-%d %H:%M:%S.%f'
    low_format = '%Y-%m-%d %H:%M:%S'
    expected_diff = np.array([0.0, 0.0], dtype=np.float64)

    # Run
    result = get_datetime_diff(
        high, low, high_datetime_format=high_format, low_datetime_format=low_format
    )

    # Assert
    np.testing.assert_array_almost_equal(result, expected_diff)


def test_get_mappable_combination():
    """Test the ``get_mappable_combination`` method."""
    # Setup
    already_mappable = ('a', 1, 1.2, 'b')
    not_mappable = ('a', 1, np.nan, 'b')

    # Run
    result_already_mappable = get_mappable_combination(already_mappable)
    result_not_mappable = get_mappable_combination(not_mappable)

    # Assert
    expected_result_not_mappable = ('a', 1, None, 'b')
    assert result_already_mappable == already_mappable
    assert result_not_mappable == expected_result_not_mappable


def test_get_datetime_format_precision_seconds():
    """Test `get_datetime_format_precision` with second-level precision."""
    # Setup
    format_str = '%Y-%m-%d %H:%M:%S'
    expected_precision = 6

    # Run
    result = get_datetime_format_precision(format_str)

    # Assert
    assert result == expected_precision


def test_get_datetime_format_precision_microseconds():
    """Test `get_datetime_format_precision` with microsecond-level precision."""
    # Setup
    format_str = '%Y-%m-%d %H:%M:%S.%f'
    expected_precision = 7

    # Run
    result = get_datetime_format_precision(format_str)

    # Assert
    assert result == expected_precision


def test_get_datetime_format_precision_minutes():
    """Test `get_datetime_format_precision` with minute-level precision."""
    # Setup
    format_str = '%Y-%m-%d %H:%M'
    expected_precision = 5

    # Run
    result = get_datetime_format_precision(format_str)

    # Assert
    assert result == expected_precision


def test_get_datetime_format_precision_days():
    """Test `get_datetime_format_precision` with day-level precision."""
    # Setup
    format_str = '%Y-%m-%d'
    expected_precision = 3

    # Run
    result = get_datetime_format_precision(format_str)

    # Assert
    assert result == expected_precision


def test_get_datetime_format_precision_no_precision():
    """Test `get_datetime_format_precision` with no precision format."""
    # Setup
    format_str = '%Y'
    expected_precision = 1

    # Run
    result = get_datetime_format_precision(format_str)

    # Assert
    assert result == expected_precision


def test_get_datetime_format_precision_mixed_format_higher_precision():
    """Test `get_datetime_format_precision` with mixed higher-precision format."""
    # Setup
    format_str = '%Y-%m-%d %H:%M:%S.%f %z'
    expected_precision = 7

    # Run
    result = get_datetime_format_precision(format_str)

    # Assert
    assert result == expected_precision


def test_get_lower_precision_format_with_different_precision():
    """Test `get_lower_precision_format` with different precision levels."""
    # Setup
    primary_format = '%Y-%m-%d %H:%M:%S'
    secondary_format = '%Y-%m-%d %H:%M:%S.%f'

    # Run
    result = get_lower_precision_format(primary_format, secondary_format)

    # Assert
    assert result == primary_format


def test_get_lower_precision_format_with_equal_precision():
    """Test `get_lower_precision_format` when both formats have the same precision."""
    # Setup
    primary_format = '%Y-%m-%d %H:%M:%S'
    secondary_format = '%Y-%m-%d %H:%M:%S'

    # Run
    result = get_lower_precision_format(primary_format, secondary_format)

    # Assert
    assert result == secondary_format == primary_format


def test_get_lower_precision_format_with_date_only():
    """Test `get_lower_precision_format` with date-only formats."""
    # Setup
    primary_format = '%Y-%m-%d'
    secondary_format = '%Y-%m'

    # Run
    result = get_lower_precision_format(primary_format, secondary_format)

    # Assert
    assert result == secondary_format


def test_get_lower_precision_format_with_week_and_day_formats():
    """Test `get_lower_precision_format` with week and day level formats."""
    # Setup
    primary_format = '%Y-%W'
    secondary_format = '%Y-%m-%d'

    # Run
    result = get_lower_precision_format(primary_format, secondary_format)

    # Assert
    assert result == secondary_format


def test_downcast_datetime_to_lower_precision():
    """Test `downcast_datetime_to_lower_precision` to ensure datetime downcasting."""
    # Setup
    data = np.array(
        ['2024-11-13 12:30:45.123456789', '2024-11-13 13:45:30.987654321'], dtype='datetime64[ns]'
    )
    target_format = '%Y-%m-%d %H:%M:%S'
    expected_result = np.array(['2024-11-13 12:30:45', '2024-11-13 13:45:30'], dtype='O')

    # Run
    result = downcast_datetime_to_lower_precision(data, target_format)

    # Assert
    np.testing.assert_array_equal(result, _cast_to_datetime64(expected_result))


def test_downcast_datetime_to_lower_precision_to_day():
    """Test `downcast_datetime_to_lower_precision` to downcast datetime to day precision."""
    # Setup
    data = np.array(
        ['2024-11-13 12:30:45.123456789', '2024-11-14 13:45:30.987654321'], dtype='datetime64[ns]'
    )
    target_format = '%Y-%m-%d'  # Downcasting to day precision
    expected_result = np.array(['2024-11-13', '2024-11-14'], dtype='O')

    # Run
    result = downcast_datetime_to_lower_precision(data, target_format)

    # Assert
    np.testing.assert_array_equal(result, _cast_to_datetime64(expected_result))


def test_format_datetime_array_with_lower_precision_format():
    """Test `format_datetime_array` formatting datetime array to a lower-precision format."""
    # Setup
    datetime_array = np.array(
        ['2024-11-13 12:30:45.123456789', '2024-11-13 13:45:30.987654321'], dtype='datetime64[ns]'
    )
    target_format = '%Y-%m-%d %H:%M:%S'
    expected_result = np.array(['2024-11-13 12:30:45', '2024-11-13 13:45:30'], dtype='O')

    # Run
    result = format_datetime_array(datetime_array, target_format)

    # Assert
    np.testing.assert_array_equal(result, expected_result)


@patch('sdv.cag._utils.downcast_datetime_to_lower_precision')
def test_match_datetime_precision_low_has_higher_precision(mock_downcast):
    """Test `match_datetime_precision` when `low` has higher precision than `high`.

    This test checks that if the `low` array has a more precise format than `high`,
    `low` is downcasted to match the `high` format.
    """
    # Setup
    low = np.array(['2024-11-13 10:34:45.123456', '2024-11-14 12:20:10.654321'], dtype='O')
    high = np.array(['2024-11-13 10:34:45', '2024-11-14 12:20:10'], dtype='O')
    low_format = '%Y-%m-%d %H:%M:%S.%f'
    high_format = '%Y-%m-%d %H:%M:%S'
    expected_low = np.array(['2024-11-13 10:34:45', '2024-11-14 12:20:10'], dtype='O')

    # Set the return value of the mock to simulate downcasting
    mock_downcast.return_value = expected_low

    # Run
    result_low, result_high = match_datetime_precision(low, high, low_format, high_format)

    # Assert
    mock_downcast.assert_called_once_with(low, high_format)
    np.testing.assert_array_equal(result_low, expected_low)
    np.testing.assert_array_equal(result_high, high)


@patch('sdv.cag._utils.downcast_datetime_to_lower_precision')
def test_match_datetime_precision_high_has_higher_precision(mock_downcast):
    """Test `match_datetime_precision` when `high` has higher precision than `low`.

    This test checks that if the `high` array has a more precise format than `low`,
    `high` is downcasted to match the `low` format.
    """
    # Setup
    low = np.array(['2024-11-13 10:34:45', '2024-11-14 12:20:10'], dtype='O')
    high = np.array(['2024-11-13 10:34:45.123456', '2024-11-14 12:20:10.654321'], dtype='O')
    low_format = '%Y-%m-%d %H:%M:%S'
    high_format = '%Y-%m-%d %H:%M:%S.%f'
    expected_high = np.array(['2024-11-13 10:34:45', '2024-11-14 12:20:10'], dtype='O')

    # Set the return value of the mock to simulate downcasting
    mock_downcast.return_value = expected_high

    # Run
    result_low, result_high = match_datetime_precision(low, high, low_format, high_format)

    # Assert
    mock_downcast.assert_called_once_with(high, low_format)
    np.testing.assert_array_equal(result_low, low)
    np.testing.assert_array_equal(result_high, expected_high)


@patch('sdv.cag._utils.warnings.warn')
def test_warn_if_timezone_aware_formats_warns(mock_warn):
    """Test it calls warnings.warn if timezone-aware format is detected."""
    # Setup
    formats_with_timezone = ['%Y-%m-%d %H:%M:%S%z', None, '%Y %m %d %Z']

    # Run
    _warn_if_timezone_aware_formats(formats_with_timezone)

    # Assert
    expected_message = (
        'Timezone information in datetime formats will be ignored when evaluating '
        'constraints. All datetime values will be treated as naive (timezone-unaware). '
        'Support for timezone-aware constraints will be added in a future release.'
    )
    mock_warn.assert_called_once_with(expected_message, UserWarning)


@patch('sdv.cag._utils.warnings.warn')
def test_warn_if_timezone_aware_formats_no_warning(mock_warn):
    """Test it does not call warnings.warn if all formats are timezone-naive."""
    # Setup
    formats_without_timezone = ['%Y-%m-%d', '%d %b %Y', None]

    # Run
    _warn_if_timezone_aware_formats(formats_without_timezone)

    # Assert
    mock_warn.assert_not_called()
