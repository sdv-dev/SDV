import operator
import re
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from sdv import version
from sdv._utils import (
    _compare_versions,
    _convert_to_timedelta,
    _create_unique_name,
    _get_datetime_format,
    _get_root_tables,
    _is_datetime_type,
    _validate_foreign_keys_not_null,
    check_sdv_versions_and_warn,
    check_synthesizer_version,
    generate_synthesizer_id,
)
from sdv.errors import SDVVersionWarning, SynthesizerInputError, VersionError
from sdv.metadata.single_table import SingleTableMetadata
from sdv.single_table.base import BaseSingleTableSynthesizer
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


def test__validate_foreign_keys_not_null():
    """Test that it crashes when foreign keys contain null data."""
    # Setup
    def side_effect_func(value):
        return ['fk'] if value == 'child_table' else []

    metadata = Mock()
    metadata._get_all_foreign_keys.side_effect = side_effect_func
    data = {
        'parent_table': pd.DataFrame({
            'id': [1, 2, 3]
        }),
        'child_table': pd.DataFrame({
            'id': [1, 2, 3],
            'fk': [None, 2, np.nan]
        })
    }

    # Run and Assert
    err_msg = re.escape(
        'The data contains null values in foreign key columns. This feature is currently '
        'unsupported. Please remove null values to fit the synthesizer.\n'
        '\n'
        'Affected columns:\n'
        "Table 'child_table', column(s) ['fk']\n"
    )
    with pytest.raises(SynthesizerInputError, match=err_msg):
        _validate_foreign_keys_not_null(metadata, data)


def test__validate_foreign_keys_not_null_no_nulls():
    """Test that it doesn't crash when foreign keys contain no null data."""
    # Setup
    def side_effect_func(value):
        return ['fk'] if value == 'child_table' else []

    metadata = Mock()
    metadata._get_all_foreign_keys.side_effect = side_effect_func
    data = {
        'parent_table': pd.DataFrame({
            'id': [1, 2, 3]
        }),
        'child_table': pd.DataFrame({
            'id': [1, 2, 3],
            'fk': [1, 2, 3]
        })
    }

    # Run
    _validate_foreign_keys_not_null(metadata, data)


@patch('sdv._utils.warnings')
def test_check_sdv_versions_and_warn_no_missmatch(mock_warnings):
    """Test that no warnings is raised when no missmatch is produced."""
    # Setup
    synthesizer = Mock()
    synthesizer._fitted_sdv_version = version.public
    synthesizer._fitted_sdv_enterprise_version = version.enterprise

    # Run
    check_sdv_versions_and_warn(synthesizer)

    # Assert
    mock_warnings.warn.assert_not_called()


def test_check_sdv_versions_and_warn_public_missmatch():
    """Test that warnings is raised when public version is missmatched."""
    # Setup
    synthesizer = Mock()
    synthesizer._fitted_sdv_version = '1.0.0'
    synthesizer._fitted_sdv_enterprise_version = version.enterprise

    # Run and Assert
    message = (
        f'You are currently on SDV version {version.public} but this synthesizer was created on '
        'version 1.0.0. The latest bug fixes and features may not be available for this '
        'synthesizer. To see these enhancements, create and train a new synthesizer on this '
        'version.'
    )
    with pytest.warns(SDVVersionWarning, match=message):
        check_sdv_versions_and_warn(synthesizer)


@patch('sdv._utils.version')
def test_check_sdv_versions_and_warn_enterprise_missmatch(mock_version):
    """Test that warnings is raised when enterprise version is missmatched."""
    # Setup
    synthesizer = Mock()
    synthesizer._fitted_sdv_version = version.public
    synthesizer._fitted_sdv_enterprise_version = '1.2.0'

    mock_version.enterprise = '1.3.0'
    mock_version.public = version.public

    # Run and Assert
    message = (
        'You are currently on SDV Enterprise version 1.3.0 but this synthesizer was created on '
        'version 1.2.0. The latest bug fixes and features may not be available for this '
        'synthesizer. To see these enhancements, create and train a new synthesizer on this '
        'version.'
    )
    with pytest.warns(SDVVersionWarning, match=message):
        check_sdv_versions_and_warn(synthesizer)


@patch('sdv._utils.version')
def test_check_sdv_versions_and_warn_public_and_enterprise_missmatch(mock_version):
    """Test that warnings is raised when both public and enterprise version missmatch."""
    # Setup
    synthesizer = Mock()
    synthesizer._fitted_sdv_version = '1.0.0'
    synthesizer._fitted_sdv_enterprise_version = '1.2.0'

    mock_version.public = '1.3.0'
    mock_version.enterprise = '1.3.3'

    # Run and Assert
    message = (
        'You are currently on SDV version 1.3.0 and SDV Enterprise version 1.3.3 but this '
        'synthesizer was created on SDV version 1.0.0 and SDV Enterprise version 1.2.0. '
        'The latest bug fixes and features may not be available for this synthesizer. '
        'To see these enhancements, create and train a new synthesizer on this version.'
    )
    with pytest.warns(SDVVersionWarning, match=message):
        check_sdv_versions_and_warn(synthesizer)


def test__compare_versions():
    """Test that _compare_versions returns True if synthesizer version is greater."""
    # Setup
    current_version = '1.2.1'
    synthesizer_version = '1.2.3'

    # Run
    result = _compare_versions(current_version, synthesizer_version)

    # Assert
    assert result is True


def test__compare_versions_equal():
    """Test that _compare_versions returns False if synthesizer version is equal."""
    # Setup
    synthesizer_version = '1.2.3'
    current_version = '1.2.3'

    # Run
    result = _compare_versions(current_version, synthesizer_version)

    # Assert
    assert result is False


def test__compare_versions_lower():
    """Test that _compare_versions returns False if synthesizer version is lower."""
    # Setup
    synthesizer_version = '1.0.3'
    current_version = '1.2.1'

    # Run
    result = _compare_versions(current_version, synthesizer_version)

    # Assert
    assert result is False


@patch('sdv._utils.version')
def test_check_synthesizer_version_public_and_enterprise_are_lower(mock_version):
    """Test that VersionError is raised when both public and enterprise version are higher."""
    # Setup
    synthesizer = Mock(
        _fitted_sdv_version='2.0.0',
        _fitted_sdv_enterprise_version='2.1.0'
    )

    mock_version.public = '1.3.0'
    mock_version.enterprise = '1.3.3'

    # Run and Assert
    message = (
        'You are currently on SDV version 1.3.0 and SDV Enterprise version 1.3.3 but this '
        'synthesizer was created on SDV version 2.0.0 and SDV Enterprise version 2.1.0. '
        'Downgrading your SDV version is not supported.'
    )
    with pytest.raises(VersionError, match=message):
        check_synthesizer_version(synthesizer)


@patch('sdv._utils.version')
def test_check_synthesizer_version_public_is_lower(mock_version):
    """Test that VersionError is raised when only public version is lower."""
    # Setup
    synthesizer = Mock(
        _fitted_sdv_version='1.4.0',
        _fitted_sdv_enterprise_version='1.2.0'
    )

    mock_version.public = '1.3.0'
    mock_version.enterprise = '1.2.0'

    # Run and Assert
    message = (
        'You are currently on SDV version 1.3.0 but this synthesizer was created on version '
        '1.4.0. Downgrading your SDV version is not supported.'
    )
    with pytest.raises(VersionError, match=message):
        check_synthesizer_version(synthesizer)


@patch('sdv._utils.version')
def test_check_synthesizer_version_enterprise_is_lower(mock_version):
    """Test that VersionError is raised when only enterprise version is lower."""
    # Setup
    synthesizer = Mock(
        _fitted_sdv_version='1.3.0',
        _fitted_sdv_enterprise_version='1.3.0'
    )

    mock_version.public = '1.3.0'
    mock_version.enterprise = '1.2.0'

    # Run and Assert
    message = (
        'You are currently on SDV Enterprise version 1.2.0 but this synthesizer was created on '
        'version 1.3.0. Downgrading your SDV version is not supported.'
    )
    with pytest.raises(VersionError, match=message):
        check_synthesizer_version(synthesizer)


@patch('sdv._utils.version')
def test_check_synthesizer_version_enterprise_is_none(mock_version):
    """Test that no VersionError is raised enterprise is None on the synthesizer."""
    # Setup
    synthesizer = Mock(
        _fitted_sdv_version='1.3.0',
        _fitted_sdv_enterprise_version=None
    )

    mock_version.public = '1.3.0'
    mock_version.enterprise = '1.2.0'

    # Run and Assert
    check_synthesizer_version(synthesizer)


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


@patch('sdv._utils.version')
def test_check_synthesizer_version_check_synthesizer_is_greater(mock_version):
    """Test that ``VersionError`` is raised when checking if synthesizer is greater.

    Ensure that this test will raise a ``VersionError`` when the synthesizer version is lower
    than the current package version.
    """
    # Setup
    synthesizer = Mock(
        _fitted_sdv_version='1.3.0',
        _fitted_sdv_enterprise_version='1.3.0'
    )

    mock_version.public = '1.3.0'
    mock_version.enterprise = '1.4.0'

    # Run and Assert
    message = (
        'You are currently on SDV Enterprise version 1.4.0 but this synthesizer was created on '
        'version 1.3.0. Fitting this synthesizer again is not supported. '
        'Please create a new synthesizer.'
    )
    with pytest.raises(VersionError, match=message):
        check_synthesizer_version(
            synthesizer,
            is_fit_method=True,
            compare_operator=operator.lt
        )


@patch('sdv._utils.version')
def test_check_synthesizer_version_check_synthesizer_is_greater_equal(mock_version):
    """Test that no ``VersionError`` is raised when versions match."""
    # Setup
    synthesizer = Mock(
        _fitted_sdv_version='1.3.0',
        _fitted_sdv_enterprise_version='1.3.0'
    )

    mock_version.public = '1.3.0'
    mock_version.enterprise = '1.3.0'

    # Run and Assert
    check_synthesizer_version(synthesizer, is_fit_method=True, compare_operator=operator.lt)


@patch('sdv._utils.version')
def test_check_synthesizer_version_check_synthesizer_is_greater_public_missmatch(mock_version):
    """Test that ``VersionError`` is raised when checking if synthesizer is greater.

    Ensure that this test will raise a ``VersionError`` when the synthesizer version is lower
    than the current package version.
    """
    # Setup
    synthesizer = Mock(
        _fitted_sdv_version='1.3.0',
        _fitted_sdv_enterprise_version='1.4.0'
    )

    mock_version.public = '1.5.0'
    mock_version.enterprise = '1.4.0'

    # Run and Assert
    message = (
        'You are currently on SDV version 1.5.0 but this synthesizer was created on '
        'version 1.3.0. Fitting this synthesizer again is not supported. '
        'Please create a new synthesizer.'
    )
    with pytest.raises(VersionError, match=message):
        check_synthesizer_version(synthesizer, is_fit_method=True, compare_operator=operator.lt)


@patch('sdv._utils.version')
def test_check_synthesizer_version_check_synthesizer_is_greater_both_missmatch(mock_version):
    """Test that ``VersionError`` is raised when public and enterprise are greater.

    Ensure that this test will raise a ``VersionError`` when the synthesizer version is lower
    than the current package version.
    """
    # Setup
    synthesizer = Mock(
        _fitted_sdv_version='1.3.0',
        _fitted_sdv_enterprise_version='1.3.2'
    )

    mock_version.public = '1.5.0'
    mock_version.enterprise = '1.4.0'

    # Run and Assert
    message = (
        'You are currently on SDV version 1.5.0 and SDV Enterprise version 1.4.0 but this '
        'synthesizer was created on SDV version 1.3.0 and SDV Enterprise version 1.3.2. '
        'Fitting this synthesizer again is not supported. Please create a new synthesizer.'
    )
    with pytest.raises(VersionError, match=message):
        check_synthesizer_version(synthesizer, is_fit_method=True, compare_operator=operator.lt)


@patch('sdv._utils.uuid')
@patch('sdv._utils.version')
def test_generate_synthesizer_id(mock_version, mock_uuid):
    """Test that ``generate_synthesizer_id`` returns the expected id."""
    # Setup
    mock_version.public = '1.0.0'
    mock_uuid.uuid4.return_value = '92aff11e-9a56-49d1-a280-990d1231a5f5'
    metadata = SingleTableMetadata()
    synthesizer = BaseSingleTableSynthesizer(metadata)

    # Run
    result = generate_synthesizer_id(synthesizer)

    # Assert
    assert result == 'BaseSingleTableSynthesizer_1.0.0_92aff11e9a5649d1a280990d1231a5f5'
