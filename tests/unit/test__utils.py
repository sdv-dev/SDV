import operator
import re
import string
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
from pandas.api.types import is_datetime64_any_dtype, is_object_dtype, is_string_dtype
from rdt.transformers.numerical import FloatFormatter

from sdv import version
from sdv._utils import (
    _check_regex_format,
    _compare_versions,
    _convert_to_timedelta,
    _create_unique_name,
    _datetime_string_matches_format,
    _get_chars_for_option,
    _get_datetime_format,
    _get_root_tables,
    _get_transformer_init_kwargs,
    _is_datetime_type,
    _is_numerical,
    _validate_datetime_format,
    _validate_foreign_keys_not_null,
    check_sdv_versions_and_warn,
    check_synthesizer_version,
    generate_synthesizer_id,
    get_possible_chars,
)
from sdv.errors import SDVVersionWarning, SynthesizerInputError, VersionError
from sdv.metadata.single_table import SingleTableMetadata
from sdv.single_table.base import BaseSingleTableSynthesizer
from tests.utils import SeriesMatcher

try:
    from re import _parser as sre_parse
except ImportError:
    import sre_parse

is_pandas_two_installed = Version(pd.__version__) >= Version('2.0.0')


@patch('sdv._utils.pd.to_timedelta')
def test__convert_to_timedelta(to_timedelta_mock):
    """Test that nans and values are properly converted to timedeltas."""
    # Setup
    column = pd.Series([7200, 3600, np.nan])
    to_timedelta_mock.return_value = pd.Series(
        [pd.Timedelta(hours=1), pd.Timedelta(hours=2), pd.Timedelta(hours=0)],
        dtype='timedelta64[ns]',
    )

    # Run
    converted_column = _convert_to_timedelta(column)

    # Assert
    to_timedelta_mock.assert_called_with(SeriesMatcher(pd.Series([7200, 3600, 0.0])))
    expected_column = pd.Series(
        [pd.Timedelta(hours=1), pd.Timedelta(hours=2), pd.NaT], dtype='timedelta64[ns]'
    )
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
    data = pd.Series(
        [pd.to_datetime('2020-01-01'), pd.to_datetime('2020-01-02'), pd.to_datetime('2020-01-03')],
    )

    # Run
    is_datetime = _is_datetime_type(data)

    # Assert
    assert is_datetime


def test__is_datetime_type_with_period():
    """Test the ``_is_datetime_type`` function when a period series is passed.

    Expect to return True when a period series is passed.

    Input:
    - A pandas.Series of type `period`
    Output:
    - True
    """
    # Setup
    data = pd.Series(pd.period_range('2023-01', periods=3, freq='M'))

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
        np.nan,
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
        np.nan,
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


def test__is_datetime_type_with_datetime_str_nanoseconds():
    """Test it for a datetime string with nanoseconds."""
    # Setup
    value = '2011-10-15 20:11:03.498707'

    # Run
    is_datetime = _is_datetime_type(value)

    # Assert
    assert is_datetime


def test__is_datetime_type_with_str_int():
    """Test it for a string with an integer."""
    # Setup
    value = '123'

    # Run
    is_datetime = _is_datetime_type(value)

    # Assert
    assert is_datetime is False


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
        'parent_table': pd.DataFrame({'id': [1, 2, 3]}),
        'child_table': pd.DataFrame({'id': [1, 2, 3], 'fk': [None, 2, np.nan]}),
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
        'parent_table': pd.DataFrame({'id': [1, 2, 3]}),
        'child_table': pd.DataFrame({'id': [1, 2, 3], 'fk': [1, 2, 3]}),
    }

    # Run
    _validate_foreign_keys_not_null(metadata, data)


@patch('sdv._utils.warnings')
def test_check_sdv_versions_and_warn_no_mismatch(mock_warnings):
    """Test that no warnings is raised when no mismatch is produced."""
    # Setup
    synthesizer = Mock()
    synthesizer._fitted_sdv_version = version.community
    synthesizer._fitted_sdv_enterprise_version = version.enterprise

    # Run
    check_sdv_versions_and_warn(synthesizer)

    # Assert
    mock_warnings.warn.assert_not_called()


def test_check_sdv_versions_and_warn_community_mismatch():
    """Test that warnings is raised when community version is mismatched."""
    # Setup
    synthesizer = Mock()
    synthesizer._fitted_sdv_version = '1.0.0'
    synthesizer._fitted_sdv_enterprise_version = version.enterprise

    # Run and Assert
    message = (
        f'You are currently on SDV version {version.community} but this synthesizer was created on '
        'version 1.0.0. The latest bug fixes and features may not be available for this '
        'synthesizer. To see these enhancements, create and train a new synthesizer on this '
        'version.'
    )
    with pytest.warns(SDVVersionWarning, match=message):
        check_sdv_versions_and_warn(synthesizer)


@patch('sdv._utils.version')
def test_check_sdv_versions_and_warn_enterprise_mismatch(mock_version):
    """Test that warnings is raised when enterprise version is mismatched."""
    # Setup
    synthesizer = Mock()
    synthesizer._fitted_sdv_version = version.community
    synthesizer._fitted_sdv_enterprise_version = '1.2.0'

    mock_version.enterprise = '1.3.0'
    mock_version.community = version.community

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
def test_check_sdv_versions_and_warn_community_and_enterprise_mismatch(mock_version):
    """Test that warnings is raised when both community and enterprise version mismatch."""
    # Setup
    synthesizer = Mock()
    synthesizer._fitted_sdv_version = '1.0.0'
    synthesizer._fitted_sdv_enterprise_version = '1.2.0'

    mock_version.community = '1.3.0'
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
def test_check_synthesizer_version_community_and_enterprise_are_lower(mock_version):
    """Test that VersionError is raised when both community and enterprise version are higher."""
    # Setup
    synthesizer = Mock(_fitted_sdv_version='2.0.0', _fitted_sdv_enterprise_version='2.1.0')

    mock_version.community = '1.3.0'
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
def test_check_synthesizer_version_community_is_lower(mock_version):
    """Test that VersionError is raised when only community version is lower."""
    # Setup
    synthesizer = Mock(_fitted_sdv_version='1.4.0', _fitted_sdv_enterprise_version='1.2.0')

    mock_version.community = '1.3.0'
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
    synthesizer = Mock(_fitted_sdv_version='1.3.0', _fitted_sdv_enterprise_version='1.3.0')

    mock_version.community = '1.3.0'
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
    synthesizer = Mock(_fitted_sdv_version='1.3.0', _fitted_sdv_enterprise_version=None)

    mock_version.community = '1.3.0'
    mock_version.enterprise = '1.2.0'

    # Run and Assert
    check_synthesizer_version(synthesizer)


def test__get_root_tables():
    """Test the ``_get_root_tables`` method."""
    # Setup
    relationships = [
        {'parent_table_name': 'parent', 'child_table_name': 'child'},
        {'parent_table_name': 'child', 'child_table_name': 'grandchild'},
        {'parent_table_name': 'parent', 'child_table_name': 'grandchild'},
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
    synthesizer = Mock(_fitted_sdv_version='1.3.0', _fitted_sdv_enterprise_version='1.3.0')

    mock_version.community = '1.3.0'
    mock_version.enterprise = '1.4.0'

    # Run and Assert
    message = (
        'You are currently on SDV Enterprise version 1.4.0 but this synthesizer was created on '
        'version 1.3.0. Fitting this synthesizer again is not supported. '
        'Please create a new synthesizer.'
    )
    with pytest.raises(VersionError, match=message):
        check_synthesizer_version(synthesizer, is_fit_method=True, compare_operator=operator.lt)


@patch('sdv._utils.version')
def test_check_synthesizer_version_check_synthesizer_is_greater_equal(mock_version):
    """Test that no ``VersionError`` is raised when versions match."""
    # Setup
    synthesizer = Mock(_fitted_sdv_version='1.3.0', _fitted_sdv_enterprise_version='1.3.0')

    mock_version.community = '1.3.0'
    mock_version.enterprise = '1.3.0'

    # Run and Assert
    check_synthesizer_version(synthesizer, is_fit_method=True, compare_operator=operator.lt)


@patch('sdv._utils.version')
def test_check_synthesizer_version_check_synthesizer_is_greater_community_mismatch(mock_version):
    """Test that ``VersionError`` is raised when checking if synthesizer is greater.

    Ensure that this test will raise a ``VersionError`` when the synthesizer version is lower
    than the current package version.
    """
    # Setup
    synthesizer = Mock(_fitted_sdv_version='1.3.0', _fitted_sdv_enterprise_version='1.4.0')

    mock_version.community = '1.5.0'
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
def test_check_synthesizer_version_check_synthesizer_is_greater_both_mismatch(mock_version):
    """Test that ``VersionError`` is raised when community and enterprise are greater.

    Ensure that this test will raise a ``VersionError`` when the synthesizer version is lower
    than the current package version.
    """
    # Setup
    synthesizer = Mock(_fitted_sdv_version='1.3.0', _fitted_sdv_enterprise_version='1.3.2')

    mock_version.community = '1.5.0'
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
    mock_version.community = '1.0.0'
    mock_uuid.uuid4.return_value = '92aff11e-9a56-49d1-a280-990d1231a5f5'
    metadata = SingleTableMetadata()
    metadata.add_column('key', sdtype='id')
    synthesizer = BaseSingleTableSynthesizer(metadata)

    # Run
    result = generate_synthesizer_id(synthesizer)

    # Assert
    assert result == 'BaseSingleTableSynthesizer_1.0.0_92aff11e9a5649d1a280990d1231a5f5'


@patch('sdv._utils._get_chars_for_option')
def test_get_possible_chars_excludes_at(mock_get_chars):
    """Test that 'at' regex operations aren't included when getting chars."""
    # Setup
    regex = '^[1-9]{1,2}$'
    mock_get_chars.return_value = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # Run
    possible_chars = get_possible_chars(regex)

    # Assert
    mock_get_chars.assert_called_once()
    mock_call = mock_get_chars.mock_calls[0]
    assert mock_call[1][0] == sre_parse.MAX_REPEAT
    assert mock_call[1][1][0] == 1
    assert mock_call[1][1][1] == 2
    assert mock_call[1][1][2].data == [(sre_parse.IN, [(sre_parse.RANGE, (49, 57))])]
    assert possible_chars == [str(i) for i in range(10)]


def test_get_possible_chars_unsupported_regex():
    """Test that an error is raised if the regex contains unsupported options."""
    # Setup
    regex = '(ab)*'

    # Run and assert
    message = 'REGEX operation: SUBPATTERN is not supported by SDV.'
    with pytest.raises(ValueError, match=message):
        get_possible_chars(regex)


@patch('sdv._utils._get_chars_for_option')
def test_get_possible_chars_handles_max_repeat(mock_get_chars):
    """Test that MAX_REPEATS are handled by recursively finding the first non MAX_REPEAT.

    One valid regex option is a MAX_REPEAT. Getting all possible values for this could be slow,
    so we just look for the first nexted option that isn't a max_repeat to get the possible
    characters instead.
    """
    # Setup
    regex = '[1-9]{1,2}'
    mock_get_chars.side_effect = lambda x, y: _get_chars_for_option(x, y)

    # Run
    possible_chars = get_possible_chars(regex)

    # Assert
    assert len(mock_get_chars.mock_calls) == 2
    assert mock_get_chars.mock_calls[1][1] == mock_get_chars.mock_calls[0][1][1][2][0]
    assert possible_chars == [str(i) for i in range(1, 10)]


def test_get_possible_chars_num_subpatterns():
    """Test that only characters for first x subpatterns are returned."""
    # Setup
    regex = 'HID_[0-9]{3}_[a-z]{3}'

    # Run
    possible_chars = get_possible_chars(regex, 3)

    # Assert
    assert possible_chars == ['H', 'I', 'D']


def test_get_possible_chars():
    """Test that all characters for regex are returned."""
    # Setup
    regex = 'HID_[0-9]{3}_[a-z]{3}'

    # Run
    possible_chars = get_possible_chars(regex)

    # Assert
    prefix = ['H', 'I', 'D', '_']
    nums = [str(i) for i in range(10)]
    lowercase_letters = list(string.ascii_lowercase)
    assert possible_chars == prefix + nums + ['_'] + lowercase_letters


def test__is_numerical():
    """Test that ensures that if passed any numerical data type we will get a ``True``."""
    # Setup
    np_int = np.int16(10)
    np_nan = np.nan

    # Run
    np_int_result = _is_numerical(np_int)
    np_nan_result = _is_numerical(np_nan)

    # Assert
    assert np_int_result
    assert np_nan_result


def test__is_numerical_string():
    """Test that ensures that if passed any other value but numerical it will return `False`."""
    # Setup
    str_value = 'None'
    datetime_value = pd.to_datetime('2012-01-01')

    # Run
    str_result = _is_numerical(str_value)
    datetime_result = _is_numerical(datetime_value)

    # Assert
    assert str_result is False
    assert datetime_result is False


def test__get_transformer_init_kwargs():
    # Setup
    transformer = FloatFormatter(
        missing_value_replacement=None,
        learn_rounding_scheme=True,
        computer_representation='Float64',
        enforce_min_max_values=False,
    )

    # Run
    transformer_kwarg_dict = _get_transformer_init_kwargs(transformer)

    # Assert
    transformer_kwarg_dict == {
        'missing_value_replacement': None,
        'learn_rounding_scheme': True,
        'computer_representation': 'Float64',
    }


@patch('sdv._utils.strings_from_regex')
def test__check_regex_format(mock_strings_from_regex):
    """Test the ``_check_regex_format``."""
    # Setup
    regex = '[a-z]{3}'
    mock_strings_from_regex.side_effect = KeyError('regex')
    expected_error = re.escape(
        'SDV synthesizers do not currently support complex regex formats such as '
        f"'{regex}', which you have provided for table 'table_name', column 'column_name'."
        ' Please use a simplified format or update to a different sdtype.'
    )

    # Run
    with pytest.raises(SynthesizerInputError, match=expected_error):
        _check_regex_format('table_name', 'column_name', regex)

    # Assert
    mock_strings_from_regex.assert_called_once_with(regex)


@pytest.mark.parametrize(
    'value, datetime_format, expected',
    [
        ('2023-06-01', '%Y-%m-%d', True),
        ('2023-6-1', '%Y-%m-%d', False),
        ('06/01/2023', '%Y-%m-%d', False),
        (pd.NA, '%Y-%m-%d', True),
        (None, '%Y-%m-%d', True),
        (datetime(2023, 6, 1), '%Y-%m-%d', False),
        (20230601, '%Y-%m-%d', False),
        ('2023-06', '%Y-%m-%d', False),
    ],
)
def test__datetime_string_matches_format(value, datetime_format, expected):
    """Test `_datetime_string_matches_format` with various input types and formats."""
    # Run
    result = _datetime_string_matches_format(value, datetime_format)

    # Assert
    assert result is expected


@pytest.fixture(scope='function')
def dates():
    return [
        '2025-06-01',
        '2025-01-01',
        '2025-12-31',
    ]


@pytest.fixture()
def datetimes():
    return [
        '2025-06-01 00:00:00',
        '2025-01-01 23:59:59',
        '2025-12-31 01:01:01',
    ]


def add_nan(column):
    """Helper method to NaN to Series."""
    max_index = column.index.max() + 1
    if is_datetime64_any_dtype(column.dtype):
        column.loc[max_index] = pd.NaT
    if is_object_dtype(column.dtype):
        column.loc[max_index] = np.nan
    elif is_string_dtype(column.dtype):
        column.loc[max_index] = pd.NA
    return column


def add_timezone(column):
    """Helper method to timezone to Series."""
    timezones = pd.Series(['-0100', '-1200', '+1400'])
    return column + timezones


@pytest.mark.parametrize(
    'dtype',
    [('object'), ('string')],
)
class TestValidateDatetimeFormat:
    def test__validate_datetime_format_valid_dates(self, dates, dtype):
        """Test _validate_datetime_format with dates (as str), valid format."""
        # Setup
        datetime_format = '%Y-%m-%d'
        column = pd.Series(dates, dtype=dtype)
        column = add_nan(column)

        # Run
        invalid_values = _validate_datetime_format(column, datetime_format)

        # Assert
        assert len(invalid_values) == 0

    def test__validate_datetime_format_invalid_dates(self, dates, dtype):
        """Test _validate_datetime_format with dates (as str), invalid format."""
        # Setup
        if not is_pandas_two_installed:
            pytest.skip('Datetimes are parsed with a consistent format with pandas >= 2.0.0')
        bad_format = '%Y/%m/%d'
        column = pd.Series(dates, dtype=dtype)
        column = add_nan(column)

        # Run
        invalid_values = _validate_datetime_format(column, bad_format)

        # Assert
        assert len(invalid_values) == 3
        assert sorted(invalid_values) == sorted(column.tolist()[:3])

    def test__validate_datetime_format_valid_datetimes(self, datetimes, dtype):
        """Test _validate_datetime_format with datetimes (as str), valid format."""
        # Setup
        datetime_format = '%Y-%m-%d %H:%M:%S'
        column = pd.Series(datetimes, dtype=dtype)
        column = add_nan(column)

        # Run
        invalid_values = _validate_datetime_format(column, datetime_format)

        # Assert
        assert len(invalid_values) == 0

    def test__validate_datetime_format_invalid_with_datetimes(self, datetimes, dtype):
        """Test _validate_datetime_format with datetimes (as str) and invalid format."""
        # Setup
        bad_format = '%Y-%m-%d %H-%M'
        column = pd.Series(datetimes, dtype=dtype)
        column = add_nan(column)

        # Run
        invalid_values = _validate_datetime_format(column, bad_format)

        # Assert
        assert len(invalid_values) == 3
        assert sorted(invalid_values) == sorted(column.tolist()[:3])

    def test__validate_datetime_format_valid_datetimes_tz(self, datetimes, dtype):
        """Test _validate_datetime_format with datetimes (as str), valid format, tz."""
        # Setup
        datetime_format = '%Y-%m-%d %H:%M:%S%z'
        column = pd.Series(datetimes, dtype=dtype)
        column = add_nan(column)
        column = add_timezone(column)

        # Run
        invalid_values = _validate_datetime_format(column, datetime_format)

        # Assert
        assert len(invalid_values) == 0


@pytest.fixture()
def datetimes_as_dts():
    return [
        datetime(2025, 1, 1, tzinfo=None),
        datetime(2025, 12, 31, tzinfo=None),
        datetime(2025, 1, 1, tzinfo=None),
    ]


@pytest.fixture()
def datetimes_as_ts():
    return [
        pd.Timestamp(2025, 1, 1, tzinfo=None),
        pd.Timestamp(2025, 12, 31, tzinfo=None),
        pd.Timestamp(2025, 1, 1, tzinfo=None),
    ]


@pytest.fixture()
def datetimes_as_dts_with_tz():
    return [
        datetime(2025, 1, 1, tzinfo=timezone.utc),
        datetime(2025, 12, 31, tzinfo=timezone(timedelta(seconds=14 * 60 * 60))),
        datetime(
            2025,
            1,
            1,
            tzinfo=timezone(timedelta(seconds=-12 * 60 * 60)),
        ),
    ]


@pytest.fixture()
def datetimes_as_ts_with_tz():
    return [
        pd.Timestamp(1513393355, tz='US/Pacific'),
        pd.Timestamp(1513393234, tz='US/Eastern'),
        pd.Timestamp(1513399934, tz='US/Mountain'),
    ]


class TestValidateDatetimeFormatObjects:
    def test__validate_datetime_format_dt_no_tz(self, datetimes_as_dts):
        """Test _validate_datetime_format with datetimes (as datetimes) and no timezones."""
        # Setup
        datetime_format = '%Y-%m-%d %H:%M:%S'
        column = pd.Series(datetimes_as_dts, dtype='object')
        column = add_nan(column)

        # Run
        invalid_values = _validate_datetime_format(column, datetime_format)

        # Assert
        assert len(invalid_values) == 0

    def test__validate_datetime_format_ts_no_tz(self, datetimes_as_ts):
        """Test _validate_datetime_format with datetimes (as pd.Timestamps) and no timezones."""
        # Setup
        datetime_format = '%Y-%m-%d %H:%M:%S'
        column = pd.Series(datetimes_as_ts, dtype='object')
        column = add_nan(column)

        # Run
        invalid_values = _validate_datetime_format(column, datetime_format)

        # Assert
        assert len(invalid_values) == 0

    def test__validate_datetime_format_dt_tz(self, datetimes_as_dts_with_tz):
        """Test _validate_datetime_format with datetimes (as datetimes) and timezones."""
        # Setup
        datetime_format = '%Y-%m-%d %H:%M:%S%z'
        column = pd.Series(datetimes_as_dts_with_tz, dtype='object')
        column = add_nan(column)

        # Run
        invalid_values = _validate_datetime_format(column, datetime_format)

        # Assert
        assert len(invalid_values) == 0

    def test__validate_datetime_format_ts_tz(self, datetimes_as_ts_with_tz):
        """Test _validate_datetime_format with datetimes (as datetimes) and timezones."""
        # Setup
        datetime_format = '%Y-%m-%d %H:%M:%S%z'
        column = pd.Series(datetimes_as_ts_with_tz, dtype='object')
        column = add_nan(column)

        # Run
        invalid_values = _validate_datetime_format(column, datetime_format)

        # Assert
        assert len(invalid_values) == 0


@pytest.mark.parametrize(
    'dtype',
    [('object'), ('string')],
)
class TestValidateDatetimeFormatSameTimezone:
    def test__validate_datetime_format_timezone_dts(self, dtype):
        """Test _validate_datetime_format with datetimes (as datetimes) and same timezones."""
        # Setup
        datetime_format = '%Y-%m-%d %H:%M:%S%z'
        tzinfo = timezone(timedelta(seconds=14 * 60 * 60))
        column = pd.Series(
            [
                datetime(2025, 1, 1, tzinfo=tzinfo),
                datetime(2025, 12, 31, tzinfo=tzinfo),
                datetime(2025, 12, 31, tzinfo=tzinfo),
                datetime(2025, 12, 31, tzinfo=tzinfo),
                datetime(
                    2025,
                    1,
                    1,
                    tzinfo=tzinfo,
                ),
            ],
            dtype=dtype,
        )
        column = add_nan(column)

        # Run
        invalid_values = _validate_datetime_format(column, datetime_format)

        # Assert
        assert len(invalid_values) == 0

    def test__validate_datetime_format_same_timezone(self, dtype):
        """Test _validate_datetime_format with datetimes (as pd.Timestamps) and same timezones."""
        # Setup
        datetime_format = '%Y-%m-%d %H:%M:%S%z'
        offset = '-1200'
        tz = 'UTC-12:00'
        column = pd.Series(
            [
                pd.Timestamp(f'2025-01-01 00:00:00{offset}', tz=tz),
                pd.Timestamp(f'2025-01-02 00:01:00{offset}', tz=tz),
                pd.Timestamp(f'2025-02-01 01:00:00{offset}', tz=tz),
                pd.Timestamp(f'2025-12-01 02:00:00{offset}', tz=tz),
                pd.Timestamp(f'2025-01-31 10:00:00{offset}', tz=tz),
            ],
            dtype=dtype,
        )
        column = add_nan(column)

        # Run
        invalid_values = _validate_datetime_format(column, datetime_format)

        # Assert
        assert len(invalid_values) == 0


def test__validate_datetime_format_same_timezone():
    # Setup
    datetime_format = '%Y-%m-%d %H:%M:%S%z'
    column = pd.Series(
        [
            '2025-01-01 00:00:00',
            '2025-01-02 00:01:00',
            '2025-02-01 01:00:00',
        ],
        dtype='datetime64[ns]',
    )
    column = column.dt.tz_localize('Europe/Warsaw')
    column = add_nan(column)

    # Run
    invalid_values = _validate_datetime_format(column, datetime_format)

    # Assert
    assert len(invalid_values) == 0
    assert isinstance(column.tolist()[0], pd.Timestamp)
