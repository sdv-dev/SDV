import re
from unittest.mock import call, patch

import pandas as pd
import pytest

from sdv.errors import SynthesizerInputError, SynthesizerProcessingError
from sdv.metadata import Metadata
from sdv.single_table.dayz import (
    DayZSynthesizer,
    _validate_column_parameters,
    _validate_parameter_structure,
    _validate_parameters,
    _validate_table_parameters,
    _validate_tables_parameter,
)


@pytest.fixture
def metadata():
    return Metadata.load_from_dict({
        'tables': {
            'table': {
                'columns': {
                    'id': {'sdtype': 'id'},
                    'numerical': {'sdtype': 'numerical'},
                    'datetime': {'sdtype': 'datetime', 'datetime_format': '%d %b %Y'},
                    'categorical': {'sdtype': 'categorical'},
                    'pii': {'sdtype': 'ssn'},
                    'extra_column': {'sdtype': 'numerical'},
                }
            }
        }
    })


@pytest.fixture
def dayz_parameters():
    return {
        'DAYZ_SPEC_VERSION': 'V1',
        'tables': {
            'table': {
                'columns': {
                    'id': {},
                    'numerical': {},
                    'datetime': {},
                    'categorical': {},
                    'pii': {},
                }
            }
        },
    }


def test__validate_parameter_structure(dayz_parameters):
    """Test validating the structure of the parameters dict."""
    # Setup
    bad_parameters_type = 'not a dictionary'
    bad_parameters_key = {'invalid_key': None}
    bad_tables_parameters = {'tables': None}
    bad_tables_parameters_value = {'tables': {'table': None}}
    bad_tables_key = {'tables': {'table': {'invalid_key': None}}}

    # Run and Assert
    expected_bad_parameters_type_msg = re.escape(
        'DayZ parameters must be a dictionary of DayZSynthesizer parameters.'
    )
    with pytest.raises(SynthesizerProcessingError, match=expected_bad_parameters_type_msg):
        _validate_parameter_structure(bad_parameters_type)

    expected_bad_parameters_key_msg = re.escape(
        "DayZ parameters contains unexpected key(s): 'invalid_key'."
    )
    with pytest.raises(SynthesizerProcessingError, match=expected_bad_parameters_key_msg):
        _validate_parameter_structure(bad_parameters_key)

    expected_bad_tables_parameters_msg = re.escape(
        "The 'tables' value must be a dictionary of table parameters."
    )
    with pytest.raises(SynthesizerProcessingError, match=expected_bad_tables_parameters_msg):
        _validate_parameter_structure(bad_tables_parameters)
    with pytest.raises(SynthesizerProcessingError, match=expected_bad_tables_parameters_msg):
        _validate_parameter_structure(bad_tables_parameters_value)

    expected_bad_tables_key_msg = re.escape(
        "DayZ parameters contain unexpected key(s) 'invalid_key' for table 'table'."
    )
    with pytest.raises(SynthesizerProcessingError, match=expected_bad_tables_key_msg):
        _validate_parameter_structure(bad_tables_key)

    _validate_parameter_structure(dayz_parameters)


def test__validate_column_parameter():
    """Test basic column parameter validation."""
    # Setup
    column_metadata = {'sdtype': 'id'}
    bad_column_parameters = {'invalid_key': None}
    bad_missing_value = {'missing_values_proportion': 100}

    # Run and Assert
    expected_bad_column_msg = re.escape(
        "The column 'column' in table 'table' contains unexpected key(s) 'invalid_key'."
    )
    with pytest.raises(SynthesizerProcessingError, match=expected_bad_column_msg):
        _validate_column_parameters('table', 'column', column_metadata, bad_column_parameters)

    expected_bad_missing_value_msg = re.escape(
        "'missing_values_proportion' for column 'column' in table 'table' "
        'must be a float between 0.0 and 1.0.'
    )
    with pytest.raises(SynthesizerProcessingError, match=expected_bad_missing_value_msg):
        _validate_column_parameters('table', 'column', column_metadata, bad_missing_value)


def test__validate_column_parameters_numerical():
    """Test column validation for numerical columns."""
    # Setup
    column_metadata = {'sdtype': 'numerical'}
    bad_parameter_value = {'min_value': '0'}
    bad_parameter_combination = {'min_value': 0}
    bad_min_max_combination = {'min_value': 100, 'max_value': 0}

    # Run and Assert
    expected_bad_parameter_value_msg = re.escape(
        "'min_value' for column 'column' in table 'table' must be a float."
    )
    with pytest.raises(SynthesizerProcessingError, match=expected_bad_parameter_value_msg):
        _validate_column_parameters('table', 'column', column_metadata, bad_parameter_value)

    expected_bad_combination_msg = re.escape(
        "Invalid parameters for column 'column' in table 'table'. Both a "
        "'min_value' and a 'max_value' must be set."
    )
    with pytest.raises(SynthesizerProcessingError, match=expected_bad_combination_msg):
        _validate_column_parameters('table', 'column', column_metadata, bad_parameter_combination)

    expected_bad_min_max_msg = re.escape(
        "Invalid parameters for column 'column' in table 'table'. 'min_value' "
        "must be less than 'max_value'"
    )
    with pytest.raises(SynthesizerProcessingError, match=expected_bad_min_max_msg):
        _validate_column_parameters('table', 'column', column_metadata, bad_min_max_combination)


def test__validate_column_parameters_datetime():
    """Test column validation for datetime columns."""
    # Setup
    column_metadata = {'sdtype': 'datetime', 'datetime_format': '%d %b %Y'}
    bad_parameter_value = {'start_timestamp': pd.Timestamp('01-01-2000')}
    bad_datetime_value = {'start_timestamp': 'not a date'}
    missing_parameter = {'start_timestamp': '31 Dec 2020'}
    bad_start_end_combination = {'start_timestamp': '31 Dec 2020', 'end_timestamp': '01 Jan 2020'}

    # Run and Assert
    expected_bad_parameter_value_msg = re.escape(
        "'start_timestamp' for column 'column' in table 'table' must be a string."
    )
    with pytest.raises(SynthesizerProcessingError, match=expected_bad_parameter_value_msg):
        _validate_column_parameters('table', 'column', column_metadata, bad_parameter_value)

    expected_bad_datetime_value_msg = re.escape(
        "The 'start_timestamp' for column 'column' in table 'table' is not a valid datetime "
        'string or does not match the date time format (%d %b %Y).'
    )
    with pytest.raises(SynthesizerProcessingError, match=expected_bad_datetime_value_msg):
        _validate_column_parameters('table', 'column', column_metadata, bad_datetime_value)

    expected_bad_value_no_format_msg = re.escape(
        "The 'start_timestamp' for column 'column' in table 'table' is not a valid datetime string."
    )
    with pytest.raises(SynthesizerProcessingError, match=expected_bad_value_no_format_msg):
        _validate_column_parameters('table', 'column', {'sdtype': 'datetime'}, bad_datetime_value)

    expected_missing_parameter_msg = re.escape(
        "Invalid parameters for column 'column' in table 'table'. Both a "
        "'start_timestamp' and an 'end_timestamp' must be set."
    )
    with pytest.raises(SynthesizerProcessingError, match=expected_missing_parameter_msg):
        _validate_column_parameters('table', 'column', column_metadata, missing_parameter)

    expected_bad_start_end_msg = re.escape(
        "Invalid parameters for column 'column' in table 'table'. The 'start_timestamp' "
        "must be less than the 'end_timestamp'"
    )
    with pytest.raises(SynthesizerProcessingError, match=expected_bad_start_end_msg):
        _validate_column_parameters('table', 'column', column_metadata, bad_start_end_combination)


def test__validate_column_parameters_categorical():
    """Test column validation for categorical columns."""
    # Setup
    column_metadata = {'sdtype': 'categorical'}
    bad_category_values = {'category_values': 'not a list'}

    # Run and Assert
    expected_msg = re.escape(
        "'category_values' for column 'column' in table 'table' must be a list."
    )
    with pytest.raises(SynthesizerProcessingError, match=expected_msg):
        _validate_column_parameters('table', 'column', column_metadata, bad_category_values)


@patch('sdv.single_table.dayz._validate_column_parameters')
def test__validate_table_parameters(mock__validate_column_parameters, metadata, dayz_parameters):
    """Test validating table parameters."""
    # Setup
    table_metadata = metadata.tables['table']
    bad_table_columns = {'columns': {'bad_column': {}}}
    bad_num_rows = {'num_rows': -1}

    # Run and Assert
    expected_bad_column_msg = re.escape(
        "Invalid DayZ parameters provided, column(s) 'bad_column' are missing from table 'table'."
    )
    with pytest.raises(SynthesizerProcessingError, match=expected_bad_column_msg):
        _validate_table_parameters('table', table_metadata, bad_table_columns)

    expected_bad_num_rows_msg = re.escape(
        "Invalid DayZ parameter 'num_rows' for table 'table'. num_rows must "
        'be an integer greater than zero.'
    )
    with pytest.raises(SynthesizerProcessingError, match=expected_bad_num_rows_msg):
        _validate_table_parameters('table', table_metadata, bad_num_rows)

    _validate_table_parameters('table', table_metadata, dayz_parameters['tables']['table'])

    # Assert
    expected_calls = [
        call('table', col, table_metadata.columns[col], col_parameters)
        for col, col_parameters in dayz_parameters['tables']['table']['columns'].items()
    ]
    mock__validate_column_parameters.assert_has_calls(expected_calls)


@patch('sdv.single_table.dayz._validate_table_parameters')
def test__validate_tables_parameter(mock__validate_table_parameters, metadata, dayz_parameters):
    """Test validating DayZ parameters."""
    # Setup
    bad_table_parameters = {'tables': {'bad_table': {}}}

    # Run and Assert
    expected_msg = re.escape(
        "Invalid DayZ parameters provided, table(s) 'bad_table' are missing from the metadata."
    )
    with pytest.raises(SynthesizerProcessingError, match=expected_msg):
        _validate_tables_parameter(metadata, bad_table_parameters)

    _validate_tables_parameter(metadata, dayz_parameters)

    # Assert
    mock__validate_table_parameters.assert_called_once_with(
        'table', metadata.tables['table'], dayz_parameters['tables']['table']
    )


@patch('sdv.single_table.dayz._validate_tables_parameter')
@patch('sdv.single_table.dayz._validate_parameter_structure')
def test__validate_parameters(
    mock__validate_parameter_structure,
    mock__validate_tables_parameter,
    metadata,
    dayz_parameters,
):
    """Test the ``_validate_parameters`` function"""
    # Run
    _validate_parameters(metadata, dayz_parameters)

    # Assert
    mock__validate_parameter_structure.assert_called_once_with(dayz_parameters)
    mock__validate_tables_parameter.assert_called_once_with(metadata, dayz_parameters)


@patch('sdv.single_table.dayz._validate_parameter_structure')
def test_validate_parameters_errors_with_relationship(
    mock__validate_parameter_structure,
    metadata,
    dayz_parameters,
):
    """Test ``validate_parameters`` errors if relationships provided.."""
    # Setup
    dayz_parameters = {**dayz_parameters, 'relationships': [{}]}

    # Run and Assert
    expected_error_msg = re.escape(
        "Invalid DayZ parameter 'relationships' for single-table DayZSynthesizer. "
        'Please use multi-table DayZSynthesizer instead.'
    )
    with pytest.raises(SynthesizerProcessingError, match=expected_error_msg):
        _validate_parameters(metadata, dayz_parameters)

    # Assert
    mock__validate_parameter_structure.assert_called_once_with(dayz_parameters)


class TestDayZSynthesizer:
    def test__init__(self):
        """Test the `__init__` method."""
        # Setup
        metadata = Metadata()
        expected_error = re.escape(
            "Only the 'DayZSynthesizer.create_parameters' and the "
            'DayZSynthesizer.validate_parameters methods are an SDV public feature. To '
            'define and use a DayZSynthesizer object you must have SDV-Enterprise.'
        )

        # Run and Assert
        with pytest.raises(SynthesizerInputError, match=expected_error):
            DayZSynthesizer(metadata, locales=['es_ES'])

    @patch('sdv.single_table.dayz.create_parameters')
    def test_create_parameters(self, mock_create):
        """Test the `create_parameters` method."""
        # Setup
        data = pd.DataFrame()
        metadata = Metadata()
        mock_create.return_value = {
            'DAYZ_SPEC_VERSION': 'V1',
            'tables': {
                'guests': {
                    'num_rows': 658,
                    'columns': {
                        'guest_email': {'missing_values_proportion': 0.0},
                        'room_type': {
                            'category_values': ['BASIC', 'DELUXE', 'SUITE'],
                            'missing_values_proportion': 0.0,
                        },
                        'numerical_feature': {
                            'missing_values_proportion': 0.0,
                            'num_decimal_digits': 2,
                            'min_value': 0,
                            'max_value': 100,
                        },
                    },
                },
            },
        }

        # Run
        result = DayZSynthesizer.create_parameters(data, metadata, 'output_filename')

        # Assert
        mock_create.assert_called_once_with(data, metadata, 'output_filename')
        assert result == mock_create.return_value

    @patch('sdv.single_table.dayz._validate_parameters')
    def test_validate_parameters(
        self,
        mock__validate_parameters,
        metadata,
        dayz_parameters,
    ):
        """Test the ``validate_parameters`` method."""
        # Run
        DayZSynthesizer.validate_parameters(metadata, dayz_parameters)

        # Assert
        mock__validate_parameters.assert_called_once_with(metadata, dayz_parameters)
