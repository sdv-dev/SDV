"""Test BaseConstraint Class."""

import logging
import re
from copy import deepcopy
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from sdv.cag.base import BaseConstraint
from sdv.data_processing.datetime_formatter import DatetimeFormatter
from sdv.data_processing.numerical_formatter import NumericalFormatter
from sdv.errors import NotFittedError
from sdv.metadata import Metadata
from tests.utils import DataFrameDictMatcher


@pytest.fixture
def data():
    return {
        'table1': pd.DataFrame({
            'col1': range(5),
            'col2': ['A', 'A', 'A', 'B', 'B'],
            'col3': [0.0, 0.1, 0.2, 0.3, 0.4],
        }),
        'table2': pd.DataFrame({'col4': range(5, 10), 'col5': ['X', 'Y', 'Z', 'Z', 'X']}),
    }


class TestBaseConstraint:
    def test___init__(self):
        """Test the ``__init__`` method."""
        # Setup
        instance = BaseConstraint()

        # Assert
        assert instance._fitted is False
        assert instance.metadata is None
        assert instance._formatters == {}

    def test__convert_data_to_dictionary(self):
        """Test the ``_convert_data_to_dictionary`` method."""
        # Setup
        instance = BaseConstraint()
        instance.table_name = 'table'
        data = pd.DataFrame({'colA': range(5)})
        data.copy = Mock()
        data.copy.return_value = data

        data_dict = {'table': pd.DataFrame({'colA': range(5)})}
        data_dict['table'].copy = Mock()
        data_dict['table'].copy.return_value = data_dict['table']

        metadata = Metadata.load_from_dict({
            'tables': {'table': {'columns': {'colA': {'sdtype': 'numerical'}}}}
        })

        # Run
        single_table_converted = instance._convert_data_to_dictionary(data, metadata, copy=True)

        instance._single_table = True
        instance._table_name = 'table'
        fitted_single_table_converted = instance._convert_data_to_dictionary(data, metadata)

        copied_dict = BaseConstraint._convert_data_to_dictionary(
            instance, data_dict, metadata, copy=True
        )

        # Assert
        data.copy.assert_called_once()
        data_dict['table'].copy.assert_called_once()
        assert isinstance(single_table_converted, dict)
        assert set(single_table_converted.keys()) == {'table'}
        assert isinstance(fitted_single_table_converted, dict)
        assert set(fitted_single_table_converted.keys()) == {'table'}
        pd.testing.assert_frame_equal(single_table_converted['table'], data)
        pd.testing.assert_frame_equal(fitted_single_table_converted['table'], data)
        pd.testing.assert_frame_equal(copied_dict['table'], data_dict['table'])

    def test__get_single_table_name(self):
        """Test the ``_get_single_table_name`` helper method."""
        # Setup
        instance = BaseConstraint()
        instance.table_name = None
        single_table_metadata = Metadata.load_from_dict({
            'tables': {
                'table1': {'columns': {'col4': {}, 'col5': {}}},
            }
        })
        multi_table_metadata = Metadata.load_from_dict({
            'tables': {
                'table1': {
                    'columns': {'col1': {}, 'col2': {}, 'col3': {}},
                },
                'table2': {'columns': {'col4': {}, 'col5': {}}},
            }
        })

        # Run
        from_single_table_name = instance._get_single_table_name(single_table_metadata)
        instance.table_name = 'table2'
        from_multi_table_name = instance._get_single_table_name(multi_table_metadata)

        # Assert
        assert from_single_table_name == 'table1'
        assert from_multi_table_name == 'table2'

    def test__get_single_table_name_errors_if_no_table_name_attr(self):
        """Test ``_get_single_table_name`` errors if ``table_name`` does not exist."""
        # Setup
        instance = BaseConstraint()
        metadata = Metadata.load_from_dict({
            'tables': {
                'table1': {'columns': {'col4': {}, 'col5': {}}},
            }
        })

        # Run and assert
        error_msg = re.escape('No ``table_name`` attribute has been set.')
        with pytest.raises(ValueError, match=error_msg):
            instance._get_single_table_name(metadata)

    def test_validate(self):
        """Test ``validate`` validates data and metadata."""
        # Setup
        instance = BaseConstraint()
        instance._validate_constraint_with_data = Mock()
        instance._validate_constraint_with_metadata = Mock()
        expected_msg = re.escape('Constraint must be fit before validating without metadata.')
        data_mock = Mock()
        metadata_mock = Mock()

        # Run
        with pytest.raises(NotFittedError, match=expected_msg):
            instance.validate()

        instance.validate(data_mock, metadata_mock)

        # Assert
        instance._validate_constraint_with_metadata.assert_called_once_with(metadata_mock)
        instance._validate_constraint_with_data.assert_called_once_with(data_mock, metadata_mock)

    def test_validate_after_fitting(self):
        """Test ``validate`` validates with metadata after being fitted."""
        # Setup
        instance = BaseConstraint()
        instance._fitted = True
        instance.metadata = Mock()
        instance._validate_constraint_with_data = Mock()
        instance._validate_constraint_with_metadata = Mock()

        # Run
        instance.validate()

        # Assert
        instance._validate_constraint_with_metadata.assert_called_once_with(instance.metadata)

    def test_validate_single_table(self):
        """Test ``validate`` handles single table data."""
        # Setup
        instance = BaseConstraint()
        instance._single_table = True
        instance._table_name = 'table1'
        instance._validate_constraint_with_data = Mock()
        instance._validate_constraint_with_metadata = Mock()
        data_mock = Mock(spec=pd.DataFrame)
        metadata_mock = Mock()

        # Run
        instance.validate(data_mock, metadata_mock)

        # Assert
        instance._validate_constraint_with_metadata.assert_called_once_with(metadata_mock)
        instance._validate_constraint_with_data.assert_called_once_with(
            {'table1': data_mock}, metadata_mock
        )

    def test_get_updated_metadata(self):
        """Test method calls private ``_get_updated_metadata`` method."""
        # Setup
        instance = BaseConstraint()
        metadata = Mock()
        instance._get_updated_metadata = Mock()
        instance.validate = Mock()

        # Run
        instance.get_updated_metadata(metadata)

        # Assert
        instance.validate.assert_called_once_with(metadata=metadata)
        instance._get_updated_metadata.assert_called_once()

    @patch('sdv.cag.base._format_invalid_values_string')
    def test__format_error_message_constraint(self, mock_format_invalid_values_string):
        """Test `_format_error_message_constraint` method."""
        # Setup
        invalid_data = {'row_1': 'value_1', 'row_2': 'value_2'}
        constraint = BaseConstraint()
        table_name = 'test_table'
        mock_format_invalid_values_string.return_value = re.escape(
            'checkin_date checkout_date\n0  31 Dec 2020   29 Dec 2020'
        )
        expected_error_message = (
            "Data is not valid for the 'BaseConstraint' constraint in table "
            "'test_table':\ncheckin_date\\ checkout_date\\\n0\\ \\ 31\\ Dec\\ 2020\\"
            ' \\ \\ 29\\ Dec\\ 2020'
        )

        # Run
        message = constraint._format_error_message_constraint(invalid_data, table_name)

        # Assert
        mock_format_invalid_values_string.assert_called_once_with(invalid_data, 5)
        assert message == expected_error_message

    def test__fit_constraint_column_formatters(self):
        """Test the `_fit_constraint_column_formatters` fits formatters for dropped columns."""
        # Setup
        instance = Mock()
        instance._formatters = {}
        instance.metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'col1': {'sdtype': 'categorical'},
                        'col2': {'sdtype': 'numerical', 'computer_representation': 'Int8'},
                        'col3': {'sdtype': 'numerical'},
                        'date_col1': {'sdtype': 'datetime'},
                        'date_col2': {'sdtype': 'datetime'},
                    },
                },
                'table2': {
                    'columns': {
                        'col4': {'sdtype': 'datetime'},
                        'col5': {'sdtype': 'numerical'},
                    }
                },
            }
        })
        instance._original_data_columns = {
            'table': ['col1', 'col2', 'col3', 'date_col1', 'date_col2'],
            'table2': ['col4', 'col5'],
        }
        instance._get_updated_metadata = Mock(
            return_value=Metadata.load_from_dict({
                'tables': {'table': {'columns': {'col1#col2': {'sdtype': 'numerical'}}}}
            })
        )
        data = {
            'table': pd.DataFrame({
                'col1': ['abc', 'def'],
                'col2': [1, 2],
                'col3': [3, 4],
                'date_col1': ['16-05-2023', '14-04-2022'],
                'date_col2': pd.to_datetime(['2021-02-15', '2022-05-16']),
            }),
            'table2': pd.DataFrame({
                'col4': ['2023-01-01', '2023-02-01'],
                'col5': [5.0, 6.0],
            }),
        }

        # Run
        BaseConstraint._fit_constraint_column_formatters(instance, data)

        # Assert
        instance._get_updated_metadata.assert_called_once_with(instance.metadata)
        formatters = instance._formatters
        assert set(formatters.keys()) == {'table', 'table2'}
        assert set(formatters['table'].keys()) == {'col2', 'col3', 'date_col1', 'date_col2'}
        assert set(formatters['table2'].keys()) == {'col4', 'col5'}

        assert isinstance(formatters['table']['col2'], NumericalFormatter)
        assert formatters['table']['col2'].enforce_rounding is True
        assert formatters['table']['col2'].enforce_min_max_values is True
        assert formatters['table']['col2'].computer_representation == 'Int8'

        assert isinstance(formatters['table']['col3'], NumericalFormatter)
        assert formatters['table']['col3'].enforce_rounding is True
        assert formatters['table']['col3'].enforce_min_max_values is True
        assert formatters['table']['col3'].computer_representation == 'Float'

        assert isinstance(formatters['table']['date_col1'], DatetimeFormatter)
        assert isinstance(formatters['table']['date_col2'], DatetimeFormatter)
        assert formatters['table']['date_col1']._dtype == 'O'
        assert formatters['table']['date_col1'].datetime_format == '%d-%m-%Y'
        assert formatters['table']['date_col2']._dtype == '<M8[ns]'
        assert formatters['table']['date_col2'].datetime_format == '%Y-%m-%d'

        assert isinstance(formatters['table2']['col4'], DatetimeFormatter)
        assert formatters['table2']['col4']._dtype == 'O'
        assert formatters['table2']['col4'].datetime_format == '%Y-%m-%d'
        assert isinstance(formatters['table2']['col5'], NumericalFormatter)
        assert formatters['table2']['col5'].enforce_rounding is True
        assert formatters['table2']['col5'].enforce_min_max_values is True

    def test__format_constraint_columns(self):
        """Test formatting all columns that were dropped by constraints."""
        # Setup
        instance = Mock()
        instance._original_data_columns = {
            'table': ['categorical', 'int', 'float', 'datetime_col'],
        }

        formatters = {
            'table': {
                'int': Mock(),
                'float': Mock(),
                'datetime_col': Mock(),
            }
        }
        formatters['table']['int'].format_data.return_value = [0, 1, 2]
        formatters['table']['float'].format_data.return_value = [0.1, 1.2, 2.3]
        formatters['table']['datetime_col'].format_data.return_value = [
            '2021-02-15',
            '2022-05-16',
            '2023-07-18',
        ]
        instance._formatters = formatters
        data = {
            'table': pd.DataFrame({
                'categorical': ['A', 'A', 'C'],
                'int': [0.0, 1.0, 2.0],
                'float': [0.11, 1.21, 2.33],
                'datetime_col': pd.to_datetime(['2021-02-15', '2022-05-16', '2023-07-18']),
            })
        }

        # Run
        formatted_data = BaseConstraint._format_constraint_columns(instance, data)

        # Assert
        expected_data = pd.DataFrame({
            'categorical': ['A', 'A', 'C'],
            'int': [0, 1, 2],
            'float': [0.1, 1.2, 2.3],
            'datetime_col': ['2021-02-15', '2022-05-16', '2023-07-18'],
        })
        pd.testing.assert_frame_equal(expected_data, formatted_data['table'])

    def test__format_constraint_columns_backwards_compatibility(self):
        """Test the BaseConstraint without the `_formatters` attribute."""
        # Setup
        instance = BaseConstraint()
        del instance._formatters
        data = {
            'table': pd.DataFrame({
                'categorical': ['A', 'A', 'C'],
                'int': [0.0, 1.0, 2.0],
                'float': [0.11, 1.21, 2.33],
                'datetime_col': pd.to_datetime(['2021-02-15', '2022-05-16', '2023-07-18']),
            })
        }

        # Run
        formatted_data = BaseConstraint._format_constraint_columns(instance, data)

        # Assert
        pd.testing.assert_frame_equal(formatted_data['table'], data['table'])

    def test_fit(self, data):
        """Test ``fit`` method."""
        # Setup
        instance = BaseConstraint()
        instance._validate_constraint_with_metadata = Mock()
        instance._validate_constraint_with_data = Mock()
        instance._fit = Mock()
        instance._fit_constraint_column_formatters = Mock()
        metadata = Metadata.load_from_dict({
            'tables': {
                'table1': {
                    'columns': {'col1': {}, 'col2': {}, 'col3': {}},
                },
                'table2': {'columns': {'col4': {}, 'col5': {}}},
            }
        })

        # Run
        instance.fit(data, metadata)

        # Assert
        assert instance.metadata == metadata
        instance._validate_constraint_with_metadata.assert_called_once_with(metadata)
        instance._validate_constraint_with_data.assert_called_once_with(data, metadata)
        instance._fit.assert_called_once_with(data, metadata)
        instance._fit_constraint_column_formatters.assert_called_once_with(data)
        assert instance._dtypes == {
            'table1': {'col1': 'int64', 'col2': 'object', 'col3': 'float64'},
            'table2': {'col4': 'int64', 'col5': 'object'},
        }
        assert instance._original_data_columns == {
            'table1': ['col1', 'col2', 'col3'],
            'table2': ['col4', 'col5'],
        }

    def test_fit_single_table(self, data):
        """Test ``fit`` method with a single table."""
        # Setup
        instance = BaseConstraint()
        instance.table_name = 'table1'
        instance._validate_constraint_with_metadata = Mock()
        instance._validate_constraint_with_data = Mock()
        instance._fit = Mock()
        metadata = Metadata.load_from_dict({
            'tables': {
                'table1': {
                    'columns': {'col1': {}, 'col2': {}, 'col3': {}},
                }
            }
        })

        # Run
        instance.fit(data['table1'], metadata)

        # Assert
        assert instance._single_table is True
        assert instance._table_name == 'table1'
        assert instance.metadata == metadata
        instance._validate_constraint_with_metadata.assert_called_once_with(metadata)
        instance._validate_constraint_with_data.assert_called_once_with(
            DataFrameDictMatcher({'table1': data['table1']}), metadata
        )
        instance._fit.assert_called_once_with(
            DataFrameDictMatcher({'table1': data['table1']}), metadata
        )
        assert instance._dtypes == {
            'table1': {'col1': 'int64', 'col2': 'object', 'col3': 'float64'},
        }
        assert instance._original_data_columns == {
            'table1': ['col1', 'col2', 'col3'],
        }

    def test_transform(self, data):
        """Test ``transform`` method."""
        # Setup
        instance = BaseConstraint()
        instance._fitted = True
        instance.validate = Mock()
        instance._transform = Mock()
        instance.metadata = Mock()

        # Run
        instance.transform(data)

        # Assert
        instance.validate.assert_called_once_with(DataFrameDictMatcher(data))
        instance._transform.assert_called_once_with(DataFrameDictMatcher(data))

    def test_transform_single_table(self, data):
        """Test ``transform`` method with a single table."""
        # Setup
        data = data['table1']
        instance = BaseConstraint()
        instance._fitted = True
        instance._single_table = True
        instance._table_name = 'table1'
        instance.validate = Mock()
        instance._transform = Mock()
        instance._transform.return_value = {'table1': data}
        instance.metadata = Mock()

        # Run
        transformed = instance.transform(data)

        # Assert
        instance.validate.assert_called_once_with(data)
        instance._transform.assert_called_once_with(DataFrameDictMatcher({'table1': data}))
        pd.testing.assert_frame_equal(transformed, data)

    def test_transform_not_fitted(self, data):
        """Test ``transform`` method errors before constraint has been fit."""
        # Setup
        instance = BaseConstraint()
        expected_msg = re.escape('Constraint must be fit using ``fit`` before transforming.')

        # Run and Assert
        with pytest.raises(NotFittedError, match=expected_msg):
            instance.transform(data)

    def test_reverse_transform(self, data):
        """Test ``reverse_transform`` method."""
        # Setup
        instance = BaseConstraint()
        instance._dtypes = {
            'table1': {'col1': 'float64', 'col2': 'object', 'col3': 'float64'},
            'table2': {'col4': 'object', 'col5': 'object'},
        }
        instance._original_data_columns = {
            'table1': ['col3', 'col2', 'col1'],
            'table2': ['col4', 'col5'],
        }
        instance._reverse_transform = Mock()
        instance._reverse_transform.return_value = deepcopy(data)

        # Run
        reversed_data = instance.reverse_transform(data)

        # Assert
        instance._reverse_transform.assert_called_once_with(DataFrameDictMatcher(data))
        assert set(reversed_data.keys()) == {'table1', 'table2'}
        expected_table1 = pd.DataFrame({
            'col3': [0.0, 0.1, 0.2, 0.3, 0.4],
            'col2': ['A', 'A', 'A', 'B', 'B'],
            'col1': [0.0, 1.0, 2.0, 3.0, 4.0],
        })
        expected_table2 = pd.DataFrame(
            {'col4': [5, 6, 7, 8, 9], 'col5': ['X', 'Y', 'Z', 'Z', 'X']}, dtype='object'
        )
        pd.testing.assert_frame_equal(reversed_data['table1'], expected_table1)
        pd.testing.assert_frame_equal(reversed_data['table2'], expected_table2)

    def test_reverse_transform_single_table(self, data):
        """Test ``reverse_transform`` method with single table data."""
        # Setup
        data = data['table1']
        instance = BaseConstraint()
        instance._single_table = True
        instance._table_name = 'table1'
        instance._dtypes = {
            'table1': {'col1': 'float64', 'col2': 'object', 'col3': 'float64'},
        }
        instance._original_data_columns = {
            'table1': ['col3', 'col2', 'col1'],
        }
        instance._reverse_transform = Mock()
        instance._reverse_transform.return_value = {'table1': data.copy()}

        # Run
        reversed_data = instance.reverse_transform(data)

        # Assert
        instance._reverse_transform.assert_called_once_with(DataFrameDictMatcher({'table1': data}))
        expected_table1 = pd.DataFrame({
            'col3': [0.0, 0.1, 0.2, 0.3, 0.4],
            'col2': ['A', 'A', 'A', 'B', 'B'],
            'col1': [0.0, 1.0, 2.0, 3.0, 4.0],
        })
        pd.testing.assert_frame_equal(reversed_data, expected_table1)

    def test_reverse_transform_cast_fallback(self, data, caplog):
        """Test ``reverse_transform`` method."""
        # Setup
        instance = BaseConstraint()
        instance._dtypes = {
            'table1': {'col1': 'float64', 'col2': 'object', 'col3': 'float64'},
            'table2': {'col4': 'int64', 'col5': 'object'},
        }
        instance._original_data_columns = {
            'table1': ['col3', 'col2', 'col1'],
            'table2': ['col4', 'col5'],
        }
        instance._reverse_transform = Mock()
        instance._reverse_transform.return_value = deepcopy(data)
        instance._reverse_transform.return_value['table2']['col4'] = pd.Series(
            [0.5, 0.6, 0.7, 0.8, np.nan], dtype='float64'
        )
        msg = "Column 'col4' is being converted to float because it contains NaNs."

        # Run
        with caplog.at_level(logging.INFO):
            reversed_data = instance.reverse_transform(data)

        # Assert
        assert any(msg in record.message for record in caplog.records)
        instance._reverse_transform.assert_called_once_with(DataFrameDictMatcher(data))
        assert set(reversed_data.keys()) == {'table1', 'table2'}
        expected_table1 = pd.DataFrame({
            'col3': [0.0, 0.1, 0.2, 0.3, 0.4],
            'col2': ['A', 'A', 'A', 'B', 'B'],
            'col1': [0.0, 1.0, 2.0, 3.0, 4.0],
        })
        expected_table2 = pd.DataFrame({
            'col4': [0.5, 0.6, 0.7, 0.8, np.nan],
            'col5': ['X', 'Y', 'Z', 'Z', 'X'],
        })
        pd.testing.assert_frame_equal(reversed_data['table1'], expected_table1)
        pd.testing.assert_frame_equal(reversed_data['table2'], expected_table2)

    def test_is_valid_errors_if_not_fitted(self, data):
        """Test the ``is_valid`` method errors if the CAG has not been fit."""
        # Setup
        instance = BaseConstraint()
        expected_msg = re.escape(
            'Constraint must be fit using ``fit`` before determining '
            'if data is valid without providing metadata.'
        )

        # Run and assert
        with pytest.raises(NotFittedError, match=expected_msg):
            instance.is_valid(data)

    def test_is_valid(self, data):
        """Test ``is_valid`` calls the ``_is_valid`` method and returns its result."""
        # Setup
        instance = BaseConstraint()
        instance._is_valid = Mock()
        instance._fitted = True
        instance.metadata = Mock()

        # Run
        is_valid_result = instance.is_valid(data)

        # Assert
        instance._is_valid.assert_called_once_with(data, instance.metadata)
        assert is_valid_result == instance._is_valid.return_value

    def test_is_valid_single_table(self, data):
        """Test ``is_valid`` with single table data."""
        # Setup
        data = data['table1']
        instance = BaseConstraint()
        instance._single_table = True
        instance._table_name = 'table1'
        instance._is_valid = Mock()
        instance._is_valid.return_value = {'table1': data.copy()}
        instance._fitted = True
        instance.metadata = Mock()

        # Run
        is_valid_result = instance.is_valid(data)

        # Assert
        instance._is_valid.assert_called_once_with(
            DataFrameDictMatcher({'table1': data}), instance.metadata
        )
        pd.testing.assert_frame_equal(is_valid_result, data)

    def test___repr___no_parameters(self):
        """Test that the ``__str__`` method returns the class name.

        The ``__repr__`` method should return the class name followed by paranthesis.
        """
        # Setup
        instance = BaseConstraint()

        # Run
        text = repr(instance)

        # Assert
        assert text == 'BaseConstraint()'

    def test___repr___with_parameters(self):
        """Test that the ``__repr__`` method returns the class name and parameters.

        The ``_repr__`` method should return the class name followed by all non-default
        parameters wrapped in parentheses.

        Setup:
            - Create a dummy class which inherits from the ``BaseConstraint`` where:
                - The class has one required parameter in it's ``__init__`` method.
                - The class has three optional parameters in it's ``__init__`` method.
                - The class instance only sets 2 optional parameters.
        """

        # Setup
        class Dummy(BaseConstraint):
            def __init__(self, param0, param1=None, param2=None, param3=None):
                self.param0 = param0
                self.param1 = param1
                self.param2 = param2
                self.param3 = param3

        instance = Dummy(param0='required', param2='value', param3=True)

        # Run
        text = repr(instance)

        # Assert
        assert text == "Dummy(param0='required', param2='value', param3=True)"

    def test__repr__with_bool_parameters(self):
        """Test that the ``__repr__`` method returns the class name and parameters.

        The ``_repr__`` method should return the class name followed by all non-default
        parameters wrapped in parentheses and not include default boolean parameters.
        """

        # Setup
        class Dummy(BaseConstraint):
            def __init__(self, param0, param1=False):
                self.param0 = param0
                self.param1 = param1

        instance = Dummy(param0='required')

        # Run
        text = repr(instance)

        # Assert
        assert text == "Dummy(param0='required')"
