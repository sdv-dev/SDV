"""Test BasePattern Class."""

import logging
import re
from copy import deepcopy
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from sdv.cag.base import BasePattern
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


class TestBasePattern:
    def test___init__(self):
        """Test the ``__init__`` method."""
        # Setup
        instance = BasePattern()

        # Assert
        assert instance._fitted is False
        assert instance.metadata is None

    def test__convert_data_to_dictionary(self):
        """Test the ``_convert_data_to_dictionary`` method."""
        # Setup
        instance = BasePattern()
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

        copied_dict = BasePattern._convert_data_to_dictionary(
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
        instance = BasePattern()
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
        instance = BasePattern()
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
        instance = BasePattern()
        instance._validate_pattern_with_data = Mock()
        instance._validate_pattern_with_metadata = Mock()
        expected_msg = re.escape('Pattern must be fit before validating without metadata.')
        data_mock = Mock()
        metadata_mock = Mock()

        # Run
        with pytest.raises(NotFittedError, match=expected_msg):
            instance.validate()

        instance.validate(data_mock, metadata_mock)

        # Assert
        instance._validate_pattern_with_metadata.assert_called_once_with(metadata_mock)
        instance._validate_pattern_with_data.assert_called_once_with(data_mock, metadata_mock)

    def test_validate_after_fitting(self):
        """Test ``validate`` validates with metadata after being fitted."""
        # Setup
        instance = BasePattern()
        instance._fitted = True
        instance.metadata = Mock()
        instance._validate_pattern_with_data = Mock()
        instance._validate_pattern_with_metadata = Mock()

        # Run
        instance.validate()

        # Assert
        instance._validate_pattern_with_metadata.assert_called_once_with(instance.metadata)

    def test_validate_single_table(self):
        """Test ``validate`` handles single table data."""
        # Setup
        instance = BasePattern()
        instance._single_table = True
        instance._table_name = 'table1'
        instance._validate_pattern_with_data = Mock()
        instance._validate_pattern_with_metadata = Mock()
        data_mock = Mock(spec=pd.DataFrame)
        metadata_mock = Mock()

        # Run
        instance.validate(data_mock, metadata_mock)

        # Assert
        instance._validate_pattern_with_metadata.assert_called_once_with(metadata_mock)
        instance._validate_pattern_with_data.assert_called_once_with(
            {'table1': data_mock}, metadata_mock
        )

    def test_get_updated_metadata(self):
        """Test method calls private ``_get_updated_metadata`` method."""
        # Setup
        instance = BasePattern()
        metadata = Mock()
        instance._get_updated_metadata = Mock()
        instance.validate = Mock()

        # Run
        instance.get_updated_metadata(metadata)

        # Assert
        instance.validate.assert_called_once_with(metadata=metadata)
        instance._get_updated_metadata.assert_called_once()

    def test_fit(self, data):
        """Test ``fit`` method."""
        # Setup
        instance = BasePattern()
        instance._validate_pattern_with_metadata = Mock()
        instance._validate_pattern_with_data = Mock()
        instance._fit = Mock()
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
        instance._validate_pattern_with_metadata.assert_called_once_with(metadata)
        instance._validate_pattern_with_data.assert_called_once_with(data, metadata)
        instance._fit.assert_called_once_with(data, metadata)
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
        instance = BasePattern()
        instance.table_name = 'table1'
        instance._validate_pattern_with_metadata = Mock()
        instance._validate_pattern_with_data = Mock()
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
        instance._validate_pattern_with_metadata.assert_called_once_with(metadata)
        instance._validate_pattern_with_data.assert_called_once_with(
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
        instance = BasePattern()
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
        instance = BasePattern()
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
        """Test ``transform`` method errors before pattern has been fit."""
        # Setup
        instance = BasePattern()
        expected_msg = re.escape('Pattern must be fit using ``fit`` before transforming.')

        # Run and Assert
        with pytest.raises(NotFittedError, match=expected_msg):
            instance.transform(data)

    def test_reverse_transform(self, data):
        """Test ``reverse_transform`` method."""
        # Setup
        instance = BasePattern()
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
        instance = BasePattern()
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
        instance = BasePattern()
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
        instance = BasePattern()
        expected_msg = re.escape(
            'Pattern must be fit using ``fit`` before determining if data is valid.'
        )

        # Run and assert
        with pytest.raises(NotFittedError, match=expected_msg):
            instance.is_valid(data)

    def test_is_valid(self, data):
        """Test ``is_valid`` calls the ``_is_valid`` method and returns its result."""
        # Setup
        instance = BasePattern()
        instance._is_valid = Mock()
        instance._fitted = True

        # Run
        is_valid_result = instance.is_valid(data)

        # Assert
        instance._is_valid.assert_called_once_with(data)
        assert is_valid_result == instance._is_valid.return_value

    def test_is_valid_single_table(self, data):
        """Test ``is_valid`` with single table data."""
        # Setup
        data = data['table1']
        instance = BasePattern()
        instance._single_table = True
        instance._table_name = 'table1'
        instance._is_valid = Mock()
        instance._is_valid.return_value = {'table1': data.copy()}
        instance._fitted = True

        # Run
        is_valid_result = instance.is_valid(data)

        # Assert
        instance._is_valid.assert_called_once_with(DataFrameDictMatcher({'table1': data}))
        pd.testing.assert_frame_equal(is_valid_result, data)
