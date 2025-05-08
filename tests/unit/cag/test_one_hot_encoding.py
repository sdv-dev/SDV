"""Unit tests for OneHotEncoding constraint."""

import re
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sdv.cag import OneHotEncoding
from sdv.cag._errors import ConstraintNotMetError
from sdv.metadata import Metadata


class TestOneHotEncoding:
    def test___init___incorrect_column_name(self):
        """Test it raises an error if column_name is not a string."""
        # Run and Assert
        err_msg = '`column_names` must be a list of strings.'
        with pytest.raises(ValueError, match=err_msg):
            OneHotEncoding(column_names=1)

        with pytest.raises(ValueError, match=err_msg):
            OneHotEncoding(column_names=['a', 1])

    def test___init___incorrect_table_name(self):
        """Test it raises an error if table_name is not a string."""
        # Run and Assert
        err_msg = '`table_name` must be a string or None.'
        with pytest.raises(ValueError, match=err_msg):
            OneHotEncoding(column_names=['a', 'b', 'c'], table_name=1)

    def test___init___(self):
        """Test it initializes correctly."""
        # Run
        instance = OneHotEncoding(column_names=['a', 'b', 'c'])

        # Asserts
        assert instance._column_names == ['a', 'b', 'c']
        assert instance.table_name is None

    @patch('sdv.cag.one_hot_encoding._validate_table_and_column_names')
    def test__validate_constraint_with_metadata(self, validate_table_and_col_names_mock):
        """Test validating the constraint with metadata."""
        # Setup
        instance = OneHotEncoding(column_names=['low', 'middle', 'high'])
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'low': {'sdtype': 'numerical'},
                        'middle': {'sdtype': 'numerical'},
                        'high': {'sdtype': 'numerical'},
                    },
                }
            }
        })

        # Run
        instance._validate_constraint_with_metadata(metadata)

        # Assert
        validate_table_and_col_names_mock.assert_called_once()

    def test__validate_constraint_with_data(self):
        """Test it when the data is not valid."""
        # Setup
        instance = OneHotEncoding(column_names=['a', 'b', 'c'])
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'a': {'sdtype': 'numerical'},
                        'b': {'sdtype': 'numerical'},
                        'c': {'sdtype': 'numerical'},
                    },
                }
            }
        })

        # Row of all zeros
        data = {'table': pd.DataFrame({'a': [1, 0, 0], 'b': [0, 0, 0], 'c': [0, 0, 1]})}
        err_msg = re.escape('The one hot encoding requirement is not met for row indices: [1]')
        with pytest.raises(ConstraintNotMetError, match=err_msg):
            instance._validate_constraint_with_data(data, metadata)

        # Row with two 1s
        data = {'table': pd.DataFrame({'a': [1, 0, 0], 'b': [0, 1, 1], 'c': [1, 0, 0]})}
        err_msg = re.escape('The one hot encoding requirement is not met for row indices: [0]')
        with pytest.raises(ConstraintNotMetError, match=err_msg):
            instance._validate_constraint_with_data(data, metadata)

        # Invalid number
        data = {'table': pd.DataFrame({'a': [1, 0, 0], 'b': [0, 2, 0], 'c': [0, 0, 1]})}
        err_msg = re.escape('The one hot encoding requirement is not met for row indices: [1]')
        with pytest.raises(ConstraintNotMetError, match=err_msg):
            instance._validate_constraint_with_data(data, metadata)

        # Nans
        data = {'table': pd.DataFrame({'a': [1, 0, 0], 'b': [0, 1, np.nan], 'c': [0, None, 1]})}
        err_msg = re.escape('The one hot encoding requirement is not met for row indices: [1, 2]')
        with pytest.raises(ConstraintNotMetError, match=err_msg):
            instance._validate_constraint_with_data(data, metadata)

    def test__fit(self):
        """Test it runs."""
        # Setup
        instance = OneHotEncoding(column_names=['a', 'b', 'c'])

        # Run
        instance._fit({}, Metadata())

    def test__transform(self):
        """Test it returns the data unchanged."""
        # Setup
        table_data = {
            'table': pd.DataFrame({
                'a': [1, 2, 3],
                'b': [4, 5, 6],
                'c': [7, 8, 9],
            })
        }
        instance = OneHotEncoding(column_names=['a', 'b', 'c'])

        # Run
        out = instance._transform(table_data)

        # Assert
        pd.testing.assert_frame_equal(out['table'], table_data['table'])

    def test_reverse_transform(self):
        """Test it reverses the transformation correctly."""
        # Setup
        instance = OneHotEncoding(column_names=['a', 'b'], table_name='table')
        instance._original_data_columns = {'table': ['a', 'b', 'c']}
        instance._dtypes = {'table': {'a': np.dtype('float'), 'b': np.dtype('float')}}

        # Run
        table_data = pd.DataFrame({'a': [0.1, 0.5, 0.8], 'b': [0.8, 0.1, 0.9], 'c': [1, 2, 3]})
        out = instance.reverse_transform({'table': table_data})

        # Assert
        expected_out = pd.DataFrame({'a': [0.0, 1.0, 0.0], 'b': [1.0, 0.0, 1.0], 'c': [1, 2, 3]})
        pd.testing.assert_frame_equal(expected_out, out['table'])

    def test_is_valid(self):
        """Test it checks if the data is valid."""
        # Setup
        instance = OneHotEncoding(column_names=['a', 'b', 'c'], table_name='table')
        instance._fitted = True

        # Run
        table_data = pd.DataFrame({
            'a': [1.0, 1.0, 0.0, 0.5, 1.0],
            'b': [0.0, 1.0, 0.0, 0.5, 0.0],
            'c': [0.0, 2.0, 0.0, 0.0, np.nan],
            'd': [1, 2, 3, 4, 5],
        })
        data = {'table': table_data, 'table2': table_data}
        out = instance.is_valid(data)

        # Assert
        expected_out = {
            'table': pd.Series([True, False, False, False, False]),
            'table2': pd.Series([True, True, True, True, True]),
        }
        for table, series in out.items():
            pd.testing.assert_series_equal(expected_out[table], series)
