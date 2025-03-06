"""Unit tests for Fixed Increments CAG pattern."""

import re
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sdv.cag import FixedIncrements
from sdv.cag._errors import PatternNotMetError
from sdv.metadata import Metadata


class TestFixedIncremenets:
    def test__validate_init_inputs(self):
        # Run and Assert
        err_msg = '`column_name` must be a string.'
        with pytest.raises(ValueError, match=err_msg):
            FixedIncrements(column_name=1, increment_value=10)

        err_msg = '`increment_value` must be greater than 0.'
        with pytest.raises(ValueError, match=err_msg):
            FixedIncrements(column_name='a', increment_value=-1)

        err_msg = '`increment_value` must be a whole number.'
        with pytest.raises(ValueError, match=err_msg):
            FixedIncrements(column_name='a', increment_value=1.5)

        err_msg = '`table_name` must be a string or None.'
        with pytest.raises(ValueError, match=err_msg):
            FixedIncrements(column_name='a', increment_value=2, table_name=1)

    def test__init__(self):
        # Setup
        instance = FixedIncrements(column_name='a', increment_value=10)

        # Asserts
        assert instance.column_name == 'a'
        assert instance.increment_value == 10
        assert instance._dtype is None
        assert instance.table_name is None

    def test__init__table_name(self):
        # Setup
        instance = FixedIncrements(column_name='a', increment_value=10, table_name='table1')

        # Asserts
        assert instance.column_name == 'a'
        assert instance.increment_value == 10
        assert instance._dtype is None
        assert instance.table_name == 'table1'

    @patch('sdv.cag.fixed_increments._validate_table_name')
    @patch('sdv.cag.fixed_increments._validate_columns_in_metadata')
    def test__validate_pattern_with_metadata(
        self, _validate_columns_in_metadata_mock, _validate_table_name_mock
    ):
        # Setup
        instance = FixedIncrements(column_name='a', increment_value=5)
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'id': {'sdtype': 'id'},
                        'a': {'sdtype': 'numerical'},
                    },
                }
            }
        })

        # Run
        instance._validate_pattern_with_metadata(metadata)

        # Assert
        _validate_table_name_mock.assert_called_once()
        _validate_columns_in_metadata_mock.assert_called_once()

    def test__validate_pattern_with_metadata_incorrect_sdtype(self):
        # Setup
        instance = FixedIncrements(column_name='a', increment_value=2)
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'id': {'sdtype': 'id'},
                        'a': {'sdtype': 'datetime'},
                    },
                }
            }
        })
        err_msg = re.escape(
            "Column 'a' has an incompatible sdtype ('datetime')."
            "The column sdtype must be 'numerical'."
        )
        # Run and Assert
        with pytest.raises(PatternNotMetError, match=err_msg):
            instance._validate_pattern_with_metadata(metadata)

    def test__validate_pattern_with_metadata_invalid_column_name(self):
        # Setup
        column_name = 'column'
        instance = FixedIncrements(column_name=column_name, increment_value=2)
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'id': {'sdtype': 'id'},
                        'a': {'sdtype': 'datetime'},
                    },
                }
            }
        })
        err_msg = re.escape(f"Table 'table' is missing columns '{column_name}")
        # Run and Assert
        with pytest.raises(PatternNotMetError, match=err_msg):
            instance._validate_pattern_with_metadata(metadata)

    def test__validate_pattern_with_metadata_no_table_name(self):
        # Setup
        instance = FixedIncrements(column_name='a', increment_value=2, table_name=None)
        metadata = Metadata.load_from_dict({
            'tables': {
                'parent': {
                    'columns': {
                        'id': {'sdtype': 'id'},
                        'a': {'sdtype': 'numerical'},
                    },
                },
                'child': {
                    'columns': {
                        'id': {'sdtype': 'id'},
                    },
                },
            }
        })
        err_msg = re.escape('Metadata contains more than 1 table but no ``table_name`` provided.')
        # Run and Assert
        with pytest.raises(PatternNotMetError, match=err_msg):
            instance._validate_pattern_with_metadata(metadata)

    @pytest.mark.parametrize(
        'values',
        [
            [1, 3, 5, 7],
            [1, 3, 5, 7, 9, 11, 13],
        ],
    )
    def test__validate_pattern_with_data(self, values):
        # Setup
        increment_value = 2
        table_name = 'table1'
        column_name = 'odd'
        data = {table_name: pd.DataFrame({column_name: values})}
        metadata = Metadata.load_from_dict({
            'tables': {
                table_name: {
                    'columns': {
                        'id': {'sdtype': 'id'},
                        column_name: {'sdtype': 'numerical'},
                    },
                },
            }
        })
        instance = FixedIncrements(
            column_name=column_name, increment_value=increment_value, table_name=table_name
        )
        indices = data[table_name].index.tolist()
        if len(indices) > 5:
            indices = '[0, 1, 2, 3, 4, +2 more]'

        err_msg = re.escape(
            'The fixed increments requirement has been met because the data is not '
            f"evenly divisible by '{increment_value}' or contains NaNs "
            f'for row indices: {indices}'
        )
        with pytest.raises(PatternNotMetError, match=err_msg):
            instance._validate_pattern_with_data(data, metadata)

    def test__get_updated_metadata(self):
        # Setup
        instance = FixedIncrements(column_name='a', increment_value=10)
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'a': {'sdtype': 'numerical'},
                    },
                }
            }
        })

        # Run
        updated_metadata = instance._get_updated_metadata(metadata)

        # Assert
        expected_metadata_dict = Metadata.load_from_dict({
            'tables': {'table': {'columns': {'a#increment': {'sdtype': 'numerical'}}}},
        }).to_dict()
        assert updated_metadata.to_dict() == expected_metadata_dict
        assert list(metadata.tables['table'].columns.keys()) == ['a']

    @pytest.mark.parametrize(
        'dtype',
        [
            'int8',
            'int16',
            'int32',
            'int64',
            'Int8',
            'Int16',
            'Int32',
            'Int64',
            'float16',
            'float32',
            'float64',
            'Float32',
            'Float64',
        ],
    )
    def test__fit(self, dtype):
        # Setup
        increment_value = 2
        table_name = 'table1'
        column_name = 'even'
        data = {table_name: pd.DataFrame({column_name: pd.Series([2, 4, 6, 8, 10], dtype=dtype)})}
        metadata = Metadata.load_from_dict({
            'tables': {
                table_name: {
                    'columns': {
                        column_name: {'sdtype': 'numerical'},
                    },
                },
            }
        })
        instance = FixedIncrements(
            column_name=column_name, increment_value=increment_value, table_name=table_name
        )

        # Run
        instance._fit(data, metadata)

        # Assert
        assert instance._dtype == dtype

    def test__transform(self):
        # Setup
        increment_value = 2
        table_name = 'table1'
        column_name = 'even'
        data = {table_name: pd.DataFrame({column_name: pd.Series([2, 4, 6, 8, 10])})}
        instance = FixedIncrements(
            column_name=column_name, increment_value=increment_value, table_name=table_name
        )

        # Run
        transform_data = instance._transform(data)

        # Assert
        expected_transform = pd.DataFrame({'even#increment': pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])})
        pd.testing.assert_frame_equal(transform_data[table_name], expected_transform)

    def test__reverse_transform(self):
        # Setup
        increment_value = 10
        table_name = 'table1'
        original_column_name = 'a'
        transformed_column_name = f'{original_column_name}#increment'
        transformed_data = {
            table_name: pd.DataFrame({
                transformed_column_name: pd.Series([1, 10, 100, 1000, 10000])
            })
        }
        instance = FixedIncrements(
            column_name=original_column_name, increment_value=increment_value, table_name=table_name
        )
        instance._original_data_columns = {table_name: [original_column_name]}
        instance._dtype = pd.Series([1], dtype='int64').dtype
        instance._fixed_increments_column_name = transformed_column_name
        instance._dtypes = {table_name: {original_column_name: pd.Series([1], dtype='int64').dtype}}

        # Run
        data = instance.reverse_transform(transformed_data)

        # Assert
        expected_out = pd.DataFrame({original_column_name: [10, 100, 1000, 10000, 100000]})
        pd.testing.assert_frame_equal(data[table_name], expected_out)

    @pytest.mark.parametrize(
        'nan',
        [np.nan, pd.NA],
    )
    def test__is_valid(self, nan):
        # Setup
        table_name = 'table'
        column_name = 'a'
        increment_value = 1000
        table_data = {
            table_name: pd.DataFrame({
                column_name: [100, 20000, 55000, 75000, 11000, nan],
            })
        }
        instance = FixedIncrements(
            column_name=column_name, table_name=table_name, increment_value=increment_value
        )
        instance._fitted = True

        # Run
        out = instance.is_valid(table_data)

        # Assert
        expected_out = [False, True, True, True, True, True]
        np.testing.assert_array_equal(expected_out, out[table_name])
