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
        err_msg = re.escape(
            "Data is not valid for the 'OneHotEncoding' constraint in table 'table':\n  "
            ' a  b  c\n1  0  0  0'
        )
        with pytest.raises(ConstraintNotMetError, match=err_msg):
            instance._validate_constraint_with_data(data, metadata)

        # Row with two 1s
        data = {'table': pd.DataFrame({'a': [1, 0, 0], 'b': [0, 1, 1], 'c': [1, 0, 0]})}
        err_msg = re.escape(
            "Data is not valid for the 'OneHotEncoding' constraint in table 'table':\n"
            '   a  b  c\n0  1  0  1'
        )
        with pytest.raises(ConstraintNotMetError, match=err_msg):
            instance._validate_constraint_with_data(data, metadata)

        # Invalid number
        data = {'table': pd.DataFrame({'a': [1, 0, 0], 'b': [0, 2, 0], 'c': [0, 0, 1]})}
        err_msg = re.escape(
            "Data is not valid for the 'OneHotEncoding' constraint in table 'table':\n"
            '   a  b  c\n1  0  2  0'
        )
        with pytest.raises(ConstraintNotMetError, match=err_msg):
            instance._validate_constraint_with_data(data, metadata)

        # Nans
        data = {'table': pd.DataFrame({'a': [1, 0, 0], 'b': [0, 1, np.nan], 'c': [0, None, 1]})}
        err_msg = re.escape(
            "Data is not valid for the 'OneHotEncoding' constraint in table 'table':\n"
            '   a    b    c\n1  0  1.0  NaN\n2  0  NaN  1.0'
        )
        with pytest.raises(ConstraintNotMetError, match=err_msg):
            instance._validate_constraint_with_data(data, metadata)

        # Valid
        data = {'table': pd.DataFrame({'a': [1, 0, 0], 'b': [0, 1, 0], 'c': [0, 0, 1]})}
        assert instance._validate_constraint_with_data(data, metadata) is None

    def test__fit(self):
        """Test it runs."""
        # Setup
        instance = OneHotEncoding(column_names=['a', 'b', 'c'])

        # Run
        instance._fit({}, Metadata())

    def test__transform_injects_near_zero_one_noise(self):
        """Test it pushes 0s toward [0, eps) and 1s toward (1-eps, 1]."""
        # Setup
        data = pd.DataFrame({
            'a': [1, 0, 0],
            'b': [0, 1, 0],
            'c': [0, 0, 1],
            'd': [10, 20, 30],
        })
        metadata = Metadata.load_from_dict({
            'columns': {
                'a': {'sdtype': 'numerical'},
                'b': {'sdtype': 'numerical'},
                'c': {'sdtype': 'numerical'},
                'd': {'sdtype': 'numerical'},
            }
        })
        instance = OneHotEncoding(column_names=['a', 'b', 'c'])
        instance.fit(data, metadata)

        # Run
        transformed = instance.transform(data)

        # Assert
        pd.testing.assert_series_equal(transformed['d'], data['d'])

        eps = np.finfo(np.float32).eps
        original = data[['a', 'b', 'c']].to_numpy()
        result = transformed[['a', 'b', 'c']].to_numpy()

        zeros_mask = original == 0
        ones_mask = original == 1

        assert np.all(result[zeros_mask] == eps)
        assert np.all(result[ones_mask] == 1 - eps)

    def test_reverse_transform(self):
        """Test it reverses the transformation correctly."""
        # Setup
        instance = OneHotEncoding(column_names=['a', 'b'], table_name='table')
        instance._original_data_columns = {'table': ['a', 'b', 'c']}
        instance._dtypes = {
            'table': {
                'a': np.dtype('float'),
                'b': np.dtype('float'),
                'c': np.dtype('int64'),
            }
        }

        # Run
        table_data = pd.DataFrame({'a': [0.1, 0.5, 0.8], 'b': [0.8, 0.1, 0.9], 'c': [1, 2, 3]})
        out = instance.reverse_transform({'table': table_data})

        # Assert
        expected_out = pd.DataFrame({'a': [0.0, 1.0, 0.0], 'b': [1.0, 0.0, 1.0], 'c': [1, 2, 3]})
        pd.testing.assert_frame_equal(expected_out, out['table'])

    def test_transform_then_reverse_transform_restores_one_hot(self):
        """Test reverse_transform restores original one-hot."""
        # Setup
        data = pd.DataFrame({
            'a': [1, 0, 0, 1],
            'b': [0, 1, 0, 0],
            'c': [0, 0, 1, 0],
        })
        metadata = Metadata.load_from_dict({
            'columns': {
                'a': {'sdtype': 'numerical'},
                'b': {'sdtype': 'numerical'},
                'c': {'sdtype': 'numerical'},
            }
        })
        instance = OneHotEncoding(column_names=['a', 'b', 'c'])
        instance.fit(data, metadata)

        # Run
        transformed = instance.transform(data)
        restored = instance.reverse_transform(transformed)

        # Assert
        pd.testing.assert_frame_equal(data, restored)

    def test__get_updated_metadata_single_table(self):
        """Test columns in column_names switch from categorical to numerical in single table."""
        # Setup
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'a': {'sdtype': 'categorical'},
                        'b': {'sdtype': 'categorical'},
                        'c': {'sdtype': 'numerical'},
                        'd': {'sdtype': 'id'},
                    }
                }
            }
        })
        instance = OneHotEncoding(column_names=['a', 'b'])

        # Run
        updated = instance._get_updated_metadata(metadata)

        # Assert
        assert updated.tables['table'].columns['a']['sdtype'] == 'numerical'
        assert updated.tables['table'].columns['b']['sdtype'] == 'numerical'
        assert updated.tables['table'].columns['c']['sdtype'] == 'numerical'
        assert updated.tables['table'].columns['d']['sdtype'] == 'id'

    def test__get_updated_metadata_multi_table(self):
        """Test columns in column_names switch from categorical to numerical in multi table."""
        # Setup
        metadata = Metadata.load_from_dict({
            'tables': {
                'table1': {
                    'columns': {
                        'a': {'sdtype': 'categorical'},
                        'b': {'sdtype': 'categorical'},
                        'c': {'sdtype': 'numerical'},
                    }
                },
                'table2': {
                    'columns': {
                        'x': {'sdtype': 'categorical'},
                    }
                },
            }
        })
        instance = OneHotEncoding(column_names=['a', 'b'], table_name='table1')

        # Run
        updated = instance._get_updated_metadata(metadata)

        # Assert
        assert updated.tables['table1'].columns['a']['sdtype'] == 'numerical'
        assert updated.tables['table1'].columns['b']['sdtype'] == 'numerical'
        assert updated.tables['table1'].columns['c']['sdtype'] == 'numerical'
        assert updated.tables['table2'].columns['x']['sdtype'] == 'categorical'

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

    def test___init___invalid_learning_strategy(self):
        """Test it raises if learning_strategy is invalid."""
        # Run and Assert
        err_msg = "`learning_strategy` must be either 'one_hot' or 'categorical'."
        with pytest.raises(ValueError, match=err_msg):
            OneHotEncoding(column_names=['a', 'b', 'c'], learning_strategy='bad')

    def test_get_updated_metadata_categorical_single(self):
        """Test updated metadata drops one-hot columns and adds a categorical column."""
        # Setup
        metadata = Metadata.load_from_dict({
            'columns': {
                'a': {'sdtype': 'numerical'},
                'b': {'sdtype': 'numerical'},
                'c': {'sdtype': 'numerical'},
            }
        })
        instance = OneHotEncoding(column_names=['a', 'b', 'c'], learning_strategy='categorical')

        # Run
        updated = instance.get_updated_metadata(metadata)

        # Assert
        cols = updated.get_column_names()
        assert cols == ['a#b#c']

        table_name = updated._get_single_table_name()
        assert updated.tables[table_name].columns[cols[0]]['sdtype'] == 'categorical'

        assert 'a' not in cols
        assert 'b' not in cols
        assert 'c' not in cols

    def test_get_updated_metadata_categorical_repeated_column_names(self):
        """Test updated metadata drops one-hot columns and adds a categorical column."""
        # Setup
        metadata = Metadata.load_from_dict({
            'columns': {
                'a': {'sdtype': 'numerical'},
                'b': {'sdtype': 'numerical'},
                'c': {'sdtype': 'numerical'},
                'a#b#c': {'sdtype': 'numerical'},
            }
        })
        instance = OneHotEncoding(column_names=['a', 'b', 'c'], learning_strategy='categorical')

        # Run
        updated = instance.get_updated_metadata(metadata)

        # Assert
        cols = updated.get_column_names()
        assert cols == ['a#b#c', 'a#b#c_']

        table_name = updated._get_single_table_name()
        assert updated.tables[table_name].columns[cols[0]]['sdtype'] == 'numerical'
        assert updated.tables[table_name].columns[cols[1]]['sdtype'] == 'categorical'

        assert 'a' not in cols
        assert 'b' not in cols
        assert 'c' not in cols

    def test_transform_and_reverse_categorical_single(self):
        """Test transform collapses to categorical and reverse restores one-hot (single table)."""
        # Setup
        data = pd.DataFrame({
            'a': [1, 0, 0, 1],
            'b': [0, 1, 0, 0],
            'c': [0, 0, 1, 0],
        })
        metadata = Metadata.load_from_dict({
            'columns': {
                'a': {'sdtype': 'numerical'},
                'b': {'sdtype': 'numerical'},
                'c': {'sdtype': 'numerical'},
            }
        })
        instance = OneHotEncoding(column_names=['a', 'b', 'c'], learning_strategy='categorical')

        # Run
        instance.fit(data, metadata)
        transformed = instance.transform(data)
        reversed_df = instance.reverse_transform(transformed)

        # Assert
        pd.testing.assert_frame_equal(transformed, pd.DataFrame({'a#b#c': ['a', 'b', 'c', 'a']}))
        pd.testing.assert_frame_equal(data, reversed_df)

    def test_transform_and_reverse_categorical_multi(self):
        """Test transform/reverse with categorical strategy (multi table)."""
        # Setup
        data = {
            'table1': pd.DataFrame({
                'a': [1, 0, 0, 1],
                'b': [0, 1, 0, 0],
                'c': [0, 0, 1, 0],
            }),
            'table2': pd.DataFrame({'id': range(4)}),
        }
        metadata = Metadata.load_from_dict({
            'tables': {
                'table1': {
                    'columns': {
                        'a': {'sdtype': 'numerical'},
                        'b': {'sdtype': 'numerical'},
                        'c': {'sdtype': 'numerical'},
                    }
                },
                'table2': {
                    'columns': {
                        'id': {'sdtype': 'id'},
                    }
                },
            }
        })
        instance = OneHotEncoding(
            column_names=['a', 'b', 'c'], table_name='table1', learning_strategy='categorical'
        )

        # Run
        instance.fit(data, metadata)
        transformed = instance.transform(data)
        reversed_data = instance.reverse_transform(transformed)

        # Assert
        pd.testing.assert_frame_equal(
            transformed['table1'], pd.DataFrame({'a#b#c': ['a', 'b', 'c', 'a']})
        )
        pd.testing.assert_frame_equal(data['table2'], transformed['table2'])

        # Assert
        pd.testing.assert_frame_equal(data['table1'], reversed_data['table1'])
        pd.testing.assert_frame_equal(data['table2'], reversed_data['table2'])

    def test___repr__(self):
        """Test the representation of the instance that was created for this class."""
        # Setup
        ohe = OneHotEncoding(
            column_names=['status_active', 'status_inactive', 'status_on_hold'],
            learning_strategy='categorical',
        )
        # Run
        result = repr(ohe)

        # Assert
        assert result == (
            "OneHotEncoding(column_names=['status_active', 'status_inactive', 'status_on_hold'], "
            "learning_strategy='categorical')"
        )
