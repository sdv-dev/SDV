from collections import defaultdict
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from sdv.sampling.hierarchical_sampler import BaseHierarchicalSampler
from tests.utils import DataFrameMatcher, get_multi_table_metadata


class TestBaseHierarchicalSampler():

    def test___init__(self):
        """Test the default initialization of the ``BaseHierarchicalSampler``."""
        # Run
        metadata = get_multi_table_metadata()
        instance = BaseHierarchicalSampler(metadata, table_synthesizers={}, table_sizes={})

        # Assert
        assert instance.metadata == metadata
        assert instance._table_synthesizers == {}
        assert instance._table_sizes == {}

    def test__recreate_child_synthesizer(self):
        """Test that ``_recreate_child_synthesizer`` raises a ``NotImplementedError``."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseHierarchicalSampler(metadata, table_synthesizers={}, table_sizes={})

        # Run and Assert
        with pytest.raises(NotImplementedError, match=''):
            instance._recreate_child_synthesizer('nescra', 'oseba', pd.Series([], dtype='Int64'))

    def test__add_foreign_key_columns(self):
        """Test that ``_add_foreign_key_columns`` raises a ``NotImplementedError``."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseHierarchicalSampler(metadata, table_synthesizers={}, table_sizes={})

        # Run and Assert
        with pytest.raises(NotImplementedError, match=''):
            instance._add_foreign_key_columns(
                child_table=pd.DataFrame(),
                parent_table=pd.DataFrame(),
                child_name='oseba',
                parent_name='nescra'
            )

    def test__sample_rows(self):
        """Test that ``_sample_rows`` samples ``num_rows`` from the synthesizer. """
        synthesizer = Mock()
        instance = Mock()

        # Run
        result = BaseHierarchicalSampler._sample_rows(instance, synthesizer, 10)

        # Assert
        assert result == synthesizer._sample_batch.return_value
        synthesizer._sample_batch.assert_called_once_with(
            10,
            keep_extra_columns=True
        )

    def test__get_num_rows_from_parent(self):
        """Test that the number of child rows is extracted from the parent row."""
        # Setup
        parent_row = pd.Series({
            '__sessions__user_id__num_rows': 10,
        })
        instance = Mock()
        instance._max_child_rows = {'__sessions__user_id__num_rows': 10}

        # Run
        result = BaseHierarchicalSampler._get_num_rows_from_parent(
            instance, parent_row, 'sessions', 'user_id')

        # Assert
        expected_result = 10.0
        assert result == expected_result

    def test__add_child_rows(self):
        """Test adding child rows when sampled data is empty."""
        # Setup
        instance = Mock()
        instance._get_num_rows_from_parent.return_value = 10
        child_synthesizer_mock = Mock()
        instance._recreate_child_synthesizer.return_value = child_synthesizer_mock

        metadata = Mock()
        sessions_meta = Mock()
        users_meta = Mock()
        users_meta.primary_key = 'user_id'
        metadata.tables = {
            'users': users_meta,
            'sessions': sessions_meta
        }
        metadata._get_foreign_keys.return_value = ['user_id']
        instance.metadata = metadata

        instance._sample_rows.return_value = pd.DataFrame({
            'session_id': ['a', 'b', 'c'],
            'os': ['linux', 'mac', 'win'],
            'country': ['us', 'us', 'es'],
        })
        parent_row = pd.DataFrame({
            'user_id': [1, 2, 3],
            'name': ['John', 'Doe', 'Johanna']
        })
        sampled_data = {}

        # Run
        BaseHierarchicalSampler._add_child_rows(
            instance, 'sessions', 'users', parent_row, sampled_data
        )

        # Assert
        expected_result = pd.DataFrame({
            'session_id': ['a', 'b', 'c'],
            'os': ['linux', 'mac', 'win'],
            'country': ['us', 'us', 'es'],
            'user_id': [1, 2, 3]
        })
        pd.testing.assert_frame_equal(sampled_data['sessions'], expected_result)

    def test__add_child_rows_with_sampled_data(self):
        """Test adding child rows when sampled data contains values.

        The new sampled data has to be concatenated to the current sampled data.
        """
        # Setup
        instance = Mock()
        instance._get_num_rows_from_parent.return_value = 10
        child_synthesizer_mock = Mock()
        instance._recreate_child_synthesizer.return_value = child_synthesizer_mock

        metadata = Mock()
        sessions_meta = Mock()
        users_meta = Mock()
        users_meta.primary_key.return_value = 'user_id'
        metadata.tables = {
            'users': users_meta,
            'sessions': sessions_meta
        }
        metadata._get_foreign_keys.return_value = ['user_id']
        instance.metadata = metadata
        instance._synthesizer_kwargs = {'a': 0.1, 'b': 0.5, 'loc': 0.25}

        instance._sample_rows.return_value = pd.DataFrame({
            'session_id': ['a', 'b', 'c'],
            'os': ['linux', 'mac', 'win'],
            'country': ['us', 'us', 'es'],
        })
        parent_row = pd.DataFrame({
            'user_id': [1, 2, 3],
            'name': ['John', 'Doe', 'Johanna']
        })
        sampled_data = {
            'sessions': pd.DataFrame({
                'user_id': [0, 1, 0],
                'session_id': ['d', 'e', 'f'],
                'os': ['linux', 'mac', 'win'],
                'country': ['us', 'us', 'es'],
            })
        }

        # Run
        BaseHierarchicalSampler._add_child_rows(
            instance, 'sessions', 'users', parent_row, sampled_data)

        # Assert
        expected_result = pd.DataFrame({
            'user_id': [0, 1, 0, 1, 2, 3],
            'session_id': ['d', 'e', 'f', 'a', 'b', 'c'],
            'os': ['linux', 'mac', 'win', 'linux', 'mac', 'win'],
            'country': ['us', 'us', 'es', 'us', 'us', 'es'],
        })
        pd.testing.assert_frame_equal(sampled_data['sessions'], expected_result)

    def test__sample_table(self):
        """Test sampling a table.

        The ``sample_table`` method will call sample children and return the sampled data
        dictionary.
        """
        # Setup
        def sample_children(table_name, sampled_data, table_rows):
            sampled_data['sessions'] = pd.DataFrame({
                'user_id': [1, 1, 3],
                'session_id': ['a', 'b', 'c'],
                'os': ['windows', 'linux', 'mac'],
                'country': ['us', 'us', 'es']
            })
            sampled_data['transactions'] = pd.DataFrame({
                'transaction_id': [1, 2, 3],
                'session_id': ['a', 'a', 'b']
            })

        table_synthesizer = Mock()

        instance = Mock()
        instance.metadata._get_child_map.return_value = {'users': ['sessions', 'transactions']}
        instance._table_sizes = {'users': 10}
        instance._table_synthesizers = {'users': Mock()}
        instance._sample_children.side_effect = sample_children
        instance._sample_rows.return_value = pd.DataFrame({
            'user_id': [1, 2, 3],
            'name': ['John', 'Doe', 'Johanna']
        })

        # Run
        result = {}
        BaseHierarchicalSampler._sample_table(instance, table_synthesizer, 'users', 3, result)

        # Assert
        expected_result = {
            'users': pd.DataFrame({
                'user_id': [1, 2, 3],
                'name': ['John', 'Doe', 'Johanna'],
            }),
            'sessions': pd.DataFrame({
                'user_id': [1, 1, 3],
                'session_id': ['a', 'b', 'c'],
                'os': ['windows', 'linux', 'mac'],
                'country': ['us', 'us', 'es'],
            }),
            'transactions': pd.DataFrame({
                'transaction_id': [1, 2, 3],
                'session_id': ['a', 'a', 'b']
            })
        }
        for result_frame, expected_frame in zip(result.values(), expected_result.values()):
            pd.testing.assert_frame_equal(result_frame, expected_frame)

    def test__finalize(self):
        """Test that finalize removes extra columns from the sampled data."""
        # Setup
        instance = Mock()
        metadata = Mock()
        metadata._get_parent_map.return_value = {
            'sessions': ['users'],
            'transactions': ['sessions']
        }
        instance.metadata = metadata

        sampled_data = {
            'users': pd.DataFrame({
                'user_id': pd.Series([0, 1, 2], dtype=np.int64),
                'name': pd.Series(['John', 'Doe', 'Johanna'], dtype=object),
                'additional_column': pd.Series([0.1, 0.2, 0.3], dtype=float),
                'another_additional_column': pd.Series([0.1, 0.2, 0.5], dtype=float),
            }),
            'sessions': pd.DataFrame({
                'user_id': pd.Series([1, 2, 1], dtype=np.int64),
                'session_id': pd.Series(['a', 'b', 'c'], dtype=object),
                'os': pd.Series(['linux', 'mac', 'win'], dtype=object),
                'country': pd.Series(['us', 'us', 'es'], dtype=object),
            }),
            'transactions': pd.DataFrame({
                'transaction_id': pd.Series([1, 2, 3], dtype=np.int64),
                'session_id': pd.Series(['a', 'a', 'b'], dtype=object),
            }),
        }

        users_synth = Mock()
        users_synth._data_processor._dtypes = {'user_id': np.int64, 'name': str}
        sessions_synth = Mock()
        sessions_synth._data_processor._dtypes = {
            'user_id': np.int64,
            'session_id': str,
            'os': str,
            'country': str
        }
        transactions_synth = Mock()
        transactions_synth._data_processor._dtypes = {
            'transaction_id': np.int64,
            'session_id': str
        }

        instance._table_synthesizers = {
            'users': users_synth,
            'sessions': sessions_synth,
            'transactions': transactions_synth
        }

        # Run
        result = BaseHierarchicalSampler._finalize(instance, sampled_data)

        # Assert
        expected_result = {
            'users': pd.DataFrame({
                'user_id': pd.Series([0, 1, 2], dtype=np.int64),
                'name': pd.Series(['John', 'Doe', 'Johanna'], dtype=object),
            }),
            'sessions': pd.DataFrame({
                'user_id': pd.Series([1, 2, 1], dtype=np.int64),
                'session_id': pd.Series(['a', 'b', 'c'], dtype=object),
                'os': pd.Series(['linux', 'mac', 'win'], dtype=object),
                'country': pd.Series(['us', 'us', 'es'], dtype=object),
            }),
            'transactions': pd.DataFrame({
                'transaction_id': pd.Series([1, 2, 3], dtype=np.int64),
                'session_id': pd.Series(['a', 'a', 'b'], dtype=object),
            }),
        }
        for result_frame, expected_frame in zip(result.values(), expected_result.values()):
            pd.testing.assert_frame_equal(result_frame, expected_frame)

    def test__sample(self):
        """Test that the ``_sample_table`` is called for root tables."""
        # Setup
        expected_sample = {
            'users': pd.DataFrame({
                'user_id': [1, 2, 3],
                'name': ['John', 'Doe', 'Johanna']
            }),
            'sessions': pd.DataFrame({
                'user_id': [1, 1, 3],
                'session_id': ['a', 'b', 'c'],
                'os': ['windows', 'linux', 'mac'],
                'country': ['us', 'us', 'es']
            }),
            'transactions': pd.DataFrame(dtype='Int64')
        }

        def _sample_table(synthesizer, table_name, num_rows, sampled_data):
            sampled_data['users'] = expected_sample['users']
            sampled_data['sessions'] = expected_sample['sessions']
            sampled_data['transactions'] = expected_sample['transactions']

        instance = Mock()
        instance._table_sizes = {
            'users': 3,
            'transactions': 9,
            'sessions': 5
        }
        instance.metadata.relationships = [
            {
                'parent_table_name': 'users',
                'parent_primary_key': 'id',
                'child_table_name': 'sessions',
                'child_foreign_key': 'id'
            }
        ]
        users_synthesizer = Mock()
        instance._table_synthesizers = defaultdict(Mock, {'users': users_synthesizer})
        instance.metadata._get_parent_map.return_value = {
            'sessions': ['users'],
            'transactions': ['sessions']
        }
        instance.metadata.tables = {
            'users': Mock(),
            'sessions': Mock(),
            'transactions': Mock(),
        }
        instance._sample_table.side_effect = _sample_table

        # Run
        result = BaseHierarchicalSampler._sample(instance)

        # Assert
        assert result == instance._finalize.return_value
        instance._sample_table.assert_called_once_with(synthesizer=users_synthesizer,
                                                       table_name='users',
                                                       num_rows=3,
                                                       sampled_data=expected_sample)
        instance._add_foreign_key_columns.assert_called_once_with(
            DataFrameMatcher(expected_sample['sessions']),
            DataFrameMatcher(expected_sample['users']),
            'sessions',
            'users')
        instance._finalize.assert_called_once_with(expected_sample)
