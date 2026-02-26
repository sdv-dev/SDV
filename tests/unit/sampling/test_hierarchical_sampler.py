from collections import defaultdict
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from sdv.sampling.hierarchical_sampler import BaseHierarchicalSampler
from tests.utils import DataFrameMatcher, SeriesMatcher, get_multi_table_metadata


class TestBaseHierarchicalSampler:
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
                parent_name='nescra',
            )

    def test__sample_rows(self):
        """Test that ``_sample_rows`` samples ``num_rows`` from the synthesizer."""
        synthesizer = Mock()
        instance = Mock()

        # Run
        result = BaseHierarchicalSampler._sample_rows(instance, synthesizer, 10)

        # Assert
        assert result == synthesizer._sample_batch.return_value
        synthesizer._sample_batch.assert_called_once_with(10, keep_extra_columns=True)

    def test__sample_rows_missing_num_rows(self):
        """Test that ``_sample_rows`` falls back to ``synthesizer._num_rows``."""
        synthesizer = Mock()
        synthesizer._num_rows = 10
        instance = Mock()

        # Run
        result = BaseHierarchicalSampler._sample_rows(instance, synthesizer)

        # Assert
        assert result == synthesizer._sample_batch.return_value
        synthesizer._sample_batch.assert_called_once_with(10, keep_extra_columns=True)

    def test__add_child_rows(self):
        """Test adding child rows when sampled data is empty."""
        # Setup
        instance = Mock()
        child_synthesizer_mock = Mock()
        instance._recreate_child_synthesizer.return_value = child_synthesizer_mock

        metadata = Mock()
        sessions_meta = Mock()
        users_meta = Mock()
        users_meta.primary_key = 'user_id'
        metadata.tables = {'users': users_meta, 'sessions': sessions_meta}
        metadata._get_foreign_keys.return_value = ['user_id']
        instance.metadata = metadata

        instance._sample_rows.return_value = pd.DataFrame({
            'session_id': ['a', 'b', 'c'],
            'os': ['linux', 'mac', 'win'],
            'country': ['us', 'us', 'es'],
        })
        parent_row = pd.DataFrame({
            'user_id': [1, 2, 3],
            'name': ['John', 'Doe', 'Johanna'],
            '__sessions__user_id__num_rows': [10, 10, 10],
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
            'user_id': [1, 2, 3],
        })
        pd.testing.assert_frame_equal(sampled_data['sessions'], expected_result)

    def test__add_child_rows_with_sampled_data(self):
        """Test adding child rows when sampled data contains values.

        The new sampled data has to be concatenated to the current sampled data.
        """
        # Setup
        instance = Mock()
        child_synthesizer_mock = Mock()
        instance._recreate_child_synthesizer.return_value = child_synthesizer_mock

        metadata = Mock()
        sessions_meta = Mock()
        users_meta = Mock()
        users_meta.primary_key.return_value = 'user_id'
        metadata.tables = {'users': users_meta, 'sessions': sessions_meta}
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
            'name': ['John', 'Doe', 'Johanna'],
            '__sessions__user_id__num_rows': [10, 10, 10],
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
            instance, 'sessions', 'users', parent_row, sampled_data
        )

        # Assert
        expected_result = pd.DataFrame({
            'user_id': [0, 1, 0, 1, 2, 3],
            'session_id': ['d', 'e', 'f', 'a', 'b', 'c'],
            'os': ['linux', 'mac', 'win', 'linux', 'mac', 'win'],
            'country': ['us', 'us', 'es', 'us', 'us', 'es'],
        })
        pd.testing.assert_frame_equal(sampled_data['sessions'], expected_result)

    def test__sample_children(self):
        """Test sampling the children of a table.

        ``_sample_table`` does not sample the root parents of a graph, only the children.
        """

        # Setup
        def sample_children(table_name, sampled_data, scale):
            sampled_data['transactions'] = pd.DataFrame({
                'transaction_id': [1, 2, 3],
                'session_id': ['a', 'a', 'b'],
            })

        def _add_child_rows(child_name, parent_name, parent_row, sampled_data, num_rows=None):
            if parent_name == 'users':
                if parent_row['user_id'] == 1:
                    sampled_data[child_name] = pd.DataFrame({
                        'user_id': [1, 1],
                        'session_id': ['a', 'b'],
                        'os': ['windows', 'linux'],
                        'country': ['us', 'us'],
                    })

                if parent_row['user_id'] == 3:
                    row = pd.DataFrame({
                        'user_id': [3],
                        'session_id': ['c'],
                        'os': ['mac'],
                        'country': ['es'],
                    })
                    sampled_data[child_name] = pd.concat([
                        sampled_data[child_name],
                        row,
                    ]).reset_index(drop=True)

        instance = Mock()
        instance.metadata._get_child_map.return_value = {'users': ['sessions', 'transactions']}
        instance.metadata._get_parent_map.return_value = {'users': []}
        instance.metadata._get_foreign_keys.return_value = ['user_id']
        instance._table_sizes = {'users': 10, 'sessions': 5, 'transactions': 3}
        instance._table_synthesizers = {'users': Mock()}
        instance._sample_children = sample_children
        instance._add_child_rows.side_effect = _add_child_rows
        instance._null_child_synthesizers = {}
        instance._null_foreign_key_percentages = {'__sessions__user_id': 0}

        # Run
        result = {'users': pd.DataFrame({'user_id': [1, 3]})}
        BaseHierarchicalSampler._sample_children(
            self=instance, table_name='users', sampled_data=result
        )

        # Assert
        expected_calls = [
            call(
                child_name='sessions',
                parent_name='users',
                parent_row=SeriesMatcher(pd.Series({'user_id': 1}, name=0, dtype=object)),
                sampled_data=result,
            ),
            call(
                child_name='sessions',
                parent_name='users',
                parent_row=SeriesMatcher(pd.Series({'user_id': 3}, name=1, dtype=object)),
                sampled_data=result,
            ),
        ]
        expected_result = {
            'users': pd.DataFrame({'user_id': [1, 3]}),
            'sessions': pd.DataFrame({
                'user_id': [1, 1, 3],
                'session_id': ['a', 'b', 'c'],
                'os': ['windows', 'linux', 'mac'],
                'country': ['us', 'us', 'es'],
            }),
            'transactions': pd.DataFrame({
                'transaction_id': [1, 2, 3],
                'session_id': ['a', 'a', 'b'],
            }),
        }
        instance._add_child_rows.assert_has_calls(expected_calls)
        for result_frame, expected_frame in zip(result.values(), expected_result.values()):
            pd.testing.assert_frame_equal(result_frame, expected_frame)

    def test__sample_children_no_rows_sampled(self):
        """Test sampling the children of a table where no rows created and no ``num_rows`` column.

        ``_sample_table`` should select the parent row with the highest ``num_rows``
        value and force a child to be created from that row.
        """

        # Setup
        def sample_children(table_name, sampled_data, scale):
            sampled_data['transactions'] = pd.DataFrame({
                'transaction_id': [1, 2],
                'session_id': ['a', 'a'],
            })

        def _add_child_rows(child_name, parent_name, parent_row, sampled_data, num_rows=None):
            if num_rows is not None:
                sampled_data['sessions'] = pd.DataFrame({'user_id': [1], 'session_id': ['a']})

        instance = Mock()
        instance.metadata._get_child_map.return_value = {'users': ['sessions', 'transactions']}
        instance.metadata._get_parent_map.return_value = {'users': []}
        instance.metadata._get_foreign_keys.return_value = ['user_id']
        instance._table_sizes = {'users': 10, 'sessions': 5, 'transactions': 3}
        instance._table_synthesizers = {'users': Mock()}
        instance._sample_children = sample_children
        instance._add_child_rows.side_effect = _add_child_rows
        instance._null_foreign_key_percentages = {'__sessions__user_id': 0}

        # Run
        result = {
            'users': pd.DataFrame({
                'user_id': [1],
                '__sessions__user_id__num_rows': pd.Series([1], dtype=object),
            })
        }
        BaseHierarchicalSampler._sample_children(
            self=instance, table_name='users', sampled_data=result
        )

        # Assert
        expected_parent_row = pd.Series(
            {'user_id': 1, '__sessions__user_id__num_rows': 1}, name=0, dtype=object
        )
        expected_calls = [
            call(
                child_name='sessions',
                parent_name='users',
                parent_row=SeriesMatcher(expected_parent_row),
                sampled_data=result,
            ),
            call(
                child_name='sessions',
                parent_name='users',
                parent_row=SeriesMatcher(expected_parent_row),
                sampled_data=result,
                num_rows=1,
            ),
        ]
        expected_result = {
            'users': pd.DataFrame({
                'user_id': [1],
                '__sessions__user_id__num_rows': pd.Series([1], dtype=object),
            }),
            'sessions': pd.DataFrame({
                'user_id': [1],
                'session_id': ['a'],
            }),
            'transactions': pd.DataFrame({'transaction_id': [1, 2], 'session_id': ['a', 'a']}),
        }
        instance._add_child_rows.assert_has_calls(expected_calls)
        for result_frame, expected_frame in zip(result.values(), expected_result.values()):
            pd.testing.assert_frame_equal(result_frame, expected_frame)

    def test__finalize(self):
        """Test that finalize removes extra columns from the sampled data."""
        # Setup
        instance = Mock()
        metadata = Mock()

        def get_column_names_mock(table_name):
            mapping = {
                'users': ['user_id', 'name'],
                'sessions': ['user_id', 'session_id', 'os', 'country'],
                'transactions': ['transaction_id', 'session_id'],
            }
            return mapping[table_name]

        metadata.get_column_names = Mock(side_effect=get_column_names_mock)
        instance.get_metadata = Mock(return_value=metadata)
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
            'country': str,
        }
        transactions_synth = Mock()
        transactions_synth._data_processor._dtypes = {'transaction_id': np.int64, 'session_id': str}

        instance._table_synthesizers = {
            'users': users_synth,
            'sessions': sessions_synth,
            'transactions': transactions_synth,
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

    @patch('sdv.sampling.hierarchical_sampler.LOGGER')
    def test__finalize_no_matching_dtype(self, mock_logging):
        """Test that finalize removes extra columns from the sampled data."""
        # Setup
        instance = Mock()
        metadata = Mock()

        def get_column_names_mock(table_name):
            mapping = {
                'users': ['user_id', 'name'],
                'sessions': ['user_id', 'session_id', 'os', 'country'],
                'transactions': ['transaction_id', 'session_id'],
            }
            return mapping[table_name]

        metadata.get_column_names = Mock(side_effect=get_column_names_mock)
        instance.get_metadata = Mock(return_value=metadata)
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
        # Incorrectly label data_processor type
        sessions_synth._data_processor._dtypes = {
            'user_id': np.int64,
            'session_id': np.int64,  # Should be str
            'os': str,
            'country': str,
        }
        transactions_synth = Mock()
        transactions_synth._data_processor._dtypes = {'transaction_id': np.int64, 'session_id': str}

        instance._table_synthesizers = {
            'users': users_synth,
            'sessions': sessions_synth,
            'transactions': transactions_synth,
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

        # Confirm log was called
        mock_logging.info.assert_called_once_with(
            "Could not cast back to column's original dtype, keeping original typing."
        )

    def test__sample(self):
        """Test that the whole dataset is sampled.

        Sampling has the following steps:
        1. The root tables should be sampled first.
        2. Then the lineage for each root is sampled by calling ``_sample_children``.
        3. Any missing parent-child relationships are added using ``_add_foreign_key_columns``.
        4. All extra columns are dropped by calling ``_finalize``.
        """
        # Setup
        users = pd.DataFrame({'id': [1, 2, 3], 'name': ['John', 'Doe', 'Johanna']})
        sessions = pd.DataFrame({
            'user_id': [1, 1, 3],
            'session_id': ['a', 'b', 'c'],
            'os': ['windows', 'linux', 'mac'],
            'country': ['us', 'us', 'es'],
        })
        transactions = pd.DataFrame({
            'user_id': [1, 2, 3],
            'transaction_id': [1, 2, 3],
            'transaction_amount': [100, 1000, 200],
        })

        def _sample_children_dummy(table_name, sampled_data, scale):
            sampled_data['sessions'] = sessions
            sampled_data['transactions'] = transactions

        instance = Mock()
        instance._table_sizes = {'users': 3, 'transactions': 9, 'sessions': 5}
        instance.metadata.relationships = [
            {
                'parent_table_name': 'users',
                'parent_primary_key': 'id',
                'child_table_name': 'sessions',
                'child_foreign_key': 'user_id',
            },
            {
                'parent_table_name': 'users',
                'parent_primary_key': 'id',
                'child_table_name': 'transactions',
                'child_foreign_key': 'user_id',
            },
        ]
        users_synthesizer = Mock()
        instance._table_synthesizers = defaultdict(Mock, {'users': users_synthesizer})
        instance.metadata._get_parent_map.return_value = {
            'sessions': ['users'],
            'transactions': ['users'],
        }
        instance.metadata.tables = {
            'users': Mock(),
            'sessions': Mock(),
            'transactions': Mock(),
        }
        instance._sample_rows.return_value = users
        instance._sample_children.side_effect = _sample_children_dummy
        instance._reverse_transform_constraints = Mock(side_effect=lambda x: x)
        instance._align_columns_with_metadata_order = Mock(side_effect=lambda x: x)

        # Run
        result = BaseHierarchicalSampler._sample(instance)

        # Assert
        expected_sample = {
            'users': DataFrameMatcher(
                pd.DataFrame({'id': [1, 2, 3], 'name': ['John', 'Doe', 'Johanna']})
            ),
            'sessions': DataFrameMatcher(
                pd.DataFrame({
                    'user_id': [1, 1, 3],
                    'session_id': ['a', 'b', 'c'],
                    'os': ['windows', 'linux', 'mac'],
                    'country': ['us', 'us', 'es'],
                })
            ),
            'transactions': DataFrameMatcher(
                pd.DataFrame({
                    'user_id': [1, 2, 3],
                    'transaction_id': [1, 2, 3],
                    'transaction_amount': [100, 1000, 200],
                })
            ),
        }
        assert result == instance._finalize.return_value
        instance._sample_children.assert_called_once_with(
            table_name='users', sampled_data=expected_sample, scale=1.0
        )
        instance._add_foreign_key_columns.assert_has_calls([
            call(expected_sample['sessions'], expected_sample['users'], 'sessions', 'users'),
            call(
                expected_sample['transactions'], expected_sample['users'], 'transactions', 'users'
            ),
        ])
        instance._reverse_transform_constraints.assert_called_once_with(expected_sample)
        instance._finalize.assert_called_once_with(expected_sample)

    def test___enforce_table_size_too_many_rows(self):
        """Test it enforces the sampled data to have the same size as the real data.

        If the sampled data has more rows than the real data, _num_rows is decreased.
        """
        # Setup
        instance = MagicMock()
        data = {'parent': pd.DataFrame({'fk': ['a', 'b', 'c'], '__child__fk__num_rows': [1, 2, 3]})}
        instance.metadata._get_foreign_keys.return_value = ['fk']
        instance._min_child_rows = {'__child__fk__num_rows': 1}
        instance._max_child_rows = {'__child__fk__num_rows': 3}
        instance._table_sizes = {'child': 4}
        instance._null_foreign_key_percentages = {'__child__fk': 0}

        # Run
        BaseHierarchicalSampler._enforce_table_size(instance, 'child', 'parent', 1.0, data)

        # Assert
        assert data['parent']['__child__fk__num_rows'].to_list() == [1, 1, 2]

    def test___enforce_table_size_not_enough_rows(self):
        """Test it enforces the sampled data to have the same size as the real data.

        If the sampled data has less rows than the real data, _num_rows is increased.
        """
        # Setup
        instance = MagicMock()
        data = {'parent': pd.DataFrame({'fk': ['a', 'b', 'c'], '__child__fk__num_rows': [1, 1, 1]})}
        instance.metadata._get_foreign_keys.return_value = ['fk']
        instance._min_child_rows = {'__child__fk__num_rows': 1}
        instance._max_child_rows = {'__child__fk__num_rows': 3}
        instance._table_sizes = {'child': 4}
        instance._null_foreign_key_percentages = {'__child__fk': 0}

        # Run
        BaseHierarchicalSampler._enforce_table_size(instance, 'child', 'parent', 1.0, data)

        # Assert
        assert data['parent']['__child__fk__num_rows'].to_list() == [2, 1, 1]

    def test___enforce_table_size_clipping(self):
        """Test it enforces the sampled data to have the same size as the real data.

        When the sampled num_rows is outside the min and max range, it should be clipped.
        """
        # Setup
        instance = MagicMock()
        data = {'parent': pd.DataFrame({'fk': ['a', 'b', 'c'], '__child__fk__num_rows': [1, 2, 5]})}
        instance.metadata._get_foreign_keys.return_value = ['fk']
        instance._min_child_rows = {'__child__fk__num_rows': 2}
        instance._max_child_rows = {'__child__fk__num_rows': 4}
        instance._table_sizes = {'child': 8}
        instance._null_foreign_key_percentages = {'__child__fk': 0}

        # Run
        BaseHierarchicalSampler._enforce_table_size(instance, 'child', 'parent', 1.0, data)

        # Assert
        assert data['parent']['__child__fk__num_rows'].to_list() == [2, 2, 4]

    def test___enforce_table_size_too_small_sample(self):
        """Test it enforces the sampled data to have the same size as the real data.

        If the sample scale is too small ensure that the function doesn't error out.
        """
        # Setup
        instance = MagicMock()
        data = {'parent': pd.DataFrame({'fk': ['a', 'b', 'c'], '__child__fk__num_rows': [1, 2, 3]})}
        instance.metadata._get_foreign_keys.return_value = ['fk']
        instance._min_child_rows = {'__child__fk__num_rows': 1}
        instance._max_child_rows = {'__child__fk__num_rows': 3}
        instance._table_sizes = {'child': 4}
        instance._null_foreign_key_percentages = {'__child__fk': 0}

        # Run
        BaseHierarchicalSampler._enforce_table_size(instance, 'child', 'parent', 0.001, data)

        # Assert
        assert data['parent']['__child__fk__num_rows'].to_list() == [0, 0, 0]

    def test___align_columns_with_metadata_order_reorders_and_appends_extra_columns(self):
        """Test that reorders columns to match metadata and appends extra columns at the end."""
        # Setup
        instance = Mock()

        metadata = Mock()
        metadata.tables = {'table': []}
        metadata.get_column_names.return_value = ['a', 'b', 'c']
        instance.get_metadata.return_value = metadata

        data = {'table': pd.DataFrame(columns=['c', 'extra1', 'a', 'b', 'extra2'])}

        # Run
        result = BaseHierarchicalSampler._align_columns_with_metadata_order(instance, data)

        # Assert
        assert result['table'].columns.to_list() == ['a', 'b', 'c', 'extra1', 'extra2']
