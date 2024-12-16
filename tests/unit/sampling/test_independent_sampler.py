import re
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from sdv.sampling.independent_sampler import BaseIndependentSampler
from tests.utils import DataFrameMatcher, get_multi_table_metadata


class TestBaseIndependentSampler:
    def test___init__(self):
        """Test the default initialization of the ``BaseIndependentSampler``."""
        # Run
        metadata = get_multi_table_metadata()
        instance = BaseIndependentSampler(metadata, table_synthesizers={}, table_sizes={})

        # Assert
        assert instance.metadata == metadata
        assert instance._table_synthesizers == {}
        assert instance._table_sizes == {}

    def test__add_foreign_key_columns(self):
        """Test that ``_add_foreign_key_columns`` raises a ``NotImplementedError``."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseIndependentSampler(metadata, table_synthesizers={}, table_sizes={})

        # Run and Assert
        with pytest.raises(NotImplementedError, match=''):
            instance._add_foreign_key_columns(
                child_table=pd.DataFrame(),
                parent_table=pd.DataFrame(),
                child_name='oseba',
                parent_name='nescra',
            )

    def test__sample_table(self):
        """Test sampling a table."""
        # Setup
        table_synthesizer = Mock()
        table_synthesizer._sample_batch.return_value = pd.DataFrame({
            'user_id': [1, 2, 3],
            'name': ['John', 'Doe', 'Johanna'],
        })

        instance = Mock()
        instance._table_synthesizers = {'users': table_synthesizer}

        # Run
        result = {}
        BaseIndependentSampler._sample_table(instance, table_synthesizer, 'users', 3, result)

        # Assert
        expected_result = {
            'users': pd.DataFrame({
                'user_id': [1, 2, 3],
                'name': ['John', 'Doe', 'Johanna'],
            })
        }
        table_synthesizer._sample_batch.assert_called_once_with(3, keep_extra_columns=True)
        pd.testing.assert_frame_equal(result['users'], expected_result['users'])

    def test__connect_table(self):
        """Test the method adds all foreign key columns to each table."""

        def _get_all_foreign_keys(child):
            foreign_keys = {
                'users': [],
                'sessions': ['users_id'],
                'transactions': ['users_id', 'sessions_id'],
            }
            return foreign_keys[child]

        _add_foreign_key_columns_mock = Mock()

        def _add_foreign_key_columns(child_data, parent_data, child, parent):
            _add_foreign_key_columns_mock(child_data.copy(), parent_data, child, parent)
            child_data[f'{parent}_id'] = pd.Series(dtype='object')

        instance = Mock()
        instance.metadata.tables = {
            'users': Mock(),
            'sessions': Mock(),
            'transactions': Mock(),
        }
        instance.metadata._get_parent_map.return_value = {
            'sessions': {'users'},
            'transactions': {'sessions', 'users'},
            'users': set(),
        }
        instance.metadata._get_child_map.return_value = {
            'users': ['transactions', 'sessions'],
            'sessions': {'transactions'},
            'transactions': set(),
        }
        instance.metadata._get_all_foreign_keys.side_effect = _get_all_foreign_keys
        instance._add_foreign_key_columns.side_effect = _add_foreign_key_columns
        sampled_data = {
            'users': pd.DataFrame(dtype='object'),
            'sessions': pd.DataFrame(dtype='object'),
            'transactions': pd.DataFrame(dtype='object'),
        }

        # Run
        BaseIndependentSampler._connect_tables(instance, sampled_data)

        # Assert
        _add_foreign_key_columns_mock.assert_has_calls([
            call(
                DataFrameMatcher(pd.DataFrame(dtype='object')),
                DataFrameMatcher(pd.DataFrame(dtype='object')),
                'transactions',
                'users',
            ),
            call(
                DataFrameMatcher(pd.DataFrame(dtype='object')),
                DataFrameMatcher(pd.DataFrame(dtype='object')),
                'sessions',
                'users',
            ),
            call(
                DataFrameMatcher(pd.DataFrame(dtype='object', columns=['users_id'])),
                DataFrameMatcher(pd.DataFrame(dtype='object', columns=['users_id'])),
                'transactions',
                'sessions',
            ),
        ])

    def test__finalize(self):
        """Test that finalize removes extra columns from the sampled data."""
        # Setup
        instance = Mock()
        metadata = Mock()
        metadata._get_parent_map.return_value = {
            'sessions': ['users'],
            'transactions': ['sessions'],
            'users': set(),
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
        result = BaseIndependentSampler._finalize(instance, sampled_data)

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

    @patch('sdv.sampling.independent_sampler.LOGGER')
    def test__finalize_id_being_string(self, mock_logger):
        """Test that finalize function when an id column is string but dtype is int."""
        # Setup
        instance = Mock()
        metadata = Mock()
        metadata._get_parent_map.return_value = {
            'sessions': ['users'],
            'transactions': ['sessions'],
            'users': set(),
        }
        instance.metadata = metadata

        sampled_data = {
            'users': pd.DataFrame({
                'user_id': pd.Series(['a', 'b', 'c'], dtype=object),
                'name': pd.Series(['John', 'Doe', 'Johanna'], dtype=object),
                'additional_column': pd.Series([0.1, 0.2, 0.3], dtype=float),
                'another_additional_column': pd.Series([0.1, 0.2, 0.5], dtype=float),
            }),
            'sessions': pd.DataFrame({
                'user_id': pd.Series(['a', 'b', 'c'], dtype=object),
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
        users_synth._data_processor._DTYPE_TO_SDTYPE = {
            'i': 'numerical',
            'f': 'numerical',
            'O': 'categorical',
            'b': 'boolean',
            'M': 'datetime',
        }

        sessions_synth = Mock()
        sessions_synth._data_processor._dtypes = {
            'user_id': np.int64,
            'session_id': str,
            'os': str,
            'country': str,
        }

        sessions_synth._data_processor._DTYPE_TO_SDTYPE = {
            'i': 'numerical',
            'f': 'numerical',
            'O': 'categorical',
            'b': 'boolean',
            'M': 'datetime',
        }
        transactions_synth = Mock()
        transactions_synth._data_processor._dtypes = {'transaction_id': np.int64, 'session_id': str}
        transactions_synth._data_processor._DTYPE_TO_SDTYPE = {
            'i': 'numerical',
            'f': 'numerical',
            'O': 'categorical',
            'b': 'boolean',
            'M': 'datetime',
        }

        instance._table_synthesizers = {
            'users': users_synth,
            'sessions': sessions_synth,
            'transactions': transactions_synth,
        }

        # Run
        result = BaseIndependentSampler._finalize(instance, sampled_data)

        # Assert
        expected_result = {
            'users': pd.DataFrame({
                'user_id': pd.Series(['a', 'b', 'c'], dtype=object),
                'name': pd.Series(['John', 'Doe', 'Johanna'], dtype=object),
            }),
            'sessions': pd.DataFrame({
                'user_id': pd.Series(['a', 'b', 'c'], dtype=object),
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

        message_users = (
            "The real data in 'users' and column 'user_id' was stored as "
            "'<class 'numpy.int64'>' but the synthetic data "
            'could not be cast back to this type. If this is a problem, please check your input '
            'data and metadata settings.'
        )

        message_sessions = (
            "The real data in 'sessions' and column 'user_id' was stored as "
            "'<class 'numpy.int64'>' but the synthetic data "
            'could not be cast back to this type. If this is a problem, please check your input '
            'data and metadata settings.'
        )
        assert mock_logger.info.call_args_list == [call(message_users), call(message_sessions)]

    @patch('sdv.sampling.independent_sampler.LOGGER')
    def test__finalize_overflow(self, mock_logger):
        """Test it logs a warning when the synthetic data overflows."""
        # Setup
        sampled_data = {
            'table': pd.DataFrame({
                'id': [99999999999999999990, 99999999999999999991, 99999999999999999992]
            }),
        }
        parent_synthesizer = Mock()
        parent_synthesizer._data_processor._dtypes = {'id': 'int64'}
        instance = Mock()
        instance._table_synthesizers = {'table': parent_synthesizer}

        # Run
        BaseIndependentSampler._finalize(instance, sampled_data)

        # Assert
        message_users = (
            "The real data in 'table' and column 'id' was stored as "
            "'int64' but the synthetic data overflowed when casting back to "
            'this type. If this is a problem, please check your input data '
            'and metadata settings.'
        )
        assert mock_logger.debug.call_args_list == [call(message_users)]

    def test__sample(self):
        """Test that the ``_sample_table`` is called for root tables."""
        # Setup
        instance = Mock()
        _sample_table_mock = Mock()
        _connect_tables_mock = Mock()

        expected_sample = {
            'users': pd.DataFrame({'user_id': [1, 2, 3], 'name': ['John', 'Doe', 'Johanna']}),
            'sessions': pd.DataFrame({
                'user_id': [1, 1, 3],
                'session_id': ['a', 'b', 'c'],
                'os': ['windows', 'linux', 'mac'],
                'country': ['us', 'us', 'es'],
            }),
            'transactions': pd.DataFrame(dtype='Int64'),
        }
        connected_transactions = pd.DataFrame({
            'user_id': [1, 3, 1],
            'sessions_id': ['a', 'c', 'b'],
        })

        def _sample_table(synthesizer, table_name, num_rows, sampled_data):
            _sample_table_mock(
                synthesizer=synthesizer,
                table_name=table_name,
                num_rows=num_rows,
                sampled_data=sampled_data.copy(),
            )
            sampled_data[table_name] = expected_sample[table_name]

        def _connect_tables(sampled_data):
            _connect_tables_mock(sampled_data.copy())
            sampled_data['transactions'] = connected_transactions

        instance._sample_table.side_effect = _sample_table
        instance._connect_tables.side_effect = _connect_tables

        instance._table_sizes = {'users': 3, 'transactions': 9, 'sessions': 5}
        instance.metadata.relationships = [
            {
                'parent_table_name': 'users',
                'parent_primary_key': 'id',
                'child_table_name': 'sessions',
                'child_foreign_key': 'id',
            }
        ]
        users_synthesizer = Mock()
        sessions_synthesizer = Mock()
        transactions_synthesizer = Mock()
        instance._table_synthesizers = {
            'users': users_synthesizer,
            'sessions': sessions_synthesizer,
            'transactions': transactions_synthesizer,
        }
        instance.metadata.tables = {
            'users': Mock(),
            'sessions': Mock(),
            'transactions': Mock(),
        }

        # Run
        result = BaseIndependentSampler._sample(instance)

        # Assert
        users_call = call(
            synthesizer=users_synthesizer, table_name='users', num_rows=3, sampled_data={}
        )
        sessions_call = call(
            synthesizer=sessions_synthesizer,
            table_name='sessions',
            num_rows=5,
            sampled_data={'users': DataFrameMatcher(expected_sample['users'])},
        )
        transactions_call = call(
            synthesizer=transactions_synthesizer,
            table_name='transactions',
            num_rows=9,
            sampled_data={
                'users': DataFrameMatcher(expected_sample['users']),
                'sessions': DataFrameMatcher(expected_sample['sessions']),
            },
        )
        _sample_table_mock.assert_has_calls([users_call, sessions_call, transactions_call])

        _connect_tables_mock.assert_called_once_with({
            key: DataFrameMatcher(table) for key, table in expected_sample.items()
        })
        instance._finalize.assert_called_once_with({
            'users': DataFrameMatcher(expected_sample['users']),
            'sessions': DataFrameMatcher(expected_sample['sessions']),
            'transactions': DataFrameMatcher(connected_transactions),
        })
        assert result == instance._finalize.return_value

    def test__sample_too_small(self):
        """Test that the ``_sample`` method works even when sample size is very small."""
        instance = Mock()
        metadata = Mock()
        metadata.tables = {
            'guests': {
                'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1',
                'primary_key': 'key_1',
                'columns': {
                    'key_1': {
                        'sdtype': 'numerical',
                    }
                },
            },
            'hotels': {
                'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1',
                'primary_key': 'key_2',
                'columns': {
                    'key_2': {
                        'sdtype': 'numerical',
                    }
                },
            },
        }
        instance._table_sizes = {'hotels': 10, 'guests': 658}
        instance._table_synthesizers = {'hotels': Mock(), 'guests': Mock()}
        instance.metadata = metadata
        warning_msg = re.escape(
            "The 'scale' parameter is too small. Some tables may have 1 row."
            ' For better quality data, please choose a larger scale.'
        )
        with pytest.warns(Warning, match=warning_msg):
            BaseIndependentSampler._sample(instance, 0.01)
