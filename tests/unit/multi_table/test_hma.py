import re
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from sdv.multi_table.hma import HMASynthesizer
from sdv.single_table.copulas import GaussianCopulaSynthesizer
from tests.utils import get_multi_table_data, get_multi_table_metadata


class TestHMASynthesizer:

    def test___init__(self):
        """Test the default initialization of the ``HMASynthesizer``."""
        # Run
        metadata = get_multi_table_metadata()
        metadata.validate = Mock()
        instance = HMASynthesizer(metadata)

        # Assert
        assert instance.metadata == metadata
        assert isinstance(instance._table_synthesizers['nesreca'], GaussianCopulaSynthesizer)
        assert isinstance(instance._table_synthesizers['oseba'], GaussianCopulaSynthesizer)
        assert isinstance(instance._table_synthesizers['upravna_enota'], GaussianCopulaSynthesizer)
        assert instance._table_parameters == {
            'nesreca': {'default_distribution': 'beta'},
            'oseba': {'default_distribution': 'beta'},
            'upravna_enota': {'default_distribution': 'beta'},
        }
        instance.metadata.validate.assert_called_once_with()

    def test__get_extension(self):
        """Test the ``_get_extension`` method.

        Test that the resulting dataframe contains extended columns using the names
        and parameters from a trained ``copulas.univariate`` model.
        """
        # Setup
        metadata = get_multi_table_metadata()
        child_table = pd.DataFrame({
            'id_nesreca': [0, 1, 2, 3],
            'upravna_enota': [0, 1, 2, 3]
        })
        instance = HMASynthesizer(metadata)

        # Run
        result = instance._get_extension('nesreca', child_table, 'upravna_enota')

        # Assert
        expected = pd.DataFrame({
            '__nesreca__upravna_enota__univariates__id_nesreca__a': [1., 1., 1., 1.],
            '__nesreca__upravna_enota__univariates__id_nesreca__b': [1., 1., 1., 1.],
            '__nesreca__upravna_enota__univariates__id_nesreca__loc': [0., 1., 2., 3.],
            '__nesreca__upravna_enota__univariates__id_nesreca__scale': [np.nan] * 4,
            '__nesreca__upravna_enota__num_rows': [1., 1., 1., 1.]
        })

        pd.testing.assert_frame_equal(result, expected)

    def test__get_extension_foreign_key_only(self):
        """Test the ``_get_extension`` method.

        Test when foreign key only is passed, just the ``num_rows`` is being captured.
        """
        # Setup
        instance = Mock()
        instance._get_all_foreign_keys.return_value = ['id_upravna_enota']
        instance._table_synthesizers = {'nesreca': Mock()}
        child_table = pd.DataFrame({
            'id_upravna_enota': [0, 1, 2, 3]
        })

        # Run
        result = HMASynthesizer._get_extension(
            instance,
            'nesreca',
            child_table,
            'id_upravna_enota'
        )

        # Assert
        expected = pd.DataFrame({
            '__nesreca__id_upravna_enota__num_rows': [1, 1, 1, 1]
        })

        pd.testing.assert_frame_equal(result, expected)

    def test__get_foreign_keys(self):
        """Test that this method returns the foreign keys for a given table name and child name."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = HMASynthesizer(metadata)

        # Run
        result = instance._get_foreign_keys('nesreca', 'oseba')

        # Assert
        assert result == ['id_nesreca']

    def test__get_all_foreign_keys(self):
        """Test that this method returns all the foreign keys for a given table name."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = HMASynthesizer(metadata)

        # Run
        result = instance._get_all_foreign_keys('nesreca')

        # Assert
        assert result == ['upravna_enota']

    def test__extend_table(self):
        """Test that ``extend_table`` extends the current table with extra columns.

        This also updates ``self._modeled_tables`` and ``self._max_child_rows``.
        """
        # Setup
        metadata = get_multi_table_metadata()
        instance = HMASynthesizer(metadata)
        metadata.add_column('nesreca', 'value', sdtype='numerical')
        metadata.add_column('oseba', 'oseba_value', sdtype='numerical')
        metadata.add_column('upravna_enota', 'name', sdtype='categorical')

        data = get_multi_table_data()
        data['nesreca']['value'] = [0, 1, 2, 3]
        data['oseba']['oseba_value'] = [0, 1, 2, 3]

        # Run
        result = instance._extend_table(data['nesreca'], data, 'nesreca')

        # Assert
        expected_result = pd.DataFrame({
            'id_nesreca': [0, 1, 2, 3],
            'upravna_enota': [0, 1, 2, 3],
            'nesreca_val': [0, 1, 2, 3],
            'value': [0, 1, 2, 3],
            '__oseba__id_nesreca__covariance__0__0': [0.] * 4,
            '__oseba__id_nesreca__univariates__oseba_val__a': [1.] * 4,
            '__oseba__id_nesreca__univariates__oseba_val__b': [1.] * 4,
            '__oseba__id_nesreca__univariates__oseba_val__loc': [0., 1., 2., 3.],
            '__oseba__id_nesreca__univariates__oseba_val__scale': [np.nan] * 4,
            '__oseba__id_nesreca__univariates__oseba_value__a': [1.] * 4,
            '__oseba__id_nesreca__univariates__oseba_value__b': [1.] * 4,
            '__oseba__id_nesreca__univariates__oseba_value__loc': [0., 1., 2., 3.],
            '__oseba__id_nesreca__univariates__oseba_value__scale': [np.nan] * 4,
            '__oseba__id_nesreca__num_rows': [1.] * 4,
        })

        pd.testing.assert_frame_equal(expected_result, result)
        assert instance._modeled_tables == ['oseba']
        assert instance._max_child_rows['__oseba__id_nesreca__num_rows'] == 1

    def test__pop_foreign_keys(self):
        """Test that this method removes the foreign keys from the ``table_data``."""
        # Setup
        instance = Mock()
        instance._get_all_foreign_keys.return_value = ['a', 'b']
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [2, 3, 4],
            'c': ['John', 'Doe', 'Johanna']
        })

        # Run
        result = HMASynthesizer._pop_foreign_keys(instance, table_data, 'table_name')

        # Assert
        pd.testing.assert_frame_equal(pd.DataFrame({'c': ['John', 'Doe', 'Johanna']}), table_data)
        np.testing.assert_array_equal(result['a'], [1, 2, 3])
        np.testing.assert_array_equal(result['b'], [2, 3, 4])

    def test__clear_nans(self):
        """Test that this method clears all the nans and substitutes them with expected values."""
        # Setup
        data = pd.DataFrame({
            'numerical': [0, 1, 2, 3, np.nan, np.nan],
            'categorical': ['John', np.nan, 'Johanna', 'John', np.nan, 'Doe'],
        })

        # Run
        HMASynthesizer._clear_nans(data)

        # Assert
        expected_data = pd.DataFrame({
            'numerical': [0, 1, 2, 3, 1.5, 1.5],
            'categorical': ['John', 'John', 'Johanna', 'John', 'John', 'Doe']
        })
        pd.testing.assert_frame_equal(expected_data, data)

    def test__model_table(self):
        """Test that ``_model_table`` performs the modeling.

        Modeling consists of getting the table for the given table name,
        learning the size of this table, removing the foreign keys and clearing
        any null values by using the ``_clear_nans`` method. Then, fitting the table model by
        calling ``fit_processed_data``,  adding back the foreign keys, updating the ``tables`` and
        marking the table name as modeled within the ``instance._modeled_tables``.
        """
        # Setup
        nesreca_model = Mock()
        instance = Mock()
        instance._synthesizer = GaussianCopulaSynthesizer

        instance._modeled_tables = []
        instance._table_sizes = {}
        instance._table_synthesizers = {'nesreca': nesreca_model}
        instance._pop_foreign_keys.return_value = {'fk': [1, 2, 3]}
        data = pd.DataFrame({
            'id_nesreca': [0, 1, 2],
            'upravna_enota': [0, 1, 2]
        })
        extended_data = pd.DataFrame({
            'id_nesreca': [0, 1, 2],
            'upravna_enota': [0, 1, 2],
            'extended': ['a', 'b', 'c']
        })

        instance._extend_table.return_value = extended_data

        tables = {'nesreca': data}

        # Run
        result = HMASynthesizer._model_table(instance, 'nesreca', tables)

        # Assert
        expected_result = pd.DataFrame({
            'id_nesreca': [0, 1, 2],
            'upravna_enota': [0, 1, 2],
            'extended': ['a', 'b', 'c'],
            'fk': [1, 2, 3]
        })
        pd.testing.assert_frame_equal(expected_result, result)

        instance._extend_table.assert_called_once_with(data, tables, 'nesreca')
        instance._pop_foreign_keys.assert_called_once_with(extended_data, 'nesreca')
        instance._clear_nans(extended_data)
        nesreca_model.fit_processed_data.assert_called_once_with(extended_data)

        assert instance._modeled_tables == ['nesreca']
        assert instance._table_sizes == {'nesreca': 3}

    def test__fit(self):
        """Test that ``_fit`` calls ``_model_table`` only if the table has no parents."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = HMASynthesizer(metadata)
        instance._model_table = Mock()
        data = get_multi_table_data()
        data['nesreca']['value'] = [0, 1, 2, 3]
        data['oseba']['oseba_value'] = [0, 1, 2, 3]

        # Run
        instance._fit(data)

        # Assert
        instance._model_table.assert_called_once_with('upravna_enota', data)

    def test__finalize(self):
        """Test that the finalize method applies the final touches to the generated data.

        The process consists of applying the propper data types to each table, and finding
        foreign keys if those are not present in the current sampled data.
        """
        # Setup
        instance = Mock()
        metadata = Mock()
        metadata._get_parent_map.return_value = {
            'sessions': ['users'],
            'transactions': ['sessions']
        }
        instance.metadata = metadata

        instance._get_foreign_keys.side_effect = [['user_id'], ['session_id']]
        instance._find_parent_ids.return_value = pd.Series(['a', 'a', 'b'], name='session_id')

        sampled_data = {
            'users': pd.DataFrame({
                'user_id': [0, 1, 2],
                'name': ['John', 'Doe', 'Johanna'],
                'additional_column': [0.1, 0.2, 0.3],
                'another_additional_column': [0.1, 0.2, 0.5]
            }, dtype=np.int64),
            'sessions': pd.DataFrame({
                'user_id': [1, 2, 1],
                'session_id': ['a', 'b', 'c'],
                'os': ['linux', 'mac', 'win'],
                'country': ['us', 'us', 'es']
            }, dtype=np.int64),
            'transactions': pd.DataFrame({
                'transaction_id': [1, 2, 3],
            }, dtype=np.int64),
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
        result = HMASynthesizer._finalize(instance, sampled_data)

        # Assert
        expected_result = {
            'users': pd.DataFrame({
                'user_id': [0, 1, 2],
                'name': ['John', 'Doe', 'Johanna'],
            }, dtype=np.int64),
            'sessions': pd.DataFrame({
                'user_id': [1, 2, 1],
                'session_id': ['a', 'b', 'c'],
                'os': ['linux', 'mac', 'win'],
                'country': ['us', 'us', 'es'],
            }, dtype=np.int64),
            'transactions': pd.DataFrame({
                'transaction_id': [1, 2, 3],
                'session_id': ['a', 'a', 'b']
            }, dtype=np.int64),
        }
        for result_frame, expected_frame in zip(result.values(), expected_result.values()):
            pd.testing.assert_frame_equal(result_frame, expected_frame)

    def test__extract_parameters(self):
        """Test that parameters are being returned without the prefix."""
        # Setup
        parent_row = pd.Series({
            '__sessions__user_id__num_rows': 10,
            '__sessions__user_id__a': 1.0,
            '__sessions__user_id__b': 0.2,
            '__sessions__user_id__loc': 0.5,
            '__sessions__user_id__scale': 0.25
        })
        instance = Mock()
        instance._max_child_rows = {'__sessions__user_id__num_rows': 10}

        # Run
        result = HMASynthesizer._extract_parameters(instance, parent_row, 'sessions', 'user_id')

        # Assert
        expected_result = {
            'a': 1.0,
            'b': 0.2,
            'loc': 0.5,
            'num_rows': 10.0,
            'scale': 0.25
        }

        assert result == expected_result

    def test__process_samples(self):
        """Test the ``_process_samples``.

        Test that the method retrieves the ``data_processor`` from the fitted ``table_synthesizer``
        and performs a ``reverse_transform`` and returns the data in the real space.
        """
        # Setup
        sampled_rows = pd.DataFrame({
            'name': [0.1, 0.25, 0.35],
            'a': [1.0, 0.25, 0.5],
            'b': [0.2, 0.6, 0.9],
            'loc': [0.5, 0.1, 0.2],
            'num_rows': [1, 2, 3],
            'scale': [0.25, 0.35, 0.15]
        })
        instance = Mock()
        users_synthesizer = Mock()
        users_synthesizer._data_processor.reverse_transform.return_value = pd.DataFrame({
            'user_id': [0, 1, 2],
            'name': ['John', 'Doe', 'Johanna']
        })
        instance._table_synthesizers = {'users': users_synthesizer}

        # Run
        result = HMASynthesizer._process_samples(instance, 'users', sampled_rows)

        # Assert
        expected_result = pd.DataFrame({
            'user_id': [0, 1, 2],
            'name': ['John', 'Doe', 'Johanna'],
            'a': [1.0, 0.25, 0.5],
            'b': [0.2, 0.6, 0.9],
            'loc': [0.5, 0.1, 0.2],
            'num_rows': [1, 2, 3],
            'scale': [0.25, 0.35, 0.15]
        })
        result = result.reindex(sorted(result.columns), axis=1)
        expected_result = expected_result.reindex(sorted(expected_result.columns), axis=1)
        pd.testing.assert_frame_equal(result, expected_result)

    def test__sample_rows(self):
        """Test sample rows.

        Test that sampling rows will return the reverse transformed data with the extension columns
        sampled by the model.
        """
        # Setup
        synthesizer = Mock()
        instance = Mock()

        # Run
        result = HMASynthesizer._sample_rows(instance, synthesizer, 'users', 10)

        # Assert
        assert result == instance._process_samples.return_value
        instance._process_samples.assert_called_once_with(
            'users',
            synthesizer._sample.return_value
        )
        synthesizer._sample.assert_called_once_with(10)

    def test__get_child_synthesizer(self):
        """Test that this method returns a synthesizer for the given child table."""
        # Setup
        instance = Mock()
        parent_row = 'row'
        table_name = 'users'
        foreign_key = 'session_id'
        table_meta = Mock()
        instance.metadata.tables = {'users': table_meta}
        instance._synthesizer_kwargs = {'a': 1}

        # Run
        synthesizer = HMASynthesizer._get_child_synthesizer(
            instance,
            parent_row,
            table_name,
            foreign_key
        )

        # Assert
        assert synthesizer == instance._synthesizer.return_value
        instance._synthesizer.assert_called_once_with(table_meta, a=1)
        synthesizer._set_parameters.assert_called_once_with(
            instance._extract_parameters.return_value
        )
        instance._extract_parameters.assert_called_once_with(parent_row, table_name, foreign_key)

    def test__sample_child_rows(self):
        """Test the sampling of child rows when sampled data is empty."""
        # Setup
        instance = Mock()
        instance._get_foreign_keys.return_value = ['user_id']
        instance._extract_parameters.return_value = {
            'a': 1.0,
            'b': 0.2,
            'loc': 0.5,
            'num_rows': 10.0,
            'scale': 0.25
        }

        metadata = Mock()
        sessions_meta = Mock()
        users_meta = Mock()
        users_meta.primary_key.return_value = 'user_id'
        metadata.tables = {
            'users': users_meta,
            'sessions': sessions_meta
        }
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
        sampled_data = {}

        # Run
        HMASynthesizer._sample_child_rows(instance, 'sessions', 'users', parent_row, sampled_data)

        # Assert
        expected_result = pd.DataFrame({
            'session_id': ['a', 'b', 'c'],
            'os': ['linux', 'mac', 'win'],
            'country': ['us', 'us', 'es'],
            'user_id': [1, 2, 3]
        })
        pd.testing.assert_frame_equal(sampled_data['sessions'], expected_result)

    def test__sample_child_rows_with_sampled_data(self):
        """Test the sampling of child rows when sampled data contains values.

        The new sampled data has to be concatenated to the current sampled data.
        """
        # Setup
        instance = Mock()
        instance._get_foreign_keys.return_value = ['user_id']
        instance._extract_parameters.return_value = {
            'a': 1.0,
            'b': 0.2,
            'loc': 0.5,
            'num_rows': 10.0,
            'scale': 0.25
        }

        metadata = Mock()
        sessions_meta = Mock()
        users_meta = Mock()
        users_meta.primary_key.return_value = 'user_id'
        metadata.tables = {
            'users': users_meta,
            'sessions': sessions_meta
        }
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
        HMASynthesizer._sample_child_rows(instance, 'sessions', 'users', parent_row, sampled_data)

        # Assert
        expected_result = pd.DataFrame({
            'user_id': [0, 1, 0, 1, 2, 3],
            'session_id': ['d', 'e', 'f', 'a', 'b', 'c'],
            'os': ['linux', 'mac', 'win', 'linux', 'mac', 'win'],
            'country': ['us', 'us', 'es', 'us', 'us', 'es'],
        })
        pd.testing.assert_frame_equal(sampled_data['sessions'], expected_result)

    def test__sample_children(self):
        """Test that child tables are being sampled recursively."""
        # Setup
        def update_sampled_data(child_name, table_name, row, sampled_data):
            sampled_data['sessions'] = pd.DataFrame({
                'user_id': [1],
                'session_id': ['d'],
                'os': ['linux'],
                'country': ['us'],
            })

        metadata = Mock()
        metadata._get_child_map.return_value = {'users': ['sessions']}
        instance = Mock()
        table_rows = pd.DataFrame({
            'user_id': [1, 2, 3],
            'name': ['John', 'Doe', 'Johanna']
        })
        sampled_data = {}
        instance.metadata = metadata
        instance._sample_child_rows.side_effect = update_sampled_data

        # Run
        HMASynthesizer._sample_children(instance, 'users', sampled_data, table_rows)

        # Assert
        assert instance._sample_child_rows.call_count == 3
        sample_calls = instance._sample_child_rows.call_args_list
        pd.testing.assert_series_equal(
            sample_calls[0][0][2],
            pd.Series({'user_id': 1, 'name': 'John'}, name=0)
        )
        pd.testing.assert_series_equal(
            sample_calls[1][0][2],
            pd.Series({'user_id': 2, 'name': 'Doe'}, name=1)
        )
        pd.testing.assert_series_equal(
            sample_calls[2][0][2],
            pd.Series({'user_id': 3, 'name': 'Johanna'}, name=2)
        )

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

        instance = Mock()
        instance._table_sizes = {'users': 10}
        instance._table_synthesizers = {'users': Mock()}
        instance._sample_children.side_effect = sample_children
        instance._sample_rows.return_value = pd.DataFrame({
            'user_id': [1, 2, 3],
            'name': ['John', 'Doe', 'Johanna']
        })

        # Run
        result = HMASynthesizer._sample_table(instance, 'users')

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

    def test__sample(self):
        """Test that the ``_sample_table`` is called for tables that don't have parents."""
        # Setup
        instance = Mock()
        instance.metadata._get_parent_map.return_value = {
            'sessions': ['users'],
            'transactions': ['sessions']
        }
        instance.metadata.tables = {
            'users': Mock(),
            'sessions': Mock(),
            'transactions': Mock(),
        }

        # Run
        result = HMASynthesizer._sample(instance)

        # Assert
        assert result == instance._finalize.return_value
        instance._sample_table.assert_called_once_with('users', scale=1, sampled_data={})
        instance._finalize.assert_called_once_with({})

    def test_get_learned_distributions(self):
        """Test that ``get_learned_distributions`` returns a dict.

        Test that it returns a dictionary with the name of the columns and the learned
        distribution and its parameters.
        """
        # Setup
        metadata = get_multi_table_metadata()
        instance = HMASynthesizer(metadata)
        data = {
            'nesreca': pd.DataFrame({
                'id_nesreca': np.arange(10),
                'upravna_enota': np.arange(10),
                'nesreca_val': np.arange(10),
            }),
            'oseba': pd.DataFrame({
                'upravna_enota': np.arange(10),
                'id_nesreca': np.arange(10),
                'oseba_val': np.arange(10),
            }),
            'upravna_enota': pd.DataFrame({
                'id_upravna_enota': np.arange(10),
                'upravna_val': [10, np.nan] * 5,
            }),
        }

        instance.fit(data)

        # Run
        result = instance.get_learned_distributions('upravna_enota')

        # Assert
        assert list(result) == ['upravna_val']
        assert result['upravna_val'] == {
            'distribution': 'beta',
            'learned_parameters': {
                'a': 1.0,
                'b': 1.0,
                'loc': 10.0,
                'scale': 0.0
            }
        }

    def test_get_learned_distributions_raises_an_error(self):
        """Test that ``get_learned_distributions`` raises an error."""
        # Setup
        metadata = get_multi_table_metadata()
        metadata.add_column('nesreca', 'value', sdtype='numerical')
        metadata.add_column('oseba', 'value', sdtype='numerical')
        metadata.add_column('upravna_enota', 'a_value', sdtype='numerical')
        instance = HMASynthesizer(metadata)

        # Run and Assert
        error_msg = re.escape(
            "Distributions have not been learned yet. Please fit your model first using 'fit'."
        )
        with pytest.raises(ValueError, match=error_msg):
            instance.get_learned_distributions('upravna_enota')
