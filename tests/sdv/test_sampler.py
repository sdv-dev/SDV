from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from sdv.data_navigator import CSVDataLoader, Table
from sdv.modeler import Modeler
from sdv.sampler import Sampler


class TestSampler(TestCase):

    @classmethod
    def setUpClass(cls):
        data_loader = CSVDataLoader('tests/data/meta.json')
        cls.data_navigator = data_loader.load_data()
        cls.data_navigator.transform_data()

        cls.modeler = Modeler(cls.data_navigator)
        cls.modeler.model_database()

    def setUp(self):
        self.sampler = Sampler(self.data_navigator, self.modeler)

    def test__rescale_values(self):
        """_rescale_values return and array satisfying  0 < array < 1."""
        # Setup
        data_navigator = MagicMock()
        modeler = MagicMock()
        sampler = Sampler(data_navigator, modeler)

        column = pd.Series([0.0, 5.0, 10], name='column')
        expected_result = pd.Series([0.0, 0.5, 1.0], name='column')

        # Run
        result = sampler._rescale_values(column)

        # Check
        assert (result == expected_result).all().all()
        assert len(data_navigator.call_args_list) == 0
        assert len(modeler.call_args_list) == 0

    @patch('sdv.sampler.Sampler._fill_text_columns', autospec=True)
    @patch('sdv.sampler.Sampler.update_mapping_list', autospec=True)
    @patch('sdv.sampler.Sampler._get_table_meta', autospec=True)
    def test_transform_synthesized_rows_no_pk_no_categorical(
            self, get_table_meta_mock, update_mock, fill_mock):

        """transform_synthesized_rows will update internal state and reverse transform rows."""
        # Setup - Class Instantiation
        data_navigator = MagicMock()
        modeler = MagicMock()
        sampler = Sampler(data_navigator, modeler)

        # Setup - Mock configuration
        table_metadata = {
            'fields': {
                'column_A': {
                    'type': 'number',
                    'subtype': 'integer'
                },
                'column_B': {
                    'name': 'column',
                    'type': 'number'
                }
            },
            'primary_key': None
        }
        table_data = pd.DataFrame(columns=['column_A', 'column_B'])
        test_table = Table(table_data, table_metadata)
        data_navigator.tables = {
            'table': test_table
        }

        data_navigator.ht.reverse_transform_table.return_value = pd.DataFrame({
            'column_A': ['some', 'transformed values'],
            'column_B': ['another', 'transformed column']
        })

        get_table_meta_mock.return_value = {
            'original': 'meta',
            'fields': []
        }

        fill_mock.return_value = pd.DataFrame({
            'column_A': ['filled', 'text_values'],
            'column_B': ['nothing', 'numerical']
        })

        # Setup - Method arguments / expected result
        synthesized_rows = pd.DataFrame({
            'column_A': [1.7, 2.5],
            'column_B': [4.7, 5.1],
            'model_parameters': ['some', 'parameters']
        })
        table_name = 'table'
        num_rows = 2

        expected_result = pd.DataFrame({
            'column_A': ['some', 'transformed values'],
            'column_B': ['another', 'transformed column']
        })

        # Run
        result = sampler.transform_synthesized_rows(synthesized_rows, table_name, num_rows)

        # Check - Result
        assert result.equals(expected_result)

        # Check - Class internal state
        assert sampler.sampled == update_mock.return_value

        # Check - Mock calls
        get_table_meta_mock.assert_called_once_with(sampler, data_navigator.meta, 'table')
        update_mock.assert_called_once_with(sampler, {}, 'table', (None, synthesized_rows))
        fill_mock.assert_called_once_with(
            sampler, synthesized_rows, ['column_A', 'column_B'], 'table')

        data_navigator.ht.reverse_transform_table.assert_called_once_with(
            fill_mock.return_value, get_table_meta_mock.return_value
        )

    @patch('sdv.sampler.Sampler._fill_text_columns', autospec=True)
    @patch('sdv.sampler.Sampler.update_mapping_list', autospec=True)
    @patch('sdv.sampler.Sampler._rescale_values', autospec=True)
    @patch('sdv.sampler.Sampler._get_table_meta', autospec=True)
    def test_transform_synthesized_rows_no_pk_but_categorical(
            self, get_table_meta_mock, rescale_mock, update_mock, fill_mock):

        """transform_synthesized_rows will update internal state and reverse transform rows."""
        # Setup - Class Instantiation
        data_navigator = MagicMock()
        modeler = MagicMock()
        sampler = Sampler(data_navigator, modeler)

        # Setup - Mock configuration
        table_metadata = {
            'fields': {
                'column_A': {
                    'type': 'categorical',
                },
                'column_B': {
                    'name': 'column',
                    'type': 'number'
                }
            },
            'primary_key': None
        }
        table_data = pd.DataFrame(columns=['column_A', 'column_B'])
        test_table = Table(table_data, table_metadata)
        data_navigator.tables = {
            'table': test_table
        }

        data_navigator.ht.reverse_transform_table.return_value = pd.DataFrame({
            'column_A': ['some', 'transformed values'],
            'column_B': ['another', 'transformed column']
        })

        get_table_meta_mock.return_value = {
            'original': 'meta',
            'fields': []
        }

        rescale_mock.side_effect = lambda x: pd.Series([0.1, 0.8], name=x.name)

        fill_mock.return_value = pd.DataFrame({
            'column_A': ['filled', 'text_values'],
            'column_B': ['nothing', 'numerical']
        })

        # Setup - Method arguments / expected result
        synthesized_rows = pd.DataFrame({
            'column_A': [1.7, 2.5],
            'column_B': [4.7, 5.1],
            'model_parameters': ['some', 'parameters']
        })
        table_name = 'table'
        num_rows = 2

        expected_result = pd.DataFrame({
            'column_A': ['some', 'transformed values'],
            'column_B': ['another', 'transformed column']
        })

        # Run
        result = sampler.transform_synthesized_rows(synthesized_rows, table_name, num_rows)

        # Check - Result
        assert result.equals(expected_result)

        # Check - Class internal state
        assert sampler.sampled == update_mock.return_value

        # Check - Mock calls
        get_table_meta_mock.assert_called_once_with(sampler, data_navigator.meta, 'table')
        update_mock.assert_called_once_with(sampler, {}, 'table', (None, synthesized_rows))
        fill_mock.assert_called_once_with(
            sampler, synthesized_rows, ['column_A', 'column_B'], 'table')

        data_navigator.ht.reverse_transform_table.assert_called_once_with(
            fill_mock.return_value, get_table_meta_mock.return_value
        )

    def test_sample_rows_parent_table(self):
        """sample_rows samples new rows for the given table."""
        # Setup
        raw_data = self.modeler.dn.tables['DEMO_CUSTOMERS'].data

        # Run
        result = self.sampler.sample_rows('DEMO_CUSTOMERS', 5)

        # Check
        assert result.shape[0] == 5
        assert (result.columns == raw_data.columns).all()

        # Primary key columns are sampled values
        assert len(result['CUSTOMER_ID'].unique()) != 1

    def test_sample_rows_children_table(self):
        """sample_rows samples new rows for the given table."""
        # Setup
        raw_data = self.modeler.dn.tables['DEMO_ORDERS'].data
        # Sampling parent table.
        self.sampler.sample_rows('DEMO_CUSTOMERS', 5)

        # Run
        result = self.sampler.sample_rows('DEMO_ORDERS', 5)

        # Check
        assert result.shape[0] == 5
        assert (result.columns == raw_data.columns).all()

        # Foreign key columns are all the same
        unique_foreign_keys = result['CUSTOMER_ID'].unique()
        sampled_parent = self.sampler.sampled['DEMO_CUSTOMERS'][0][1]
        assert len(unique_foreign_keys) == 1
        assert unique_foreign_keys[0] in sampled_parent['CUSTOMER_ID'].values

    @patch('sdv.sampler.pd.concat')
    @patch('sdv.sampler.Sampler.reset_indices_tables')
    @patch('sdv.sampler.Sampler._sample_child_rows')
    @patch('sdv.sampler.Sampler.sample_rows')
    def test_sample_all(self, rows_mock, child_mock, reset_mock, concat_mock):
        """Check sample_all and returns some value."""
        # Setup
        data_navigator = MagicMock()
        data_navigator.tables = ['TABLE_A', 'TABLE_B']
        data_navigator.get_parents.side_effect = lambda x: x != 'TABLE_A'
        modeler = MagicMock()
        sampler = Sampler(data_navigator, modeler)

        def fake_dataframe(name, number):
            return pd.DataFrame([{name: 0} for i in range(number)], index=[0]*number)

        rows_mock.side_effect = fake_dataframe
        concat_mock.return_value = 'concatenated_dataframe'

        expected_get_parents_call_list = [(('TABLE_A',), {}), (('TABLE_B',), {})]
        expected_rows_mock_call_list = [(('TABLE_A', 1), {}) for i in range(5)]

        # Run
        result = sampler.sample_all(num_rows=5)

        # Check
        assert data_navigator.get_parents.call_args_list == expected_get_parents_call_list
        assert result == reset_mock.return_value

        assert rows_mock.call_args_list == expected_rows_mock_call_list
        assert child_mock.call_count == 5
        reset_mock.assert_called_once_with({'TABLE_A': 'concatenated_dataframe'})

    def test_unflatten_dict(self):
        """unflatten_dict restructure flatten dicts."""
        # Setup
        data_navigator = MagicMock()
        modeler = MagicMock()
        sampler = Sampler(data_navigator, modeler)
        flat = {
            'first_key__a': 1,
            'first_key__b': 2,
            'second_key__x': 0
        }

        expected_result = {
            'first_key': {
                'a': 1,
                'b': 2
            },
            'second_key': {
                'x': 0
            }
        }

        # Run
        result = sampler._unflatten_dict(flat)

        # Check
        assert result == expected_result
        data_navigator.assert_not_called()
        modeler.assert_not_called()

    def test_unflatten_dict_mixed_array(self):
        """unflatten_dict restructure arrays."""
        # Setup
        data_navigator = MagicMock()
        modeler = MagicMock()
        sampler = Sampler(data_navigator, modeler)
        flat = {
            'first_key__0__0': 1,
            'first_key__0__1': 0,
            'first_key__1__0': 0,
            'first_key__1__1': 1,
            'second_key__0__std': 0.5,
            'second_key__0__mean': 0.5,
            'second_key__1__std': 0.25,
            'second_key__1__mean': 0.25
        }

        expected_result = {
            'first_key': [
                [1, 0],
                [0, 1]
            ],
            'second_key': [
                {
                    'std': 0.5,
                    'mean': 0.5
                },
                {
                    'std': 0.25,
                    'mean': 0.25
                }
            ]
        }

        # Run

        result = sampler._unflatten_dict(flat)

        # Check
        assert result == expected_result
        data_navigator.assert_not_called()
        modeler.assert_not_called()

    def test_unflatten_dict_child_name(self):
        """unflatten_dict will respect the name of child tables."""
        # Setup
        data_navigator = MagicMock()
        data_navigator.get_children.return_value = ['CHILD_TABLE']
        modeler = MagicMock()
        sampler = Sampler(data_navigator, modeler)

        flat = {
            'first_key__a': 1,
            'first_key____CHILD_TABLE__model_param': 0,
            'distribs____CHILD_TABLE__distribs__UNIT_PRICE__std__mean': 0
        }
        table_name = 'TABLE_NAME'
        expected_result = {
            'first_key': {
                'a': 1,
                '__CHILD_TABLE': {
                    'model_param': 0
                }
            },
            'distribs': {
                '__CHILD_TABLE__distribs__UNIT_PRICE__std': {
                    'mean': 0
                }
            }
        }

        # Run
        result = sampler._unflatten_dict(flat, table_name)

        # Check
        assert result == expected_result
        modeler.assert_not_called()
        data_navigator.get_children.assert_called_once_with('TABLE_NAME')

    def test_unflatten_respect_covariance_matrix(self):
        """unflatten_dict restructures the covariance matrix into an square matrix."""
        # Setup
        data_navigator = MagicMock()
        modeler = MagicMock()
        sampler = Sampler(data_navigator, modeler)

        def fake_values(i, j):
            return '{}, {}'.format(i, j)

        expected_result = {
            'covariance': np.array([
                [fake_values(i, j) for j in range(40)]
                for i in range(40)
            ]).tolist()
        }

        flat = {
            'covariance__{}__{}'.format(i, j): fake_values(i, j)
            for i in range(40) for j in range(40)
        }
        table_name = 'TABLE_NAME'

        # Run
        result = sampler._unflatten_dict(flat, table_name)

        # Check
        assert result == expected_result
