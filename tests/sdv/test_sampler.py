from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from sdv.data_navigator import CSVDataLoader
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

    def test__unflatten_dict(self):
        """ """
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

    def test__unflatten_dict_mixed_array(self):
        """unflatten_dict restruicture arrays"""
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

    def test__unflatten_dict_child_name(self):
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

    def test__unflatten_dict_respect_covariance_matrix(self):
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

    def test__unflatten_gaussian_copula(self):
        """_unflatten_gaussian_copula add the distribution, type and fitted kwargs."""
        # Setup
        data_navigator = MagicMock()
        modeler = MagicMock()
        modeler.model_kwargs = {
            'distribution': 'distribution_name'
        }
        sampler = Sampler(data_navigator, modeler)

        model_parameters = {
            'some': 'key',
            'covariance': [
                [1, 0],
                [0, 1]
            ],
            'distribs': {
                0: {
                    'first': 'distribution',
                    'std': 1
                },
                1: {
                    'second': 'distribution',
                    'std': 1
                }
            }
        }
        expected_result = {
            'some': 'key',
            'distribution': 'distribution_name',
            'covariance': [
                [1, 0],
                [0, 1]
            ],
            'distribs': {
                0: {
                    'type': 'distribution_name',
                    'fitted': True,
                    'first': 'distribution',
                    'std': 1
                },
                1: {
                    'type': 'distribution_name',
                    'fitted': True,
                    'second': 'distribution',
                    'std': 1
                }
            }
        }

        # Run
        result = sampler._unflatten_gaussian_copula(model_parameters)

        # Check
        assert result == expected_result

        data_navigator.assert_not_called()
        modeler.assert_not_called()

    def test__unflatten_gaussian_copula_negative_std(self):
        """_unflatten_gaussian_copula will transform negative or 0 std into positive."""
        # Setup
        data_navigator = MagicMock()
        modeler = MagicMock()
        modeler.model_kwargs = {
            'distribution': 'distribution_name'
        }
        sampler = Sampler(data_navigator, modeler)

        model_parameters = {
            'some': 'key',
            'covariance': [
                [1, 0],
                [0, 1]
            ],
            'distribs': {
                0: {
                    'first': 'distribution',
                    'std': 0
                },
                1: {
                    'second': 'distribution',
                    'std': -1
                }
            }
        }
        expected_result = {
            'some': 'key',
            'distribution': 'distribution_name',
            'covariance': [
                [1, 0],
                [0, 1]
            ],
            'distribs': {
                0: {
                    'type': 'distribution_name',
                    'fitted': True,
                    'first': 'distribution',
                    'std': 1
                },
                1: {
                    'type': 'distribution_name',
                    'fitted': True,
                    'second': 'distribution',
                    'std': np.exp(-1)
                }
            }
        }

        # Run
        result = sampler._unflatten_gaussian_copula(model_parameters)

        # Check
        assert result == expected_result

        data_navigator.assert_not_called()
        modeler.assert_not_called()
