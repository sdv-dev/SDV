from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from sdv.data_navigator import CSVDataLoader, DataNavigator, Table
from sdv.modeler import GaussianMultivariate, Modeler
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

    def test__square_matrix(self):
        """_square_matrix transform triagular list of list into square matrix."""
        # Setup
        data_navigator = MagicMock()
        modeler = MagicMock()
        sampler = Sampler(data_navigator, modeler)

        triangular_matrix = [
            [1],
            [1, 1],
            [1, 1, 1]
        ]

        expected_result = [
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1]
        ]

        # Run
        result = sampler._square_matrix(triangular_matrix)

        # Check
        assert result == expected_result

    @patch('sdv.sampler.Sampler._fill_text_columns', autospec=True)
    @patch('sdv.sampler.Sampler.update_mapping_list')
    @patch('sdv.sampler.Sampler._get_table_meta', autospec=True)
    def test_transform_synthesized_rows_no_pk(
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

        data_navigator.ht.transformers = {
            ('table', 'column_A'): None,
            ('table', 'column_B'): None
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
        }, columns=[column[1] for column in data_navigator.ht.transformers])

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
        update_mock.assert_called_once_with({}, 'table', (None, synthesized_rows))
        fill_mock.assert_called_once_with(
            sampler, synthesized_rows, ['column_A', 'column_B'], 'table')

        call_args = data_navigator.ht.reverse_transform_table.call_args_list
        assert len(call_args) == 1
        assert len(call_args[0][0]) == 2
        assert call_args[0][0][0].equals(fill_mock.return_value)
        assert call_args[0][0][1] == get_table_meta_mock.return_value
        assert call_args[0][1] == {}

    def test__prepare_sampled_covariance(self):
        """ """
        # Setup
        data_navigator = MagicMock()
        modeler = MagicMock()
        sampler = Sampler(data_navigator, modeler)

        covariance = [
            [1.0],
            [0.5, 1.0],
            [0.5, 0.5, 1.0]
        ]

        expected_result = np.array([
            [1.0, 0.5, 0.5],
            [0.5, 1.0, 0.5],
            [0.5, 0.5, 1.0]
        ])
        # Run
        result = sampler._prepare_sampled_covariance(covariance)

        # Check
        assert (result == expected_result).all().all()

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
            return pd.DataFrame([{name: 0} for i in range(number)], index=[0] * number)

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

    def test__unflatten_dict_mixed_array(self):
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
                [1],
                [0, 1]
            ],
            'distribs': {
                0: {
                    'first': 'distribution',
                    'std': 0
                },
                1: {
                    'second': 'distribution',
                    'std': 0
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
                [1],
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

    def test__sample_valid_rows_respect_categorical_values(self):
        """_sample_valid_rows will return rows with valid values for categorical columns."""
        # Setup
        data_navigator = MagicMock(spec=DataNavigator)
        modeler = MagicMock(spec=Modeler)
        sampler = Sampler(data_navigator, modeler)

        data = pd.DataFrame(columns=['field_A', 'field_B'])
        modeler.tables = {
            'table_name': data,
        }

        data_navigator.meta = {
            'tables': [
                {
                    'name': 'table_name',
                    'fields': [
                        {
                            'name': 'field_A',
                            'type': 'categorical'
                        },
                        {
                            'name': 'field_B',
                            'type': 'categorical'
                        }
                    ]
                }
            ]
        }

        num_rows = 5
        table_name = 'table_name'
        model = MagicMock(spec=GaussianMultivariate)
        model.fitted = True
        sample_dataframe = pd.DataFrame([
            {'field_A': 0.5, 'field_B': 0.5},
            {'field_A': 0.5, 'field_B': 0.5},
            {'field_A': 0.5, 'field_B': 0.5},
            {'field_A': 0.5, 'field_B': 1.5},  # Invalid field_B
            {'field_A': 1.5, 'field_B': 0.5},  # Invalid field_A
        ])

        model.sample.side_effect = lambda x: sample_dataframe.iloc[:x].copy()

        expected_model_call_args_list = [
            ((5,), {}),
            ((2,), {})
        ]

        expected_result = pd.DataFrame([
            {'field_A': 0.5, 'field_B': 0.5},
            {'field_A': 0.5, 'field_B': 0.5},
            {'field_A': 0.5, 'field_B': 0.5},
            {'field_A': 0.5, 'field_B': 0.5},
            {'field_A': 0.5, 'field_B': 0.5},
        ])

        # Run
        result = sampler._sample_valid_rows(model, num_rows, table_name)

        # Check
        assert result.equals(expected_result)

        modeler.assert_not_called()
        assert len(modeler.method_calls) == 0

        data_navigator.assert_not_called()
        assert len(data_navigator.method_calls) == 0

        assert model.sample.call_args_list == expected_model_call_args_list

    def test__sample_valid_rows_raises_unfitted_model(self):
        """_sample_valid_rows raise an exception for invalid models."""
        # Setup
        data_navigator = MagicMock(spec=DataNavigator)
        modeler = MagicMock(spec=Modeler)
        sampler = Sampler(data_navigator, modeler)

        data_navigator.get_parents.return_value = set()

        num_rows = 5
        table_name = 'table_name'
        model = None

        # Run
        with self.assertRaises(ValueError):
            sampler._sample_valid_rows(model, num_rows, table_name)

        # Check
        modeler.assert_not_called()
        assert len(modeler.method_calls) == 0

        data_navigator.assert_not_called()
        data_navigator.get_parents.assert_called_once_with('table_name')

    def test__get_missing_valid_rows(self):
        """get_missing_valid_rows return an a dataframe and an integer.

        The dataframe contains valid_rows concatenated to synthesized and their index reset.
        The integer is the diference between num_rows and the returned dataframe rows.
        """
        # Setup
        data_navigator = MagicMock(spec=DataNavigator)
        modeler = MagicMock(spec=Modeler)
        sampler = Sampler(data_navigator, modeler)

        synthesized = pd.DataFrame(columns=list('AB'), index=range(3, 5))
        drop_indices = pd.Series(False, index=range(3, 5))
        valid_rows = pd.DataFrame(columns=list('AB'), index=range(2))
        num_rows = 5

        # Run
        result = sampler._get_missing_valid_rows(synthesized, drop_indices, valid_rows, num_rows)
        missing_rows, valid_rows = result

        # Check
        assert missing_rows == 1
        assert valid_rows.equals(pd.DataFrame(columns=list('AB'), index=[0, 1, 2, 3]))

        data_navigator.assert_not_called()
        assert data_navigator.method_calls == []

        modeler.assert_not_called()
        assert modeler.method_calls == []

    def test__get_missing_valid_rows_excess_rows(self):
        """If more rows than required are passed, the result is cut to num_rows."""
        # Setup
        data_navigator = MagicMock(spec=DataNavigator)
        modeler = MagicMock(spec=Modeler)
        sampler = Sampler(data_navigator, modeler)

        synthesized = pd.DataFrame(columns=list('AB'), index=range(3, 7))
        drop_indices = pd.Series(False, index=range(3, 7))
        valid_rows = pd.DataFrame(columns=list('AB'), index=range(2))
        num_rows = 5

        # Run
        result = sampler._get_missing_valid_rows(synthesized, drop_indices, valid_rows, num_rows)
        missing_rows, valid_rows = result

        # Check
        assert missing_rows == 0
        assert valid_rows.equals(pd.DataFrame(columns=list('AB'), index=range(5)))

        data_navigator.assert_not_called()
        assert data_navigator.method_calls == []

        modeler.assert_not_called()
        assert modeler.method_calls == []

    @patch('sdv.sampler.get_qualified_name')
    def test__sample_model(self, qualified_mock):
        """_sample_model sample the number of rows from the given model."""
        # Setup
        data_navigator = MagicMock(spec=DataNavigator)
        modeler = MagicMock(spec=Modeler)

        sampler = Sampler(data_navigator, modeler)
        model = MagicMock()
        values = np.array([
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3]
        ])

        qualified_mock.return_value = 'package.module.full_qualified_name'

        model.sample.return_value = values
        num_rows = 3
        columns = list('ABC')

        expected_result = pd.DataFrame(values, columns=columns)

        # Run
        result = sampler._sample_model(model, num_rows, columns)

        # Check
        assert result.equals(expected_result)

        qualified_mock.assert_called_once_with(model)
        model.sample.assert_called_once_with(3)

    @patch('sdv.sampler.get_qualified_name')
    def test__sample_model_vine(self, qualified_mock):
        """_sample_model sample the number of rows from the given model."""
        # Setup
        data_navigator = MagicMock(spec=DataNavigator)
        modeler = MagicMock(spec=Modeler)

        sampler = Sampler(data_navigator, modeler)
        model = MagicMock()
        values = [
            np.array([1, 1, 1]),
            np.array([2, 2, 2]),
            np.array([3, 3, 3])
        ]

        qualified_mock.return_value = 'copulas.multivariate.vine.VineCopula'

        model.sample.side_effect = values
        num_rows = 3
        columns = list('ABC')

        expected_result = pd.DataFrame(values, columns=columns)

        # Run
        result = sampler._sample_model(model, num_rows, columns)

        # Check
        assert result.equals(expected_result)

        qualified_mock.assert_called_once_with(model)
        assert model.sample.call_args_list == [
            ((3,), ),
            ((3,), ),
            ((3,), )
        ]
