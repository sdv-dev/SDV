from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from sdv.data_navigator import DataNavigator
from sdv.modeler import Modeler
from sdv.sampler import Sampler


class TestSampler(TestCase):

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

    @patch('sdv.sampler.Sampler.sample_rows', autospec=True)
    def test_sample_all(self, rows_mock):
        """Check sample_all and returns some value."""
        # Setup
        data_navigator = MagicMock()
        data_navigator.tables = ['TABLE_A', 'TABLE_B']
        data_navigator.get_parents.side_effect = lambda x: x != 'TABLE_A'
        modeler = MagicMock()
        sampler = Sampler(data_navigator, modeler)

        def fake_dataframe(*args, **kwargs):
            kwargs['sampled_data'][args[1]] = 'sampled_data'

        rows_mock.side_effect = fake_dataframe

        expected_get_parents_call_list = [(('TABLE_A',), {}), (('TABLE_B',), {})]
        expected_result = {
            'TABLE_A': 'sampled_data'
        }

        # Run
        result = sampler.sample_all(num_rows=5)

        # Check
        assert result == expected_result

        assert data_navigator.get_parents.call_args_list == expected_get_parents_call_list
        rows_mock.assert_called_once_with(
            sampler, 'TABLE_A', 5, sampled_data={'TABLE_A': 'sampled_data'})

    def test__unflatten_dict(self):
        """unflatten_dict restructure flatten dicts."""
        # Setup
        data_navigator = MagicMock()
        modeler = MagicMock()
        sampler = Sampler(data_navigator, modeler)
        flat = {
            'a__first_key__a': 1,
            'a__first_key__b': 2,
            'b__second_key__x': 0
        }

        expected_result = {
            'a': {
                'first_key': {
                    'a': 1,
                    'b': 2
                },
            },
            'b': {
                'second_key': {
                    'x': 0
                },
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
            'second_key': {
                0: {
                    'std': 0.5,
                    'mean': 0.5
                },
                1: {
                    'std': 0.25,
                    'mean': 0.25
                }
            }
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
        modeler = MagicMock()
        sampler = Sampler(data_navigator, modeler)

        flat = {
            'first_key__a__b': 1,
            'first_key____CHILD_TABLE__model_param': 0,
            'distribs____CHILD_TABLE__distribs__UNIT_PRICE__std__mean': 0
        }
        expected_result = {
            'first_key': {
                'a': {
                    'b': 1
                },
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
        result = sampler._unflatten_dict(flat)

        # Check
        assert result == expected_result
        modeler.assert_not_called()
        data_navigator.assert_not_called()

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

        # Run
        result = sampler._unflatten_dict(flat)

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

    def test__reset_primary_keys_generators(self):
        """_reset_primary_keys deletes all generators and counters."""
        # Setup
        data_navigator = MagicMock(spec=DataNavigator)
        modeler = MagicMock(spec=Modeler)
        sampler = Sampler(data_navigator=data_navigator, modeler=modeler)

        sampler.primary_key = {
            'table': 'generator for table'
        }
        sampler.remaining_primary_key = {
            'table': 'counter for table'
        }

        # Run
        sampler._reset_primary_keys_generators()

        # Check
        assert sampler.primary_key == dict()
        assert sampler.remaining_primary_key == dict()

    @patch('sdv.sampler.exrex.count', autospec=True)
    @patch('sdv.sampler.exrex.generate', autospec=True)
    def test__get_primary_keys_create_generator(self, exrex_gen_mock, exrex_count_mock):
        """If there's a primary key, but no generator, a new one is created and used."""
        # Setup
        data_navigator = MagicMock(spec=DataNavigator)
        data_navigator.get_meta_data.return_value = {
            'primary_key': 'table_pk',
            'fields': {
                'table_pk': {
                    'regex': 'regex for table_pk',
                    'type': 'number',
                    'subtype': 'integer'
                },
            }
        }
        modeler = MagicMock(spec=Modeler)
        sampler = Sampler(data_navigator=data_navigator, modeler=modeler)

        exrex_gen_mock.return_value = (str(x) for x in range(10))
        exrex_count_mock.return_value = 10

        expected_primary_key = 'table_pk'
        expected_primary_key_values = pd.Series(range(5))

        # Run
        result = sampler._get_primary_keys('table', 5)

        # Check
        primary_key, primary_key_values = result
        assert primary_key == expected_primary_key
        primary_key_values.equals(expected_primary_key_values)

        assert sampler.primary_key['table'] == exrex_gen_mock.return_value
        assert sampler.remaining_primary_key['table'] == 5

        data_navigator.get_meta_data.assert_called_once_with('table')
        exrex_count_mock.assert_called_once_with('regex for table_pk')
        exrex_gen_mock.assert_called_once_with('regex for table_pk')

    def test__get_primary_keys_no_pk(self):
        """If no primary key, _get_primary_keys return a duple of None """
        # Setup
        data_navigator = MagicMock(spec=DataNavigator)
        data_navigator.get_meta_data.return_value = {}
        modeler = MagicMock(spec=Modeler)
        sampler = Sampler(data_navigator=data_navigator, modeler=modeler)

        # Run
        result = sampler._get_primary_keys('table', 5)

        # Check
        primary_key, primary_key_values = result
        assert primary_key is None
        assert primary_key_values is None

    def test__get_primary_keys_raises_error(self):
        """_get_primary_keys raises an exception if there aren't enough values."""
        # Setup
        data_navigator = MagicMock(spec=DataNavigator)
        data_navigator.get_meta_data.return_value = {
            'primary_key': 'table_pk',
            'fields': {
                'table_pk': {
                    'regex': 'regex for table_pk',
                    'type': 'number',
                    'subtype': 'integer'
                },
            }
        }
        modeler = MagicMock(spec=Modeler)
        sampler = Sampler(data_navigator=data_navigator, modeler=modeler)
        sampler.primary_key['table'] = 'a generator'
        sampler.remaining_primary_key['table'] = 0

        # Run / Check
        with self.assertRaises(ValueError):
            sampler._get_primary_keys('table', 5)

    @patch('sdv.sampler.Sampler.sample_rows', autospec=True)
    def test_sample_table(self, rows_mock):
        """ """
        # Setup
        data_navigator = MagicMock(spec=DataNavigator)
        data_navigator.tables = {
            'table': MagicMock(**{'data.shape': ('rows', 'columns')})
        }
        modeler = MagicMock(spec=Modeler)
        sampler = Sampler(data_navigator=data_navigator, modeler=modeler)

        rows_mock.return_value = 'samples'

        table_name = 'table'
        reset_primary_keys = False

        expected_result = 'samples'

        # Run
        result = sampler.sample_table(table_name, reset_primary_keys=reset_primary_keys)

        # Check
        assert result == expected_result

        rows_mock.assert_called_once_with(
            sampler, 'table', 'rows', sample_children=False, reset_primary_keys=False)
