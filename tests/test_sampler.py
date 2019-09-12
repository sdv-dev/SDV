import unittest
from unittest import TestCase
from unittest.mock import Mock, MagicMock, patch

import numpy as np
import pandas as pd

from rdt.hyper_transformer import HyperTransformer
from sdv.data_navigator import DataNavigator, Table
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

    @patch('sdv.sampler.Sampler._make_positive_definite')
    @patch('sdv.sampler.Sampler._check_matrix_symmetric_positive_definite')
    def test__unflatten_gaussian_copula_not_matrix_symmetric(self, mock_check, mock_make):
        """unflatte with not matrix symmetric"""
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

        mock_check.return_value = False
        mock_make.return_value = np.array([[1, 0], [0, 1]])

        result = sampler._unflatten_gaussian_copula(model_parameters)

        assert result == expected_result

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

    def test__fill_text_columns(self):
        """Fill columns"""
        data_navigator = MagicMock(spec=DataNavigator)
        data_navigator.tables = {
            'DEMO': Table('any_data', {
                'fields': {
                    'id': {
                        'name': 'id',
                        'type': 'id',
                        'regex': '^[0-9]{10}$'
                    },
                    'id2': {
                        'name': 'id2',
                        'type': 'id',
                        'regex': '^[0-9]{10}$',
                        'ref': {
                            'table': 'DEMO_REF',
                            'field': 'DEMO_REF_ID'
                        }
                    },
                    'name': {
                        'name': 'name',
                        'type': 'text',
                        'regex': '^[a-z]{3}$'
                    },
                }
            })
        }
        modeler = MagicMock(spec=Modeler)
        sampler = Sampler(data_navigator, modeler)
        df = pd.DataFrame(
            [['aaa'], ['bbb'], ['ccc']],
            index=['cobra', 'viper', 'sidewinder'],
            columns=['name']
        )
        labels = ['id', 'id2', 'name']

        with patch('sdv.sampler.Sampler.sample_rows', return_value={'DEMO_REF_ID': 69}):
            sampler._fill_text_columns(df, labels, 'DEMO')

            assert 'id' in df.columns
            assert 'id2' in df.columns
            assert not np.array_equal(df.get('name').values, np.array(['aaa', 'bbb', 'ccc']))

    def test__transform_synthesized_rows(self):
        """Reverse transform synthetized data."""

        transformed_table = pd.DataFrame(
            [[1, 2, 1], [4, 5, 4], [7, 8, 7]],
            columns=['foo', 'bar', 'tar']
        )

        hyper_transformer = MagicMock(spec=HyperTransformer)
        ht_instance = hyper_transformer.return_value
        ht_instance.transformers = [
            ('demo', 'foo'),
            ('demo', 'bar'),
            ('demo', 'tar'),
        ]
        ht_instance.reverse_transform_table.return_value = transformed_table

        data_navigator = MagicMock(spec=DataNavigator)
        data_navigator.ht = ht_instance
        data_navigator.get_meta_data.return_value = {
            'fields': {
                'foo': {
                    'name': 'foo',
                    'subtype': 'integer'
                },
                'bar': {
                    'name': 'bar',
                    'subtype': 'integer'
                },
                'tar': {
                    'name': 'tar',
                    'subtype': 'integer'
                }
            }
        }

        modeler = MagicMock(spec=Modeler)
        sampler = Sampler(data_navigator, modeler)
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['foo', 'bar', 'tar'])

        with patch('sdv.sampler.Sampler._fill_text_columns',
                   return_value=transformed_table) as fill_mock:

            result = sampler._transform_synthesized_rows(df, 'demo')
            assert result['tar'].tolist() == [1, 4, 7]
            fill_mock.assert_called_once()
            data_navigator.get_meta_data.assert_called_once()
            ht_instance.reverse_transform_table.assert_called_once()

    @patch('sdv.sampler.Sampler._setdefault')
    def test__unflatten_dict_raise_value_error_row_index(self, setdefault_mock):
        """Raises ValueError by row_index"""
        data_navigator = MagicMock(spec=DataNavigator)
        modeler = MagicMock(spec=Modeler)
        sampler = Sampler(data_navigator=data_navigator, modeler=modeler)

        setdefault_mock.return_value = [1, 2, 3, 4, 5]

        flat = {
            'foo__1__1': 'foo'
        }

        with self.assertRaises(ValueError):
            sampler._unflatten_dict(flat)

    @patch('sdv.sampler.Sampler._setdefault')
    def test__unflatten_dict_raise_value_error_column_index(self, setdefault_mock):
        """Raises ValueError by column_index"""
        data_navigator = MagicMock(spec=DataNavigator)
        modeler = MagicMock(spec=Modeler)
        sampler = Sampler(data_navigator=data_navigator, modeler=modeler)

        setdefault_mock.return_value = [[1, 2, 3, 4]]

        flat = {
            'foo__1__1': 'foo'
        }

        with self.assertRaises(ValueError):
            sampler._unflatten_dict(flat)

    def test__unflatten_dict_alrady_unflatted(self):
        """Already unflatted dict."""
        data_navigator = MagicMock(spec=DataNavigator)
        modeler = MagicMock(spec=Modeler)
        sampler = Sampler(data_navigator=data_navigator, modeler=modeler)

        flat = {
            'foo': 'bar'
        }

        result = sampler._unflatten_dict(flat)

        assert result == flat

    @patch('sdv.sampler.Sampler._check_matrix_symmetric_positive_definite')
    def test__make_positive_definite_no_iterate(self, check_mock):
        """Make positive when check_matrix returns True without iterate"""
        data_navigator = MagicMock(spec=DataNavigator)
        modeler = MagicMock(spec=Modeler)
        sampler = Sampler(data_navigator=data_navigator, modeler=modeler)

        expect = np.array([[1.15578924, 1.52151675, 1.88724426],
                           [1.52151675, 2.00297177, 2.4844268 ],
                           [1.88724426, 2.4844268 , 3.08160934]])

        matrix = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

        result = sampler._make_positive_definite(matrix)

        assert np.array_equal(np.around(result, decimals=8), np.around(expect, decimals=8))
        check_mock.assert_called_once()

    @patch('sdv.sampler.Sampler._check_matrix_symmetric_positive_definite')
    def test__make_positive_definite_iterate(self, check_mock):
        """Make positive when check_matrix returns True with iterations"""
        data_navigator = MagicMock(spec=DataNavigator)
        modeler = MagicMock(spec=Modeler)
        sampler = Sampler(data_navigator=data_navigator, modeler=modeler)

        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        check_mock.side_effect = [False, False, True]

        result = sampler._make_positive_definite(matrix)

        assert check_mock.call_count == 3

    def test__check_matrix_symmetric_positive_definite(self):
        """Check matrix symmetric positive return false"""
        data_navigator = MagicMock(spec=DataNavigator)
        modeler = MagicMock(spec=Modeler)
        sampler = Sampler(data_navigator=data_navigator, modeler=modeler)

        matrix = np.array([[-1, -2], [-4, -69]])

        with patch('numpy.linalg.cholesky') as error_mock:
            error_mock.side_effect = np.linalg.LinAlgError
            result = sampler._check_matrix_symmetric_positive_definite(matrix)
            error_mock.assert_called_once_with(matrix)
            assert result == False

    @patch('numpy.linalg.LinAlgError')
    def test__check_matrix_symmetric_positive_definite_error(self, error_mock):
        """Check matrix symmetric positive return false raise error"""
        data_navigator = MagicMock(spec=DataNavigator)
        modeler = MagicMock(spec=Modeler)
        sampler = Sampler(data_navigator=data_navigator, modeler=modeler)

        matrix = np.array([1, 1])

        result = sampler._check_matrix_symmetric_positive_definite(matrix)
        error_mock.call_count == 0
        assert result == False

    def test__get_extension(self):
        """Retrieve the generated parent row extension"""
        data_navigator = MagicMock(spec=DataNavigator)
        modeler = MagicMock(spec=Modeler)
        sampler = Sampler(data_navigator=data_navigator, modeler=modeler)

        parent_row = pd.DataFrame([[1, 1], [1, 1]], columns=['__demo__foo', '__demo__bar'])
        table_name = 'demo'
        parent_name = 'parent'

        expect = {'foo': {0: 1, 1: 1}, 'bar': {0: 1, 1: 1}}
        # import ipdb; ipdb.set_trace()
        result = sampler._get_extension(parent_row, table_name, parent_name)

        assert result == expect

    def test__get_model(self):
        """Retrieve the model with parameters"""
        pass


if __name__ == '__main__':
    unittest.main()
