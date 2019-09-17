import unittest
from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd

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

    def test_sample_all_with_reset_primary_key(self):
        """Check sample_all with reset_primary_keys True"""

        # Setup
        reset_primary_keys_generators_mock = Mock()

        dn_mock = Mock()
        dn_mock.tables = {
            'DEMO': Table(pd.DataFrame(), {'some': 'meta'})
        }
        dn_mock.get_parents.return_value = True

        # Run
        sampler_mock = Mock()
        sampler_mock._reset_primary_keys_generators = reset_primary_keys_generators_mock
        sampler_mock.dn = dn_mock

        Sampler.sample_all(sampler_mock, reset_primary_keys=True)

        # Asserts
        reset_primary_keys_generators_mock.assert_called_once_with()

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

    def test__unflatten_gaussian_copula_not_matrix_symmetric(self):
        """unflatte with not matrix symmetric"""

        # Setup
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

        modeler_mock = Mock()
        modeler_mock.model_kwargs = {
            'distribution': 'distribution_name'
        }

        prepare_mock = Mock()
        prepare_mock.return_value = [[1], [0, 1]]

        check_mock = Mock()
        check_mock.return_value = False

        make_mock = Mock()
        make_mock.return_value = np.array([[1, 0], [0, 1]])

        # Run
        sampler_mock = Mock()
        sampler_mock.modeler = modeler_mock
        sampler_mock._prepare_sampled_covariance = prepare_mock
        sampler_mock._check_matrix_symmetric_positive_definite = check_mock
        sampler_mock._make_positive_definite = make_mock

        result = Sampler._unflatten_gaussian_copula(sampler_mock, model_parameters)

        # Asserts
        assert result['covariance'] == [[1, 0], [0, 1]]
        prepare_mock.assert_called_once_with([[1], [0, 1]])
        check_mock.assert_called_once_with([[1], [0, 1]])
        make_mock.assert_called_once_with([[1], [0, 1]])

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

        # Setup
        data_navigator_mock = Mock()
        data_navigator_mock.tables = {
            'DEMO': Table(
                pd.DataFrame(),
                {
                    'fields': {
                        'a_field': {
                            'name': 'a_field',
                            'type': 'id',
                            'ref': {
                                'table': 'table_ref',
                                'field': 'table_ref_id'
                            }
                        },
                        'b_field': {
                            'name': 'b_field',
                            'type': 'id',
                            'regex': '^[0-9]{10}$'
                        },
                        'c_field': {
                            'name': 'c_field',
                            'type': 'text',
                            'regex': '^[a-z]{10}$'
                        }
                    }
                }
            )
        }

        sample_rows_mock = Mock()
        sample_rows_mock.return_value = {'table_ref_id': {'name': 'table_ref_id'}}

        # Run
        sampler_mock = Mock()
        sampler_mock.dn = data_navigator_mock
        sampler_mock.sample_rows = sample_rows_mock

        row = pd.DataFrame({
            'c_field': ['foo', 'bar', 'tar']
        })
        labels = ['a_field', 'b_field', 'c_field']
        table_name = 'DEMO'

        Sampler._fill_text_columns(sampler_mock, row, labels, table_name)

        # Asserts
        sample_rows_mock.assert_called_once_with('table_ref', 1)

    def test__transform_synthesized_rows(self):
        """Reverse transform synthetized data."""

        # Setup
        ht_mock = Mock()
        ht_mock.transformers = ['foo', 'bar']
        ht_mock.reverse_transform_table.return_value = pd.DataFrame({
            'foo': [1, 2, 3],
            'bar': ['aaa', 'bbb', 'ccc']
        })

        dn_mock = Mock()
        dn_mock.ht = ht_mock
        dn_mock.get_meta_data.return_value = {
            'fields': {
                'foo': {
                    'subtype': 'integer'
                },
                'bar': {
                    'subtype': 'text'
                },
            }
        }

        fill_text_mock = Mock()
        fill_text_mock.return_value = pd.DataFrame({
            'foo': [1, 2, 3],
            'bar': ['aaa', 'bbb', 'ccc']
        })

        # Run
        sampler_mock = Mock()
        sampler_mock.dn = dn_mock
        sampler_mock._fill_text_columns = fill_text_mock

        table_name = 'DEMO'
        synthesized = pd.DataFrame(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            columns=['foo', 'bar', 'tar']
        )

        Sampler._transform_synthesized_rows(sampler_mock, synthesized, table_name)

        # Asserts
        exp_called_synthesized = pd.DataFrame({
            'foo': [1, 2, 3],
            'bar': ['aaa', 'bbb', 'ccc'],
            'tar': [3, 6, 9]
        })
        exp_called_labels = ['foo', 'bar']

        exp_called_reverse_meta = {
            'fields': [
                {'subtype': 'integer', 'name': 'foo'},
                {'subtype': 'text', 'name': 'bar'}
            ],
            'name': 'DEMO'
        }

        dn_mock.get_meta_data.assert_called_once_with('DEMO')

        fill_text_args, fill_text_kwargs = fill_text_mock.call_args
        fill_text_data_frame, fill_text_labels, fill_text_table_name = fill_text_args

        fill_text_data_frame.sort_index(axis=1, inplace=True)
        exp_called_synthesized.sort_index(axis=1, inplace=True)

        assert fill_text_mock.call_count == 1
        assert sorted(fill_text_labels) == sorted(exp_called_labels)
        assert fill_text_table_name == 'DEMO'

        pd.testing.assert_frame_equal(
            fill_text_data_frame,
            exp_called_synthesized
        )

        rt_args, rt_kwargs = ht_mock.reverse_transform_table.call_args
        rt_arg_text_filled, rt_arg_meta = rt_args

        pd.testing.assert_frame_equal(rt_arg_text_filled, pd.DataFrame(index=[0, 1, 2]))
        assert rt_arg_meta == exp_called_reverse_meta

    def test__unflatten_dict_raise_value_error_row_index(self):
        """Raises ValueError by row_index"""

        # Setup
        setdefault_mock = Mock()
        setdefault_mock.return_value = [1, 2, 3, 4, 5]

        # Run and assert
        sampler = Mock()
        sampler._setdefault = setdefault_mock

        flat = {
            'foo__1__1': 'foo'
        }

        with self.assertRaises(ValueError):
            Sampler._unflatten_dict(sampler, flat)

    def test__unflatten_dict_raise_value_error_column_index(self):
        """Raises ValueError by column_index"""

        # Setup
        setdefault_mock = Mock()
        setdefault_mock.return_value = [[1, 2, 3, 4]]

        # Run and assert
        sampler = Mock()
        sampler._setdefault = setdefault_mock

        flat = {
            'foo__1__1': 'foo'
        }

        with self.assertRaises(ValueError):
            Sampler._unflatten_dict(sampler, flat)

    def test__unflatten_dict_alrady_unflatted(self):
        """Already unflatted dict."""

        # Setup

        # Run
        sampler = Mock()

        flat = {
            'foo': 'bar'
        }

        result = Sampler._unflatten_dict(sampler, flat)

        # Asserts
        exp_dict = {
            'foo': 'bar'
        }

        assert result == exp_dict

    @patch('sdv.sampler.Sampler._check_matrix_symmetric_positive_definite')
    def test__make_positive_definite_no_iterate(self, check_mock):
        """Make positive when check_matrix returns True without iterate"""

        # Setup
        check_matrix_mock = Mock()
        check_matrix_mock.return_value = True

        # Run
        sampler_mock = Mock()
        sampler_mock._check_matrix_symmetric_positive_definite = check_matrix_mock

        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        Sampler._make_positive_definite(sampler_mock, matrix)

        # Asserts
        assert check_matrix_mock.call_count == 1

    def test__make_positive_definite_iterate(self):
        """Make positive when check_matrix returns True with iterations"""

        # Setup
        check_matrix_mock = Mock()
        check_matrix_mock.side_effect = [False, False, True]

        # Run
        sampler_mock = Mock()
        sampler_mock._check_matrix_symmetric_positive_definite = check_matrix_mock

        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        Sampler._make_positive_definite(sampler_mock, matrix)

        # Asserts
        assert check_matrix_mock.call_count == 3

    def test__check_matrix_symmetric_positive_definite(self):
        """Check matrix symmetric positive return false"""

        # Setup

        # Run
        sampler_mock = Mock()

        matrix = np.array([-4, -69])

        result = Sampler._check_matrix_symmetric_positive_definite(sampler_mock, matrix)

        # Asserts

        assert result is False

    def test__check_matrix_symmetric_positive_definite_error(self):
        """Check matrix symmetric positive return false raise error"""

        # Setup

        # Run
        sampler_mock = Mock()

        matrix = np.array([[1, 1], [1, 1]])

        result = Sampler._check_matrix_symmetric_positive_definite(sampler_mock, matrix)

        # Asserts

        assert result is False

    def test__get_extension(self):
        """Retrieve the generated parent row extension"""

        # Setup

        # Run
        sampler_mock = Mock()

        parent_row = pd.Series([[1, 1], [1, 1]], ['__demo__foo', '__demo__bar'])
        table_name = 'demo'
        parent_name = 'parent'

        result = Sampler._get_extension(sampler_mock, parent_row, table_name, parent_name)

        # Asserts
        expect = {'foo': [1, 1], 'bar': [1, 1]}

        assert result == expect

    @patch('sdv.sampler.get_qualified_name')
    def test__get_model(self, qualified_name):
        """Retrieve the model with parameters"""

        # Setup
        unflatten_dict_mock = Mock()
        unflatten_dict_mock.return_value = dict()

        qualified_name.return_value = 'copulas.multivariate.gaussian.GaussianMultivariate'

        unflatten_gaussian_mock = Mock()
        unflatten_gaussian_mock.return_value = None

        model_mock = Mock()
        model_mock.from_dict.return_value = None

        modeler_mock = Mock()
        modeler_mock.model = model_mock

        # Run
        sampler_mock = Mock()
        sampler_mock._unflatten_dict = unflatten_dict_mock
        sampler_mock.modeler = modeler_mock
        sampler_mock._unflatten_gaussian_copula = unflatten_gaussian_mock

        Sampler._get_model(sampler_mock, None)

        # Asserts
        exp_unflatten_gaussian_called = {
            'fitted': True,
            'type': 'copulas.multivariate.gaussian.GaussianMultivariate'
        }

        qualified_name.assert_called_once_with(modeler_mock.model)
        unflatten_dict_mock.assert_called_once_with(None)
        unflatten_gaussian_mock.assert_called_once_with(exp_unflatten_gaussian_called)
        model_mock.from_dict.assert_called_once_with(None)

    def test_sample_rows_sample_children(self):
        """sample_rows with sample_children True"""

        # Setup
        reset_pk_generators_mock = Mock()

        sample_valid_rows_mock = Mock()
        sample_valid_rows_mock.return_value = {}

        get_pk_mock = Mock()
        get_pk_mock.return_value = None

        transform_mock = Mock()

        modeler_mock = Mock()
        modeler_mock.models = {
            'DEMO': {}
        }

        dn_mock = Mock()
        dn_mock.get_parents.return_value = {}
        dn_mock.foreign_keys = {}

        # Run
        sampler_mock = Mock()
        sampler_mock._reset_primary_keys_generators = reset_pk_generators_mock
        sampler_mock._sample_valid_rows = sample_valid_rows_mock
        sampler_mock._get_primary_keys = get_pk_mock
        sampler_mock._transform_synthesized_rows = transform_mock
        sampler_mock.modeler = modeler_mock
        sampler_mock.dn = dn_mock

        table_name = 'DEMO'
        num_rows = 5

        Sampler.sample_rows(sampler_mock, table_name, num_rows, reset_primary_keys=True)

        # Asserts
        reset_pk_generators_mock.assert_called_once_with()
        sample_valid_rows_mock.assert_called_once_with({}, 5, 'DEMO')

    def test_sample_rows_no_sample_children(self):
        """sample_rows with sample_children True"""

        # Setup
        reset_pk_generators_mock = Mock()

        sample_valid_rows_mock = Mock()
        sample_valid_rows_mock.return_value = {}

        get_pk_mock = Mock()
        get_pk_mock.return_value = None, ['foo']

        transform_mock = Mock()

        modeler_mock = Mock()
        modeler_mock.models = {
            'DEMO': {}
        }

        dn_mock = Mock()
        dn_mock.get_parents.return_value = {'foo': 'bar'}
        dn_mock.foreign_keys = {
            ('DEMO', 'foo'): (None, 'tar')
        }

        # Run
        sampler_mock = Mock()
        sampler_mock._reset_primary_keys_generators = reset_pk_generators_mock
        sampler_mock._sample_valid_rows = sample_valid_rows_mock
        sampler_mock._get_primary_keys = get_pk_mock
        sampler_mock._transform_synthesized_rows = transform_mock
        sampler_mock.modeler = modeler_mock
        sampler_mock.dn = dn_mock

        table_name = 'DEMO'
        num_rows = 5

        Sampler.sample_rows(sampler_mock, table_name, num_rows, sample_children=False)

        # Asserts
        transform_mock.assert_called_once_with({'tar': 'foo'}, 'DEMO')

    def test__sample_without_previous(self):
        """Check _sample without previous"""

        # Setup
        get_extension_mock = Mock()
        get_extension_mock.return_value = {'child_rows': 0.999}

        get_model_mock = Mock()
        get_model_mock.return_value = None

        sample_valid_rows_mock = Mock()
        sample_valid_rows_mock.return_value = {}

        sample_children_mock = Mock()

        dn_mock = Mock()
        dn_mock.foreign_keys = {
            ('DEMO', 'p_name'): ('parent_id', 'foreign_key')
        }

        # Run
        sampler_mock = Mock()
        sampler_mock._get_extension = get_extension_mock
        sampler_mock._get_model = get_model_mock
        sampler_mock._sample_valid_rows = sample_valid_rows_mock
        sampler_mock._sample_children = sample_children_mock
        sampler_mock.dn = dn_mock

        table_name = 'DEMO'
        parent_name = 'p_name'
        parent_row = {'parent_id': 'foo'}
        sampled = {}

        Sampler._sample(sampler_mock, table_name, parent_name, parent_row, sampled)

        # Asserts
        get_extension_mock.assert_called_once_with({'parent_id': 'foo'}, 'DEMO', 'p_name')
        get_model_mock.assert_called_once_with({'child_rows': 0.999})
        sample_valid_rows_mock.assert_called_once_with(None, 1, 'DEMO')
        sample_children_mock.assert_called_once_with('DEMO', {'DEMO': {'foreign_key': 'foo'}})

    def test__sample_with_previous(self):
        """Check _sample with previous"""

        # Setup
        get_extension_mock = Mock()
        get_extension_mock.return_value = {'child_rows': 0.999}

        get_model_mock = Mock()
        get_model_mock.return_value = None

        sample_valid_rows_mock = Mock()
        sample_valid_rows_mock.return_value = pd.DataFrame({'foo': [0, 1]})

        sample_children_mock = Mock()

        dn_mock = Mock()
        dn_mock.foreign_keys = {
            ('DEMO', 'p_name'): ('parent_id', 'foreign_key')
        }

        # Run
        sampler_mock = Mock()
        sampler_mock._get_extension = get_extension_mock
        sampler_mock._get_model = get_model_mock
        sampler_mock._sample_valid_rows = sample_valid_rows_mock
        sampler_mock._sample_children = sample_children_mock
        sampler_mock.dn = dn_mock

        table_name = 'DEMO'
        parent_name = 'p_name'
        parent_row = {'parent_id': 'foo'}
        sampled = {'DEMO': pd.DataFrame({
            'bar': [1, 2]
        })}

        Sampler._sample(sampler_mock, table_name, parent_name, parent_row, sampled)

        # Asserts
        exp_dataframe_sampled = pd.DataFrame({
            'bar': [1, 2, np.NaN, np.NaN],
            'foo': [np.NaN, np.NaN, 0, 1],
            'foreign_key': [np.NaN, np.NaN, 'foo', 'foo']
        })
        args_sample_children, kwargs_sample_children = sample_children_mock.call_args
        exp_arg_table_name, exp_arg_sampled = args_sample_children

        get_extension_mock.assert_called_once_with({'parent_id': 'foo'}, 'DEMO', 'p_name')
        get_model_mock.assert_called_once_with({'child_rows': 0.999})
        sample_valid_rows_mock.assert_called_once_with(None, 1, 'DEMO')

        assert exp_arg_table_name == 'DEMO'

        pd.testing.assert_frame_equal(exp_arg_sampled['DEMO'], exp_dataframe_sampled)

    def test__sample_children(self):
        """Sample children"""

        # Setup
        dn_mock = Mock()
        dn_mock.get_children.return_value = ['aaa', 'bbb', 'ccc']

        sample_mock = Mock()

        # Run
        sampler_mock = Mock()
        sampler_mock.dn = dn_mock
        sampler_mock._sample = sample_mock

        table_name = 'DEMO'
        sampled = {
            'DEMO': pd.DataFrame({
                'foo': [0, 1]
            })
        }

        Sampler._sample_children(sampler_mock, table_name, sampled)

        # Asserts
        exp_sampled = {
            'DEMO': pd.DataFrame({
                'foo': [0, 1]
            })
        }

        exp_sample_arguments = [
            ('aaa', 'DEMO', pd.Series({'foo': 0}, name=0), exp_sampled),
            ('aaa', 'DEMO', pd.Series({'foo': 1}, name=1), exp_sampled),
            ('bbb', 'DEMO', pd.Series({'foo': 0}, name=0), exp_sampled),
            ('bbb', 'DEMO', pd.Series({'foo': 1}, name=1), exp_sampled),
            ('ccc', 'DEMO', pd.Series({'foo': 0}, name=0), exp_sampled),
            ('ccc', 'DEMO', pd.Series({'foo': 1}, name=1), exp_sampled)
        ]

        dn_mock.get_children.assert_called_once_with('DEMO')

        assert sample_mock.call_count == 6

        for called, expected in zip(sample_mock.call_args_list, exp_sample_arguments):
            assert called[0][0] == expected[0]
            assert called[0][1] == expected[1]
            pd.testing.assert_series_equal(called[0][2], expected[2])
            pd.testing.assert_frame_equal(called[0][3]['DEMO'], expected[3]['DEMO'])


if __name__ == '__main__':
    unittest.main()
