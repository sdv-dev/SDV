from unittest import TestCase
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from sdv.metadata import Metadata
from sdv.sampler import Sampler


class TestSampler(TestCase):

    def test___init__(self):
        """Test create a default instance of Sampler class"""
        # Run
        models = {'test': Mock()}
        sampler = Sampler('test_metadata', models)

        # Asserts
        assert sampler.metadata == 'test_metadata'
        assert sampler.models == models
        assert sampler.primary_key == dict()
        assert sampler.remaining_primary_key == dict()

    def test__square_matrix(self):
        """Test fill zeros a triangular matrix"""
        # Run
        matrix = [[0.1, 0.5], [0.3]]
        result = Sampler._square_matrix(matrix)

        # Asserts
        expected = [[0.1, 0.5], [0.3, 0.0]]
        assert result == expected

    def test__prepare_sampled_covariance(self):
        """Test prepare_sampler_covariante"""
        # Run
        covariance = [[0, 1], [1]]
        result = Sampler(None, None)._prepare_sampled_covariance(covariance)

        # Asserts
        expected = np.array([[1., 1.], [1., 1.0]])
        np.testing.assert_almost_equal(result, expected)

    def test__reset_primary_keys_generators(self):
        """Test reset values"""
        # Setup
        sampler = Mock(spec=Sampler)
        sampler.primary_key = 'something'
        sampler.remaining_primary_key = 'else'

        # Run
        Sampler._reset_primary_keys_generators(sampler)

        # Asserts
        assert sampler.primary_key == dict()
        assert sampler.remaining_primary_key == dict()

    def test__transform_synthesized_rows(self):
        """Test transform synthesized rows"""
        # Setup
        metadata_reverse_transform = pd.DataFrame({'foo': [0, 1], 'bar': [2, 3], 'tar': [4, 5]})

        sampler = Mock(spec=Sampler)
        sampler.metadata = Mock(spec=Metadata)
        sampler.metadata.reverse_transform.return_value = metadata_reverse_transform
        sampler.metadata.get_fields.return_value = {'foo': 'some data', 'tar': 'some data'}

        # Run
        synthesized = pd.DataFrame({'data': [1, 2, 3]})
        result = Sampler._transform_synthesized_rows(sampler, synthesized, 'test')

        # Asserts
        expected = pd.DataFrame({'foo': [0, 1], 'tar': [4, 5]})
        pd.testing.assert_frame_equal(result.sort_index(axis=1), expected.sort_index(axis=1))

    def test__get_primary_keys_none(self):
        """Test returns a tuple of none when a table doesn't have a primary key"""
        # Setup
        sampler = Mock(spec=Sampler)
        sampler.metadata = Mock(spec=Metadata)
        sampler.metadata.get_primary_key.return_value = None

        # Run
        result = Sampler._get_primary_keys(sampler, 'test', 5)

        # Asserts
        assert result == (None, None)

    def test__get_primary_keys_raise_value_error_field_not_id(self):
        """Test a ValueError is raised when generator is None and field type not id."""
        # Setup
        sampler = Mock(spec=Sampler)
        sampler.metadata = Mock(spec=Metadata)
        sampler.metadata.get_primary_key.return_value = 'pk_field'
        sampler.metadata.get_fields.return_value = {'pk_field': {'type': 'not id'}}
        sampler.primary_key = {'test': None}

        # Run
        with pytest.raises(ValueError):
            Sampler._get_primary_keys(sampler, 'test', 5)

    def test__get_primary_keys_raise_value_error_field_not_supported(self):
        """Test a ValueError is raised when a field subtype is not supported."""
        # Setup
        sampler = Mock(spec=Sampler)
        sampler.metadata = Mock(spec=Metadata)
        sampler.metadata.get_primary_key.return_value = 'pk_field'
        sampler.metadata.get_fields.return_value = {'pk_field': {'type': 'id', 'subtype': 'X'}}
        sampler.primary_key = {'test': None}

        # Run
        with pytest.raises(ValueError):
            Sampler._get_primary_keys(sampler, 'test', 5)

    def test__get_primary_keys_raises_not_implemented_error_datetime(self):
        """Test a NotImplementedError is raised when pk field is datetime."""
        # Setup
        sampler = Mock(spec=Sampler)
        sampler.metadata = Mock(spec=Metadata)
        sampler.metadata.get_primary_key.return_value = 'pk_field'
        sampler.metadata.get_fields.return_value = {
            'pk_field': {
                'type': 'id',
                'subtype': 'datetime'
            }
        }
        sampler.primary_key = {'test': None}

        # Run
        with pytest.raises(NotImplementedError):
            Sampler._get_primary_keys(sampler, 'test', 5)

    def test__get_primary_keys_raises_value_error_remaining(self):
        """Test a ValueError is raised when there are not enough uniques values"""
        # Setup
        sampler = Mock(spec=Sampler)
        sampler.metadata = Mock(spec=Metadata)
        sampler.metadata.get_primary_key.return_value = 'pk_field'
        sampler.metadata.get_fields.return_value = {
            'pk_field': {
                'type': 'id',
                'subtype': 'datetime'
            }
        }
        sampler.primary_key = {'test': 'generator'}
        sampler.remaining_primary_key = {'test': 4}

        # Run
        with pytest.raises(ValueError):
            Sampler._get_primary_keys(sampler, 'test', 5)

    def test__key_order(self):
        """Test key order"""
        # Run
        key_value = ['foo__0__1']
        result = Sampler._key_order(key_value)

        # Asserts
        assert result == ['foo', 0, 1]

    def test__unflatten_dict_raises_error_row_index(self):
        """Test unflatten dict raises error row_index"""
        # Setup
        sampler = Mock(autospec=Sampler)
        flat = {
            'foo__0__1': 'some value'
        }

        # Run
        with pytest.raises(ValueError):
            Sampler._unflatten_dict(sampler, flat)

    def test__unflatten_dict_raises_error_column_index(self):
        """Test unflatten dict raises error column_index"""
        # Setup
        sampler = Mock(spec=Sampler)
        flat = {
            'foo__1__0': 'some value'
        }

        # Run
        with pytest.raises(ValueError):
            Sampler._unflatten_dict(sampler, flat)

    def test__unflatten_dict(self):
        """Test unflatten_dict"""
        # Setup
        sampler = Mock(spec=Sampler)
        sampler._key_order = None

        # Run
        flat = {
            'foo__0__foo': 'foo value',
            'bar__0__0': 'bar value',
            'tar': 'tar value'
        }
        result = Sampler._unflatten_dict(sampler, flat)

        # Asserts
        expected = {
            'foo': {0: {'foo': 'foo value'}},
            'bar': [['bar value']],
            'tar': 'tar value',
        }
        assert result == expected

    def test__make_positive_definite(self):
        """Test find the nearest positive-definite matrix"""
        # Setup
        sampler = Mock(spec=Sampler)
        sampler._check_matrix_symmetric_positive_definite.return_value = True

        # Run
        matrix = np.array([[0, 1], [1, 0]])
        result = Sampler._make_positive_definite(sampler, matrix)

        # Asserts
        expected = np.array([[0.5, 0.5], [0.5, 0.5]])
        np.testing.assert_equal(result, expected)

        assert sampler._check_matrix_symmetric_positive_definite.call_count == 1

    def test__make_positive_definite_iterate(self):
        """Test find the nearest positive-definite matrix iterating"""
        # Setup
        sampler = Mock(spec=Sampler)
        sampler._check_matrix_symmetric_positive_definite.side_effect = [False, False, True]

        # Run
        matrix = np.array([[-1, -5], [-3, -7]])
        result = Sampler._make_positive_definite(sampler, matrix)

        # Asserts
        expected = np.array([[0.8, -0.4], [-0.4, 0.2]])
        np.testing.assert_array_almost_equal(result, expected)

        assert sampler._check_matrix_symmetric_positive_definite.call_count == 3

    def test__check_matrix_symmetric_positive_definite_shape_error(self):
        """Test check matrix shape error"""
        # Setup
        sampler = Mock(spec=Sampler)

        # Run
        matrix = np.array([])
        result = Sampler._check_matrix_symmetric_positive_definite(sampler, matrix)

        # Asserts
        assert not result

    def test__check_matrix_symmetric_positive_definite_np_error(self):
        """Test check matrix numpy raise error"""
        # Setup
        sampler = Mock(spec=Sampler)

        # Run
        matrix = np.array([[-1, 0], [0, 0]])
        result = Sampler._check_matrix_symmetric_positive_definite(sampler, matrix)

        # Asserts
        assert not result

    def test__check_matrix_symmetric_positive_definite(self):
        """Test check matrix numpy"""
        # Setup
        sampler = Mock(spec=Sampler)

        # Run
        matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
        result = Sampler._check_matrix_symmetric_positive_definite(sampler, matrix)

        # Asserts
        assert result

    def test__unflatten_gaussian_copula(self):
        """Test unflatte gaussian copula"""
        # Setup
        sampler = Mock(autospec=Sampler)
        sampler._prepare_sampled_covariance.return_value = [[0.4, 0.2], [0.2, 0.0]]

        # Run
        model_parameters = {
            'distribs': {
                'foo': {'std': 0.5}
            },
            'covariance': [[0.4, 0.1], [0.1]],
            'distribution': 'GaussianUnivariate'
        }
        result = Sampler._unflatten_gaussian_copula(sampler, model_parameters)

        # Asserts
        expected = {
            'distribs': {
                'foo': {
                    'fitted': True,
                    'std': 1.6487212707001282,
                    'type': 'GaussianUnivariate'
                }
            },
            'distribution': 'GaussianUnivariate',
            'covariance': [[0.4, 0.2], [0.2, 0.0]]
        }
        assert result == expected

    def test__get_extension(self):
        """Test get extension"""
        # Setup
        sampler = Mock(spec=Sampler)

        # Run
        parent_row = pd.Series([[0, 1], [1, 0]], index=['__foo__field', '__foo__field2'])
        table_name = 'foo'
        result = Sampler._get_extension(sampler, parent_row, table_name)

        # Asserts
        expected = {'field': [0, 1], 'field2': [1, 0]}
        assert result == expected

    def test__get_model(self):
        """Test get model"""
        # Setup
        sampler = Mock(spec=Sampler)
        sampler._unflatten_dict.return_value = {'unflatten': 'dict'}
        sampler._unflatten_gaussian_copula.return_value = {'unflatten': 'gaussian'}
        table_model = Mock()
        table_model.to_dict.return_value = {
            'distribution': 'copulas.multivariate.gaussian.GaussianMultivariate'
        }

        # Run
        extension = {'extension': 'dict'}
        Sampler._get_model(sampler, extension, table_model)

        # Asserts
        sampler._unflatten_dict.assert_called_once_with({'extension': 'dict'})

        expected_unflatten_gaussian_call = {
            'unflatten': 'dict',
            'fitted': True,
            'distribution': 'copulas.multivariate.gaussian.GaussianMultivariate'
        }
        sampler._unflatten_gaussian_copula.assert_called_once_with(
            expected_unflatten_gaussian_call)

        table_model.from_dict.assert_called_once_with({'unflatten': 'gaussian'})

    def test__sample_rows(self):
        """Test sample rows from model"""
        # Setup
        sampler = Mock(spec=Sampler)
        sampler._get_primary_keys.return_value = ('pk', [1, 2, 3, 4])

        model = Mock()
        model.sample.return_value = dict()

        # Run
        result = Sampler._sample_rows(sampler, model, num_rows=5, table_name='test')

        # Asserts
        assert result == {'pk': [1, 2, 3, 4]}

        sampler._get_primary_keys.assert_called_once_with('test', 5)
        model.sample.called_once_with(5)

    def test__sample_children(self):
        """Test sample children"""
        # Setup
        sampler = Mock(spec=Sampler)
        sampler.metadata.get_children.return_value = ['child A', 'child B', 'child C']

        # Run
        sampled = {
            'test': pd.DataFrame({'field': [11, 22, 33]})
        }
        Sampler._sample_children(sampler, 'test', sampled)

        # Asserts
        sampler.metadata.get_children.assert_called_once_with('test')

        expected_calls = [
            ['child A', 'test', pd.Series([11], index=['field'], name=0), sampled],
            ['child A', 'test', pd.Series([22], index=['field'], name=1), sampled],
            ['child A', 'test', pd.Series([33], index=['field'], name=2), sampled],
            ['child B', 'test', pd.Series([11], index=['field'], name=0), sampled],
            ['child B', 'test', pd.Series([22], index=['field'], name=1), sampled],
            ['child B', 'test', pd.Series([33], index=['field'], name=2), sampled],
            ['child C', 'test', pd.Series([11], index=['field'], name=0), sampled],
            ['child C', 'test', pd.Series([22], index=['field'], name=1), sampled],
            ['child C', 'test', pd.Series([33], index=['field'], name=2), sampled],
        ]
        actual_calls = sampler._sample_table.call_args_list
        for result_call, expected_call in zip(actual_calls, expected_calls):
            assert result_call[0][0] == expected_call[0]
            assert result_call[0][1] == expected_call[1]
            assert result_call[0][3] == expected_call[3]
            pd.testing.assert_series_equal(result_call[0][2], expected_call[2])

    def test__sample_table_sampled_empty(self):
        """Test sample table when sampled is still an empty dict."""
        # Setup
        sampler = Mock(spec=Sampler)
        sampler._get_extension.return_value = {'child_rows': 5}

        table_model_mock = Mock()
        sampler.models = {'test': table_model_mock}

        model_mock = Mock()
        sampler._get_model.return_value = model_mock
        sampler._sample_rows.return_value = pd.DataFrame({
            'value': [1, 2, 3, 4, 5]
        })
        sampler.metadata.get_primary_key.return_value = 'id'
        sampler.metadata.get_foreign_key.return_value = 'parent_id'

        # Run
        parent_row = pd.Series({'id': 0})
        sampled = dict()
        Sampler._sample_table(sampler, 'test', 'parent', parent_row, sampled)

        # Asserts
        sampler._get_extension.assert_called_once_with(parent_row, 'test')
        sampler._get_model.assert_called_once_with({'child_rows': 5}, table_model_mock)
        sampler._sample_rows.assert_called_once_with(model_mock, 5, 'test')

        assert sampler._sample_children.call_count == 1
        assert sampler._sample_children.call_args[0][0] == 'test'

        expected_sampled = pd.DataFrame({
            'value': [1, 2, 3, 4, 5],
            'parent_id': [0, 0, 0, 0, 0]
        }, columns=['value', 'parent_id'])
        pd.testing.assert_frame_equal(
            sampler._sample_children.call_args[0][1]['test'],
            expected_sampled
        )

    def test__sample_table_sampled_not_empty(self):
        """Test sample table when sampled previous sampled rows exist."""
        # Setup
        sampler = Mock(spec=Sampler)
        sampler._get_extension.return_value = {'child_rows': 5}

        table_model_mock = Mock()
        sampler.models = {'test': table_model_mock}

        model_mock = Mock()
        sampler._get_model.return_value = model_mock
        sampler._sample_rows.return_value = pd.DataFrame({
            'value': [6, 7, 8, 9, 10]
        })
        sampler.metadata.get_primary_key.return_value = 'id'
        sampler.metadata.get_foreign_key.return_value = 'parent_id'

        # Run
        parent_row = pd.Series({'id': 1})
        sampled = {
            'test': pd.DataFrame({
                'value': [1, 2, 3, 4, 5],
                'parent_id': [0, 0, 0, 0, 0]
            })
        }
        Sampler._sample_table(sampler, 'test', 'parent', parent_row, sampled)

        # Asserts
        sampler._get_extension.assert_called_once_with(parent_row, 'test')
        sampler._get_model.assert_called_once_with({'child_rows': 5}, table_model_mock)
        sampler._sample_rows.assert_called_once_with(model_mock, 5, 'test')

        assert sampler._sample_children.call_count == 1
        assert sampler._sample_children.call_args[0][0] == 'test'

        expected_sampled = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'parent_id': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        })
        pd.testing.assert_frame_equal(
            sampler._sample_children.call_args[0][1]['test'],
            expected_sampled
        )

    def test_sample_all(self):
        """Test sample all regenerating the primary keys"""
        # Setup
        def sample_side_effect(table, num_rows):
            return {table: pd.DataFrame({'foo': range(num_rows)})}

        sampler = Mock(spec=Sampler)
        sampler.metadata.get_tables.return_value = ['table a', 'table b', 'table c']
        sampler.metadata.get_parents.side_effect = [False, True, False]
        sampler.sample.side_effect = sample_side_effect

        # Run
        result = Sampler.sample_all(sampler, num_rows=3, reset_primary_keys=True)

        # Asserts
        assert sampler.metadata.get_parents.call_count == 3
        assert sampler._reset_primary_keys_generators.call_count == 1
        pd.testing.assert_frame_equal(result['table a'], pd.DataFrame({'foo': range(3)}))
        pd.testing.assert_frame_equal(result['table c'], pd.DataFrame({'foo': range(3)}))

    def test_sample_no_sample_children(self):
        """Test sample no sample children"""
        # Setup
        sampler = Mock(spec=Sampler)
        sampler.models = {'test': 'model'}
        sampler.metadata.get_parents.return_value = None

        # Run
        Sampler.sample(sampler, 'test', 5, sample_children=False)
