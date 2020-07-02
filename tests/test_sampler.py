from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from sdv.metadata import Metadata
from sdv.models.base import SDVModel
from sdv.sampler import Sampler


class TestSampler:

    def test___init__(self):
        """Test create a default instance of Sampler class"""
        # Run
        models = {'test': Mock()}
        sampler = Sampler(
            'test_metadata',
            models,
            SDVModel,
            {'model': 'kwargs'},
            {'table': 'sizes'}
        )

        # Asserts
        assert sampler.metadata == 'test_metadata'
        assert sampler.models == models
        assert sampler.primary_key == dict()
        assert sampler.remaining_primary_key == dict()
        assert sampler.model == SDVModel
        assert sampler.model_kwargs == {'model': 'kwargs'}
        assert sampler.table_sizes == {'table': 'sizes'}

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

    def test__finalize(self):
        """Test finalize"""
        # Setup
        sampler = Mock(spec=Sampler)
        sampler.metadata = Mock(spec=Metadata)
        sampler.metadata.get_parents.return_value = ['b', 'c']

        sampler.metadata.reverse_transform.side_effect = lambda x, y: y

        sampler.metadata.get_fields.return_value = {
            'a': 'some data',
            'b': 'some data',  # fk
            'c': 'some data'   # fk
        }

        sampler._find_parent_ids.return_value = [4, 5]
        sampler.metadata.get_foreign_key.side_effect = [
            'b',
            'c',
        ]

        # Run
        sampled_data = {
            'test': pd.DataFrame({
                'a': [0, 1],  # actual data
                'b': [2, 3],  # existing fk key
                'z': [6, 7]   # not used
            })
        }
        result = Sampler._finalize(sampler, sampled_data)

        # Asserts
        assert isinstance(result, dict)
        expected = pd.DataFrame({
            'a': [0, 1],
            'b': [2, 3],
            'c': [4, 5],
        })
        pd.testing.assert_frame_equal(
            result['test'].sort_index(axis=1),
            expected.sort_index(axis=1)
        )
        assert sampler._find_parent_ids.call_count == 1

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

    def test__extract_parameters(self):
        """Test extract parameters"""
        # Setup
        sampler = Mock(spec=Sampler)

        # Run
        parent_row = pd.Series([[0, 1], [1, 0]], index=['__foo__field', '__foo__field2'])
        table_name = 'foo'
        result = Sampler._extract_parameters(sampler, parent_row, table_name)

        # Asserts
        expected = {'field': [0, 1], 'field2': [1, 0]}
        assert result == expected

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
        sampled_data = {
            'test': pd.DataFrame({'field': [11, 22, 33]})
        }
        Sampler._sample_children(sampler, 'test', sampled_data)

        # Asserts
        sampler.metadata.get_children.assert_called_once_with('test')

        expected_calls = [
            ['child A', 'test', pd.Series([11], index=['field'], name=0), sampled_data],
            ['child A', 'test', pd.Series([22], index=['field'], name=1), sampled_data],
            ['child A', 'test', pd.Series([33], index=['field'], name=2), sampled_data],
            ['child B', 'test', pd.Series([11], index=['field'], name=0), sampled_data],
            ['child B', 'test', pd.Series([22], index=['field'], name=1), sampled_data],
            ['child B', 'test', pd.Series([33], index=['field'], name=2), sampled_data],
            ['child C', 'test', pd.Series([11], index=['field'], name=0), sampled_data],
            ['child C', 'test', pd.Series([22], index=['field'], name=1), sampled_data],
            ['child C', 'test', pd.Series([33], index=['field'], name=2), sampled_data],
        ]
        actual_calls = sampler._sample_child_rows.call_args_list
        for result_call, expected_call in zip(actual_calls, expected_calls):
            assert result_call[0][0] == expected_call[0]
            assert result_call[0][1] == expected_call[1]
            assert result_call[0][3] == expected_call[3]
            pd.testing.assert_series_equal(result_call[0][2], expected_call[2])

    def test__sample_child_rows_sampled_empty(self):
        """Test sample table when sampled is still an empty dict."""
        # Setup
        model = Mock(spec=SDVModel)
        model.return_value = model

        sampler = Mock(spec=Sampler)
        sampler.model = model
        sampler.model_kwargs = dict()

        sampler._extract_parameters.return_value = {'child_rows': 5}

        table_model_mock = Mock()
        sampler.models = {'test': table_model_mock}

        sampler._sample_rows.return_value = pd.DataFrame({
            'value': [1, 2, 3, 4, 5]
        })
        sampler.metadata.get_primary_key.return_value = 'id'
        sampler.metadata.get_foreign_key.return_value = 'parent_id'

        # Run
        parent_row = pd.Series({'id': 0})
        sampled = dict()
        Sampler._sample_child_rows(sampler, 'test', 'parent', parent_row, sampled)

        # Asserts
        sampler._extract_parameters.assert_called_once_with(parent_row, 'test')
        sampler._sample_rows.assert_called_once_with(model, 5, 'test')

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

    def test__sample_child_rows_sampled_not_empty(self):
        """Test sample table when sampled previous sampled rows exist."""
        # Setup
        model = Mock(spec=SDVModel)
        model.return_value = model

        sampler = Mock(spec=Sampler)
        sampler.model = model
        sampler.model_kwargs = dict()
        sampler._extract_parameters.return_value = {'child_rows': 5}

        table_model_mock = Mock()
        sampler.models = {'test': table_model_mock}

        # model_mock = Mock()
        # sampler._get_model.return_value = model_mock
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
        Sampler._sample_child_rows(sampler, 'test', 'parent', parent_row, sampled)

        # Asserts
        sampler._extract_parameters.assert_called_once_with(parent_row, 'test')
        sampler._sample_rows.assert_called_once_with(model, 5, 'test')

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

    @patch('sdv.sampler.np.random.choice')
    def test__find_parent_id_all_0(self, choice_mock):
        """If all likelihoods are 0, use num_rows."""
        likelihoods = pd.Series([0, 0, 0, 0])
        num_rows = pd.Series([1, 2, 3, 4])

        Sampler._find_parent_id(likelihoods, num_rows)

        expected_weights = np.array([1 / 10, 2 / 10, 3 / 10, 4 / 10])

        assert choice_mock.call_count == 1
        assert list(choice_mock.call_args[0][0]) == list(likelihoods.index)
        np.testing.assert_array_equal(choice_mock.call_args[1]['p'], expected_weights)

    @patch('sdv.sampler.np.random.choice')
    def test__find_parent_id_all_singlar_matrix(self, choice_mock):
        """If all likelihoods got singular matrix, use num_rows."""
        likelihoods = pd.Series([None, None, None, None])
        num_rows = pd.Series([1, 2, 3, 4])

        Sampler._find_parent_id(likelihoods, num_rows)

        expected_weights = np.array([1 / 10, 2 / 10, 3 / 10, 4 / 10])

        assert choice_mock.call_count == 1
        assert list(choice_mock.call_args[0][0]) == list(likelihoods.index)
        np.testing.assert_array_equal(choice_mock.call_args[1]['p'], expected_weights)

    @patch('sdv.sampler.np.random.choice')
    def test__find_parent_id_all_0_or_singlar_matrix(self, choice_mock):
        """If likehoods are either 0 or NaN, fill the gaps with num_rows."""
        likelihoods = pd.Series([0, None, 0, None])
        num_rows = pd.Series([1, 2, 3, 4])

        Sampler._find_parent_id(likelihoods, num_rows)

        expected_weights = np.array([0, 2 / 6, 0, 4 / 6])

        assert choice_mock.call_count == 1
        assert list(choice_mock.call_args[0][0]) == list(likelihoods.index)
        np.testing.assert_array_equal(choice_mock.call_args[1]['p'], expected_weights)

    @patch('sdv.sampler.np.random.choice')
    def test__find_parent_id_some_good(self, choice_mock):
        """If some likehoods are good, fill the gaps with num_rows."""
        likelihoods = pd.Series([0.5, None, 1.5, None])
        num_rows = pd.Series([1, 2, 3, 4])

        Sampler._find_parent_id(likelihoods, num_rows)

        expected_weights = np.array([0.5 / 4, 1 / 4, 1.5 / 4, 1 / 4])

        assert choice_mock.call_count == 1
        assert list(choice_mock.call_args[0][0]) == list(likelihoods.index)
        np.testing.assert_array_equal(choice_mock.call_args[1]['p'], expected_weights)

    @patch('sdv.sampler.np.random.choice')
    def test__find_parent_id_all_good(self, choice_mock):
        """If all are good, use the likelihoods unmodified."""
        likelihoods = pd.Series([0.5, 1, 1.5, 2])
        num_rows = pd.Series([1, 2, 3, 4])

        Sampler._find_parent_id(likelihoods, num_rows)

        expected_weights = np.array([0.5 / 5, 1 / 5, 1.5 / 5, 2 / 5])

        assert choice_mock.call_count == 1
        assert list(choice_mock.call_args[0][0]) == list(likelihoods.index)
        np.testing.assert_array_equal(choice_mock.call_args[1]['p'], expected_weights)
