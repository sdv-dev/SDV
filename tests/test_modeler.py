from unittest import TestCase
from unittest.mock import Mock, call, patch

import pandas as pd
from copulas.multivariate import GaussianMultivariate

from sdv.metadata import Metadata
from sdv.modeler import Modeler


class TestModeler(TestCase):

    def test___init__default(self):
        """Test create new Modeler instance with default values"""
        # Run
        modeler = Modeler('test')

        # Asserts
        assert modeler.models == dict()
        assert modeler.metadata == 'test'
        assert modeler.model == GaussianMultivariate
        assert modeler.model_kwargs == dict()

    def test___init__with_arguments(self):
        # Run
        model = Mock()
        modeler = Modeler({'some': 'metadata'}, model=model, model_kwargs={'some': 'kwargs'})

        # Asserts
        assert modeler.models == dict()
        assert modeler.metadata == {'some': 'metadata'}
        assert modeler.model == model
        assert modeler.model_kwargs == {'some': 'kwargs'}

    def test__flatten_array(self):
        """Test get flatten array"""
        # Run
        nested = [['foo', 'bar'], 'tar']
        result = Modeler._flatten_array(nested, prefix='test')

        # Asserts
        expected = {
            'test__0__0': 'foo',
            'test__0__1': 'bar',
            'test__1': 'tar'
        }
        assert result == expected

    def test__flatten_dict(self):
        """Test get flatten dict with some result"""
        # Run
        nested = {
            'foo': 'value',
            'bar': {'bar_dict': 'value_bar_dict'},
            'tar': ['value_tar_list_0', 'value_tar_list_1'],
            'fitted': 'value_1',
            'distribution': 'value_2',
            'type': 'value_3'
        }
        result = Modeler._flatten_dict(nested, prefix='test')

        # Asserts
        expected = {
            'test__foo': 'value',
            'test__bar__bar_dict': 'value_bar_dict',
            'test__tar__0': 'value_tar_list_0',
            'test__tar__1': 'value_tar_list_1'
        }
        assert result == expected

    @patch('numpy.log')
    def test__get_model_dict_default_model(self, log_mock):
        """Test get flatten model dict with default model"""
        # Setup
        model_fitted = Mock()
        model_fitted.covariance = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]

        distrib1 = Mock()
        distrib1.std = 0.2
        distrib2 = Mock()
        distrib2.std = None
        distrib3 = Mock()
        distrib3.std = 0.5
        distrib4 = Mock()
        distrib4.std = 0.3

        model_fitted.distribs = {
            'distrib1': distrib1,
            'distrib2': distrib2,
            'distrib3': distrib3,
            'distrib4': distrib4
        }

        modeler = Mock(spec=Modeler)
        modeler._fit_model.return_value = model_fitted

        # Run
        data = pd.DataFrame({'data': [1, 2, 3]})
        result = Modeler._get_model_dict(modeler, data)

        # Asserts
        assert result == modeler._flatten_dict.return_value

        expected_log_mock_call = [
            call(0.2),
            call(0.5),
            call(0.3)
        ]
        assert sorted(log_mock.call_args_list) == sorted(expected_log_mock_call)

        pd.testing.assert_frame_equal(
            modeler._fit_model.call_args[0][0],
            pd.DataFrame({'data': [1, 2, 3]})
        )

    def test__get_extensions(self):
        """Test get list of extensions from childs"""
        # Setup
        modeler = Mock()
        model_dict = [
            {'model': 'data 1'},
            {'model': 'data 2'},
            {'model': 'data 3'}
        ]
        modeler._get_model_dict.side_effect = model_dict

        # Run
        child_table = pd.DataFrame({'foo': ['aaa', 'bbb', 'ccc']})
        result = Modeler._get_extension(modeler, 'some_name', child_table, 'foo')

        # Asserts
        expected = pd.DataFrame({
            '__some_name__model': ['data 1', 'data 2', 'data 3'],
            '__some_name__child_rows': [1, 1, 1]
        }, index=['aaa', 'bbb', 'ccc'])
        pd.testing.assert_frame_equal(result, expected)
        assert modeler._get_model_dict.call_count == 3

    def test_cpa_with_tables_no_primary_key(self):
        """Test CPA with tables and no primary key."""
        # Setup
        modeler = Mock(spec=Modeler)
        modeler.metadata = Mock(spec=Metadata)
        modeler.models = dict()
        modeler.metadata.transform.return_value = pd.DataFrame({'data': [1, 2, 3]})
        modeler.metadata.get_primary_key.return_value = None
        modeler._fit_model.return_value = 'fitted model'

        # Run
        tables = {'test': pd.DataFrame({'data': ['a', 'b', 'c']})}
        result = Modeler.cpa(modeler, 'test', tables)

        # Asserts
        expected = pd.DataFrame({'data': [1, 2, 3]})
        expected_transform_call = pd.DataFrame({'data': ['a', 'b', 'c']})

        assert modeler.metadata.load_table.call_count == 0
        assert modeler.metadata.transform.call_args[0][0] == 'test'
        pd.testing.assert_frame_equal(
            modeler.metadata.transform.call_args[0][1],
            expected_transform_call
        )
        pd.testing.assert_frame_equal(modeler._fit_model.call_args[0][0], expected)
        pd.testing.assert_frame_equal(result, expected)

    def test__impute(self):
        """Test _impute data"""
        # Setup
        data = pd.DataFrame({'foo': [0, None, 1], 'bar': ['a', None, 'b']})

        # Run
        result = Modeler._impute(data)

        # Asserts
        expected = pd.DataFrame({'foo': [0, 0.5, 1], 'bar': ['a', 'a', 'b']})
        pd.testing.assert_frame_equal(result, expected)

    def test_model_database(self):
        """Test model using RCPA"""
        # Setup
        def rcpa_side_effect(table_name, tables):
            tables[table_name] = table_name

        metadata_table_names = ['foo', 'bar', 'tar']
        metadata_parents = [None, 'bar_parent', None]

        modeler = Mock()
        modeler.metadata.get_tables.return_value = metadata_table_names
        modeler.metadata.get_parents.side_effect = metadata_parents
        modeler.rcpa.side_effect = rcpa_side_effect
        modeler.models = dict()

        # Run
        Modeler.model_database(modeler)

        # Asserts
        expected_metadata_parents_call_count = 3
        expected_metadata_parents_call = [call('foo'), call('bar'), call('tar')]
        assert modeler.metadata.get_parents.call_count == expected_metadata_parents_call_count
        assert modeler.metadata.get_parents.call_args_list == expected_metadata_parents_call
