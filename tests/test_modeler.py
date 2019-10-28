from unittest import TestCase
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
from copulas.multivariate import GaussianMultivariate

from sdv.metadata import Metadata
from sdv.modeler import Modeler


class TestModeler(TestCase):

    @patch('sdv.modeler.open')
    @patch('sdv.modeler.pickle')
    def test_save(self, pickle_mock, open_mock):
        metadata = Mock(autopsec=Metadata)
        modeler = Modeler(metadata)
        modeler.save('save/path.pkl')

        open_mock.assert_called_once_with('save/path.pkl', 'wb')
        output = open_mock.return_value.__enter__.return_value
        pickle_mock.dump.assert_called_once_with(modeler, output)

    @patch('sdv.modeler.open')
    @patch('sdv.modeler.pickle')
    def test_load(self, pickle_mock, open_mock):
        returned = Modeler.load('save/path.pkl')

        open_mock.assert_called_once_with('save/path.pkl', 'rb')
        output = open_mock.return_value.__enter__.return_value
        pickle_mock.load.assert_called_once_with(output)
        assert returned is pickle_mock.load.return_value

    def test___init__default(self):
        """Test create new Modeler instance with default values"""
        # Run
        modeler = Modeler('test')

        # Asserts
        assert modeler.models == dict()
        assert modeler.metadata == 'test'
        assert modeler.model == GaussianMultivariate
        assert modeler.model_kwargs == dict()

    def test__flatten_array(self):
        """Test get flatten array"""
        # Run
        nested = [['foo', 'bar'], 'tar']
        prefix = 'test'

        result = Modeler._flatten_array(nested, prefix=prefix)

        # Asserts
        expected = {
            'test__0__0': 'foo',
            'test__0__1': 'bar',
            'test__1': 'tar'
        }

        assert result == expected

    def test__flatten_dict_of_ignored_keys(self):
        """Test get flatten dict of ignored keys"""
        # Run
        nested = {
            'fitted': 'value_1',
            'distribution': 'value_2',
            'type': 'value_3'
        }
        prefix = 'test'

        result = Modeler._flatten_dict(nested, prefix=prefix)

        # Asserts
        expected = {}
        assert result == expected

    def test__flatten_dict(self):
        """Test get flatten dict with some result"""
        # Run
        nested = {
            'foo': 'value',
            'bar': {'bar_dict': 'value_bar_dict'},
            'tar': ['value_tar_list']
        }

        result = Modeler._flatten_dict(nested, prefix='test')

        # Asserts
        expected = {
            'test__foo': 'value',
            'test__bar__bar_dict': 'value_bar_dict',
            'test__tar__0': 'value_tar_list'
        }

        assert result == expected

    @patch('numpy.log')
    def test__get_model_dict_default_model(self, log_mock):
        """Test get flatten model dict with default model"""
        # Setup
        model_fitted = None

        # Run
        modeler = Mock(spec=Modeler)
        modeler._fit_model.return_value = model_fitted
        modeler._flatten_dict.return_value = dict()

        data = pd.DataFrame()

        result = Modeler._get_model_dict(modeler, 'test', data)

        # Asserts
        assert result == 'dict'

    def test_fit_model(self):
        """Test fit model"""
        # Setup
        model_mock = Mock()

        # Run
        modeler = Mock()
        modeler.model_kwargs = {'foo': 'bar'}
        modeler.model.return_value = model_mock
        Modeler.fit_model(modeler, None)

        # Asserts
        expected_foo = 'bar'
        expected_fit_call = None

        modeler.model.assert_called_once_with(foo=expected_foo)
        model_mock.fit.assert_called_once_with(expected_fit_call)

    def test__get_extensions(self):
        """Test get list of extensions from childs"""
        # Setup
        model_dict = [
            {'model': 'data 1'},
            {'model': 'data 2'},
            {'model': 'data 3'}
        ]

        # Run
        modeler = Mock()
        modeler._get_model_dict.side_effect = model_dict

        child_name = 'some_name'
        child_table = pd.DataFrame({'foo': ['aaa', 'bbb', 'ccc']})

        result = Modeler._get_extension(modeler, child_name, child_table, 'foo')

        # Asserts
        expected = pd.DataFrame({
            '__some_name__model': ['data 1', 'data 2', 'data 3'],
            '__some_name__child_rows': [1, 1, 1]
        }, index=['aaa', 'bbb', 'ccc'])

        pd.testing.assert_frame_equal(result, expected)
        assert modeler._get_model_dict.call_count == 3

    def test_cpa(self):
        """Test CPA with extensions"""
        # Setup
        metadata_table_data = pd.DataFrame({'pk_field': [0, 1]})
        metadata_primary_key = 'pk_field'
        extensions = [pd.Series([1, 0], name='foo')]

        # Run
        modeler = Mock()
        modeler.metadata.get_table_data.return_value = metadata_table_data
        modeler.metadata.get_primary_key.return_value = metadata_primary_key
        modeler._get_extensions.return_value = extensions

        table_name = 'test'
        tables = {
            'test': None
        }

        result = Modeler.cpa(modeler, table_name, tables)

        # Asserts
        expected = pd.DataFrame({'pk_field': [0, 1], 'foo': [1, 0]})

        modeler.metadata.get_table_data.assert_called_once_with('test', transform=True)
        modeler.metadata.get_children.assert_called_once_with('test')
        modeler.metadata.get_primary_key.assert_called_once_with('test')
        pd.testing.assert_frame_equal(result.sort_index(axis=1), expected.sort_index(axis=1))

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

        # Run
        modeler = Mock()
        modeler.metadata.get_table_names.return_value = metadata_table_names
        modeler.metadata.get_parents.side_effect = metadata_parents
        modeler.rcpa.side_effect = rcpa_side_effect
        modeler.models = dict()

        Modeler.model_database(modeler)

        # Asserts
        expected_metadata_parents_call_count = 3
        expected_metadata_parents_call = [call('foo'), call('bar'), call('tar')]

        assert modeler.metadata.get_parents.call_count == expected_metadata_parents_call_count
        assert modeler.metadata.get_parents.call_args_list == expected_metadata_parents_call
