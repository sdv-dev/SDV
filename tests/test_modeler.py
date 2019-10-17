from unittest import TestCase
from unittest.mock import Mock, call, patch

import pandas as pd
import pytest
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import GaussianUnivariate

from sdv.modeler import Modeler


def test_save_and_load(tmp_path):
    """Test save and load a SDV instance"""
    # Setup
    metadata = tmp_path / "instance.pkl"
    modeler = Modeler(None)

    # Run "save"
    Modeler.save(modeler, str(metadata))

    # Asserts "save"
    assert metadata.exists()
    assert metadata.is_file()

    # Run "load"
    instance = Modeler.load(str(metadata))

    # Asserts "load"
    assert isinstance(instance, Modeler)

    assert modeler.models == dict()
    assert modeler.metadata is None
    assert modeler.model == GaussianMultivariate
    assert modeler.model_kwargs == {
        'distribution': 'copulas.univariate.gaussian.GaussianUnivariate'
    }


class TestModeler(TestCase):

    @patch('sdv.modeler.get_qualified_name')
    def test___init__default(self, mock_copulas):
        """Test create new Modeler instance with default values"""
        # Run
        modeler = Modeler(None)

        # Asserts
        expect_model_kwargs = {'distribution': mock_copulas.return_value}

        mock_copulas.assert_called_once_with(GaussianUnivariate)
        assert modeler.model_kwargs == expect_model_kwargs

    @patch('sdv.modeler.get_qualified_name', return_value='foo')
    def test___init__distribution(self, mock_copulas):
        """Test create new Modeler instance with distribution"""
        # Setup
        distribution = Mock()

        # Run
        modeler = Modeler(None, distribution=distribution)

        # Asserts
        expect_model_kwargs = {'distribution': 'foo'}

        mock_copulas.assert_called_once_with(distribution)
        assert modeler.model_kwargs == expect_model_kwargs

    def test___init__raise_error(self):
        """Test create new Modeler instance raise a ValueError"""
        # Run & asserts
        with pytest.raises(ValueError):
            Modeler(None, model=Mock(), distribution=Mock())

    def test__flatten_array(self):
        """Test get flatten array"""
        # Run
        nested = [['foo', 'bar'], 'tar']
        prefix = 'test'

        result = Modeler._flatten_array(nested, prefix=prefix)

        # Asserts
        expect = {
            'test__0__0': 'foo',
            'test__0__1': 'bar',
            'test__1': 'tar'
        }

        assert result == expect

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
        expect = {}
        assert result == expect

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
        expect = {
            'test__foo': 'value',
            'test__bar__bar_dict': 'value_bar_dict',
            'test__tar__0': 'value_tar_list'
        }

        assert result == expect

    @patch('numpy.log')
    @patch('sdv.modeler.get_qualified_name', return_value='foo')
    def test__get_model_dict_default_model(self, mock_copulas, mock_np):
        """Test get flatten model dict with default model"""
        # Setup
        std = Mock()
        std.std = 0.2
        mock_model = Mock()
        mock_model.covariance = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        mock_model.distribs = {'x': std, 'y': std, 'z': std}

        model_kwargs = {'distribution': 'foo'}

        # Run
        modeler = Mock()
        modeler.fit_model.return_value = mock_model
        modeler.model = GaussianMultivariate
        modeler.model_kwargs = model_kwargs

        Modeler._get_model_dict(modeler, None)

        # Asserts
        expect_copulas_call_count = 1
        expect_numpy_call_count = 3

        assert mock_copulas.call_count == expect_copulas_call_count
        assert mock_np.call_count == expect_numpy_call_count

    @patch('sdv.modeler.get_qualified_name')
    def test__get_model_dict_not_default_model(self, mock_copulas):
        """Test get flatten model dict without default model"""
        # Run
        modeler = Mock()
        modeler.model = None

        Modeler._get_model_dict(modeler, None)

        # Asserts
        expect_copulas_call_count = 0
        assert mock_copulas.call_count == expect_copulas_call_count

    def test_get_foreign_key_not_exist(self):
        """Test try to find a foreign key, but is not exist"""
        # Run
        modeler = Mock()
        fields = {'foo': {'ref': {'field': 'a'}}}
        primary = 'b'

        result = Modeler.get_foreign_key(modeler, fields, primary)

        # Asserts
        assert result is None

    def test_get_foreign_key_exists(self):
        """Test try to find a foreign key, exist"""
        # Run
        modeler = Mock()
        fields = {'foo': {'ref': {'field': 'a'}, 'name': 'foreign key'}}
        primary = 'a'

        result = Modeler.get_foreign_key(modeler, fields, primary)

        # Asserts
        expect = 'foreign key'
        assert result == expect

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
        expect_foo = 'bar'
        expect_fit_call = None

        modeler.model.assert_called_once_with(foo=expect_foo)
        model_mock.fit.assert_called_once_with(expect_fit_call)

    @pytest.mark.skip(reason="can't test this case properly")
    def test__create_extension_key_error(self, mock_error):
        """Test create extension, but an exception return a none"""
        pass

    def test__create_extension_zero_num_child_rows(self):
        """Test create extension, num_child_rows length is zero."""
        # Run
        modeler = Mock()
        foreign = pd.DataFrame({'foreign_key': []})
        child_table_data = pd.DataFrame({'bar': []})
        table_info = ('foreign_key', 'child_name')

        result = Modeler._create_extension(modeler, foreign, child_table_data, table_info)

        # Asserts
        assert result is None

    def test___create_extension_not_none(self):
        """Test create extension, num_child_rows length not zero."""
        # Setup
        model_dict = {}

        # Run
        modeler = Mock()
        modeler._get_model_dict.return_value = model_dict

        foreign = pd.DataFrame({'foreign_key': [1, 0]})
        child_table_data = pd.DataFrame({'bar': [0, 1]})
        table_info = ('bar', 'child_name')

        result = Modeler._create_extension(modeler, foreign, child_table_data, table_info)

        # Asserts
        assert result is not None

    @pytest.mark.skip(reason="TODO")
    def test__get_extensions(self):
        """Test get list of extensions from childs"""
        pass

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
        expect = pd.DataFrame({'pk_field': [0, 1], 'foo': [1, 0]})

        modeler.metadata.get_table_data.assert_called_once_with('test', transform=True)
        modeler.metadata.get_children.assert_called_once_with('test')
        modeler.metadata.get_primary_key.assert_called_once_with('test')
        pd.testing.assert_frame_equal(result.sort_index(axis=1), expect.sort_index(axis=1))

    def test_rcpa(self):
        """Test RCPA"""
        # Setup
        metadata_children = ['child 1', ['child 2.1', 'child 2.2'], 'child 3']

        # Run
        modeler = Mock()
        modeler.metadata.get_children.return_value = metadata_children
        table_name = 'test'
        tables = {'test': 'data'}

        Modeler.rcpa(modeler, table_name, tables)

        # Asserts
        expect_tables = {'test': modeler.cpa()}
        expect_call_count = 3
        expect_call_args_list = [
            call('child 1', expect_tables),
            call(['child 2.1', 'child 2.2'], expect_tables),
            call('child 3', expect_tables)
        ]

        assert modeler.rcpa.call_count == expect_call_count
        assert modeler.rcpa.call_args_list == expect_call_args_list

    def test__impute(self):
        """Test _impute data"""
        # Setup
        data = pd.DataFrame({'foo': [0, None, 1], 'bar': ['a', None, 'b']})

        # Run
        result = Modeler._impute(data)

        # Asserts
        expect = pd.DataFrame({'foo': [0, 0.5, 1], 'bar': ['a', 'a', 'b']})

        pd.testing.assert_frame_equal(result, expect)

    def test_model_database(self):
        """Test model using RCPA"""
        # Setup
        def rcpa_side_effect(table_name, tables):
            tables[table_name] = table_name

        metadata_table_names = ['foo', 'bar', 'tar']
        metadata_parents = [
            None,
            'bar_parent',
            None
        ]

        # Run
        modeler = Mock()
        modeler.metadata.get_table_names.return_value = metadata_table_names
        modeler.metadata.get_parents.side_effect = metadata_parents
        modeler.rcpa.side_effect = rcpa_side_effect
        modeler.models = dict()

        Modeler.model_database(modeler)

        # Asserts
        expect_metadata_parents_call_count = 3
        expect_metadata_parents_call_arg_list = [call('foo'), call('bar'), call('tar')]

        assert modeler.metadata.get_parents.call_count == expect_metadata_parents_call_count
        assert modeler.metadata.get_parents.call_args_list == expect_metadata_parents_call_arg_list
