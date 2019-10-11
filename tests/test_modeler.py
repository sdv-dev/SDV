from collections import OrderedDict
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from copulas import EPSILON
from copulas.multivariate import GaussianMultivariate, VineCopula
from copulas.univariate import GaussianUnivariate, KDEUnivariate

from sdv.modeler import Modeler


class TestModeler(TestCase):

    @patch('sdv.modeler.get_qualified_name')
    def test___init__default(self, mock_copulas):
        """Test create new Modeler instance with default values"""
        # Run
        modeler = Modeler(None)

        # Asserts
        expect_model_kwargs = {'distribution': mock_copulas.return_value}

        mock_copulas.assert_called_once_with(GaussianUnivariate)
        self.assertEqual(modeler.model_kwargs, expect_model_kwargs)

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
        self.assertEqual(modeler.model_kwargs, expect_model_kwargs)

    def test___init__raise_error(self):
        """Test create new Modeler instance raise a ValueError"""
        # Run & asserts
        with self.assertRaises(ValueError):
            Modeler(None, model=Mock(), distribution=Mock())

    @pytest.mark.skip(reason="currently not implemented")
    def test_save(self):
        """Test save Modeler instance"""
        pass

    @pytest.mark.skip(reason="currently not implemented")
    def test_load(self):
        """Test load Modeler instance"""
        pass

    def test_get_primary_key_value_default(self):
        """Test get primary key is DEFAULT_PRIMARY_KEY"""
        # Run
        modeler = Mock()
        modeler.DEFAULT_PRIMARY_KEY = 'foo'

        result = Modeler.get_primary_key_value(modeler, 'foo', 0, {})

        # Asserts
        expect = 'foo0'
        self.assertEqual(result, expect)

    def test_get_primary_key_value_other(self):
        """Test get primary key is not DEFAULT_PRIMARY_KEY"""
        # Run
        modeler = Mock()
        modeler.DEFAULT_PRIMARY_KEY = 'foo'

        result = Modeler.get_primary_key_value(modeler, 'bar', 0, {'bar': 'bar'})

        # Asserts
        expect = 'bar'
        self.assertEqual(result, expect)

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

        self.assertEqual(result, expect)

    def test__flatten_dict_of_ignored_keys(self):
        """Test get flatten dict of ignored keys"""
        # Run
        nested = {
            'fitted': 'value_1',
            'distribution': 'value_2',
            'type': 'value_3'
        }

        result = Modeler._flatten_dict(nested, prefix='test')

        # Asserts
        expect = {}
        self.assertEqual(result, expect)

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

        self.assertEqual(result, expect)

    @patch('numpy.log')
    @patch('sdv.modeler.get_qualified_name', return_value='foo')
    def test__get_model_dict_default_model(self, mock_copulas, mock_np):
        """Test get flatten model dict with default model"""
        # Setup
        std = Mock()
        std.std = 0.2
        mock_model = Mock()
        mock_model.covariance = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
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

        self.assertEqual(mock_copulas.call_count, expect_copulas_call_count)
        self.assertEqual(mock_np.call_count, expect_numpy_call_count)

    @patch('sdv.modeler.get_qualified_name')
    def test__get_model_dict_not_default_model(self, mock_copulas):
        """Test get flatten model dict without default model"""
        # Run
        modeler = Mock()
        modeler.model = None

        Modeler._get_model_dict(modeler, None)

        # Asserts
        expect_copulas_call_count = 0
        self.assertEqual(mock_copulas.call_count, expect_copulas_call_count)

    def test_get_foreign_key_not_exist(self):
        """Test try to find a foreign key, but is not exist"""
        # Run
        modeler = Mock()
        fields = {'foo': {'ref': {'field': 'a'}}}
        primary = 'b'

        result = Modeler.get_foreign_key(modeler, fields, primary)

        # Asserts
        self.assertIsNone(result)

    def test_get_foreign_key_exists(self):
        """Test try to find a foreign key, exist"""
        # Run
        modeler = Mock()
        fields = {'foo': {'ref': {'field': 'a'}, 'name': 'foreign key'}}
        primary = 'a'

        result = Modeler.get_foreign_key(modeler, fields, primary)

        # Asserts
        expect = 'foreign key'
        self.assertEqual(result, expect)

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
        self.assertIsNone(result)

    def test___create_extension_not_none(self):
        """Test create extension, num_child_rows length not zero."""
        # Setup
        model_dict = {}

        # Run
        modeler = Mock()
        modeler._get_model_dict.return_value = model_dict

        foreign = pd.DataFrame({'foreign_key': [0, 1]})
        child_table_data = pd.DataFrame({'bar': [1, 0]})
        table_info = ('foreign_key', 'child_name')

        result = Modeler._create_extension(modeler, foreign, child_table_data, table_info)

        # Asserts
        self.assertIsNone(result)
