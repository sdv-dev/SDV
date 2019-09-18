import unittest
from unittest import TestCase
from unittest.mock import Mock, call, patch

import pandas as pd

from sdv.data_navigator import CSVDataLoader, DataLoader, DataNavigator, Table


class TestDataLoader(TestCase):

    @patch('json.load')
    @patch('builtins.open')
    def test___init__(self, open_mock, load_mock):
        """load meta on create dataloader instance"""

        # Setup
        # mock_open()
        load_mock.return_value = {'some': 'meta'}

        # Run
        meta_filename = 'meta_filename.json'

        result = DataLoader(meta_filename)

        # Asserts
        assert result.meta == {'some': 'meta'}

    def test_load_data(self):
        """raise not implemented exception"""

        # Setup

        # Run and asserts
        mock_data_loader = Mock()
        with self.assertRaises(NotImplementedError):
            DataLoader.load_data(mock_data_loader)


class TestCSVDataLoader(TestCase):

    def test__format_table_meta(self):
        """format table meta dict"""

        # Setup

        # Run
        meta = {
            'fields': [{
                'name': 'a_field',
                'foo': 'foo'
            }]
        }

        csv_data_loader_mock = Mock()

        result = CSVDataLoader._format_table_meta(csv_data_loader_mock, meta)

        # Asserts
        expect = {
            'fields': {
                'a_field': {
                    'name': 'a_field',
                    'foo': 'foo'
                }
            }
        }

        assert result == expect

    @patch('sdv.data_navigator.DataNavigator')
    @patch('pandas.read_csv')
    def test_load_data(self, read_mock, dn_mock):
        """load_data to build a DataNavigator"""

        # SetUp
        meta = {
            'path': '',
            'tables': [{
                'use': True,
                'name': 'DEMO',
                'path': 'some_path.csv',
                'fields': [{
                    'name': 'a_field',
                    'foo': 'foo'
                }]
            }]
        }
        meta_filename = 'meta_filename.json'

        dn_mock.return_value = Mock()

        format_mock = Mock()
        format_mock.return_value = {'some': 'meta'}

        read_mock.return_value = pd.DataFrame({'foo': [0, 1]})

        # Run
        csv_data_loader_mock = Mock()
        csv_data_loader_mock.meta = meta
        csv_data_loader_mock.meta_filename = meta_filename
        csv_data_loader_mock._format_table_meta = format_mock

        CSVDataLoader.load_data(csv_data_loader_mock)

        # Asserts
        exp_format_args = {
            'use': True,
            'name': 'DEMO',
            'path': 'some_path.csv',
            'fields': [{
                'name': 'a_field',
                'foo': 'foo'
            }]
        }

        exp_data_navigator_meta = {
            'path': '',
            'tables': [{
                'use': True,
                'name': 'DEMO',
                'path': 'some_path.csv',
                'fields': [{
                    'name': 'a_field',
                    'foo': 'foo'
                }]
            }]
        }

        exp_data_navigator_tables = {
            'DEMO': Table(pd.DataFrame({'foo': [0, 1]}), {'some': 'meta'})
        }

        assert_meta_filename, assert_meta, assert_tables = dn_mock.call_args[0]

        format_mock.assert_called_once_with(exp_format_args)

        assert assert_meta_filename == 'meta_filename.json'
        assert assert_meta == exp_data_navigator_meta
        assert assert_tables.keys() == exp_data_navigator_tables.keys()

        pd.testing.assert_frame_equal(
            assert_tables['DEMO'].data,
            exp_data_navigator_tables['DEMO'].data
        )


class TestDataNavigator(TestCase):

    def test__anonymize_data(self):
        """Anonymoze data in tables with pii fields"""

        # Setup
        def side_effect_get_pii(ht_meta):
            if ht_meta['ht'] == 'a':
                return ['a_fields']

        anonymized_table = pd.DataFrame({
            'a_fields': [1, 2, 3]
        })
        anonymized_another_table = pd.DataFrame({
            'a_fields': [4, 5, 6]
        })

        ht_mock = Mock()
        ht_mock.table_dict = {
            'a_table': (anonymized_table, {'ht': 'a'}),
            'another_table': (anonymized_another_table, {'ht': 'b'})
        }

        ht_mock._get_pii_fields.side_effect = side_effect_get_pii

        a_table = pd.DataFrame({
            'a_fields': [1, 2, 3]
        })
        another_table = Table(pd.DataFrame(), {'another': 'metadata'})
        tables = {
            'a_table': Table(a_table, {'some': 'metadata'}),
            'another_table': another_table,
        }

        # Run
        data_navigator_mock = Mock()
        data_navigator_mock.tables = tables
        data_navigator_mock.ht = ht_mock

        DataNavigator._anonymize_data(data_navigator_mock)

        # Asserts
        exp_call_args_list = [
            call({'ht': 'a'}),
            call({'ht': 'b'}),
        ]
        exp_a_table_dataframe = pd.DataFrame({
            'a_fields': [1, 2, 3]
        })
        exp_another_dataframe = pd.DataFrame()

        pd.testing.assert_frame_equal(tables['a_table'].data, exp_a_table_dataframe)
        pd.testing.assert_frame_equal(tables['another_table'].data, exp_another_dataframe)

        for arg_item in ht_mock._get_pii_fields.call_args_list:
            assert arg_item in exp_call_args_list

    @patch('sdv.data_navigator.DataNavigator._get_relationships')
    @patch('sdv.data_navigator.DataNavigator._anonymize_data')
    @patch('sdv.data_navigator.HyperTransformer')
    def test___init__(self, ht_mock, anonymize_mock, relationships_mock):
        """__init__ without missing."""

        # Setup
        ht_mock.return_value = Mock()
        relationships_mock.return_value = 'foo', 'bar', 'tar'

        # Run
        data_navigator_mock = Mock()
        data_navigator_mock._anonymize_data = anonymize_mock
        data_navigator_mock._get_relationships = relationships_mock

        DataNavigator(
            'meta_filename',
            {'some': 'meta'},
            {'a_table': 'table'}
        )

        # Asserts
        ht_mock.assert_called_once_with('meta_filename', missing=None)

        data_navigator_mock._anonymize_data.assert_called_once_with()
        data_navigator_mock._get_relationships.assert_called_once_with({'a_table': 'table'})

    @patch('sdv.data_navigator.DataNavigator._get_relationships')
    @patch('sdv.data_navigator.DataNavigator._anonymize_data')
    @patch('sdv.data_navigator.HyperTransformer')
    def test__init__with_missing(self, ht_mock, anonymize_mock, relationships_mock):
        """__init__ with missing."""

        # Setup
        ht_mock.return_value = Mock()
        relationships_mock.return_value = 'foo', 'bar', 'tar'

        # Run
        data_navigator_mock = Mock()
        data_navigator_mock._anonymize_data = anonymize_mock
        data_navigator_mock._get_relationships = relationships_mock

        DataNavigator(
            'meta_filename',
            {'some': 'meta'},
            {'a_table': 'table'}
        )

        # Asserts
        ht_mock.assert_called_once_with('meta_filename', missing=None)

        data_navigator_mock._anonymize_data.assert_called_once_with()
        data_navigator_mock._get_relationships.assert_called_once_with({'a_table': 'table'})

    def test_get_children(self):
        """get_children returns the relational children of a table."""

        # Setup
        childs = {
            'DEMO': {'child_a': 'foo'}
        }

        # Run
        data_navigator_mock = Mock()
        data_navigator_mock.child_map = childs

        result = DataNavigator.get_children(data_navigator_mock, 'DEMO')

        # Asserts
        expect = {'child_a': 'foo'}

        assert expect == result

    def test_get_children_no_childrens(self):
        """No children from the given table"""

        # Setup
        childs = {}

        # Run
        data_navigator_mock = Mock()
        data_navigator_mock.child_map = childs

        result = DataNavigator.get_children(data_navigator_mock, 'DEMO')

        # Asserts
        expect = set()

        assert expect == result

    def test_get_parents(self):
        """get_parents returns the relational parent of a table."""

        # Setup
        parents = {
            'DEMO': {'parent_a': 'foo'}
        }

        # Run
        data_navigator_mock = Mock()
        data_navigator_mock.parent_map = parents

        result = DataNavigator.get_parents(data_navigator_mock, 'DEMO')

        # Asserts
        expect = {'parent_a': 'foo'}

        assert expect == result

    def test_get_parents_no_parents(self):
        """No parents from the given table"""

        # Setup
        parents = {}

        # Run
        data_navigator_mock = Mock()
        data_navigator_mock.parent_map = parents

        result = DataNavigator.get_parents(data_navigator_mock, 'DEMO')

        # Asserts
        expect = set()

        assert expect == result

    def test_get_data(self):
        """Retrieve table data"""

        # Setup
        data = pd.DataFrame({
            'foo': [0, 1]
        })
        tables = {
            'DEMO': Table(data, 'meta')
        }

        # Run
        data_navigator_mock = Mock()
        data_navigator_mock.tables = tables

        result = DataNavigator.get_data(data_navigator_mock, 'DEMO')

        # Asserts
        expect = pd.DataFrame({
            'foo': [0, 1]
        })

        pd.testing.assert_frame_equal(result, expect)

    def test_get_meta_data(self):
        """Retrieve table meta"""

        # Setup
        tables = {
            'DEMO': Table(None, 'meta')
        }

        # Run
        data_navigator_mock = Mock()
        data_navigator_mock.tables = tables

        result = DataNavigator.get_meta_data(data_navigator_mock, 'DEMO')

        # Asserts
        expect = 'meta'

        assert result == expect

    def test_update_mapping_when_item(self):
        """update_mapping when item is something"""

        # Setup

        # Run
        mapping = {'foo': set(['bar'])}
        key = 'foo'
        value = 'tar'

        data_navigator_mock = Mock()

        result = DataNavigator.update_mapping(data_navigator_mock, mapping, key, value)

        # Asserts
        expect = {'foo': set(['bar', 'tar'])}

        assert result == expect

    def test_update_mapping_when_no_item(self):
        """update_mapping when item is nothing"""

        # Setup

        # Run
        mapping = {}
        key = 'foo'
        value = 'tar'

        data_navigator_mock = Mock()

        result = DataNavigator.update_mapping(data_navigator_mock, mapping, key, value)

        # Asserts
        expect = {'foo': set(['tar'])}

        assert result == expect

    def test__get_relashionships(self):
        """_get_relashionships returns parents, children and foreign_keys dicts."""

        # Setup
        meta = {
            'fields': {
                'a_field': {
                    'name': 'a_field',
                    'ref': {
                        'table': 'DEMO_2',
                        'field': 'DEMO_2_ID'
                    }
                }
            }
        }

        tables = {
            'DEMO': Table('data', meta)
        }

        update_mock = Mock()
        update_mock.side_effect = [
            'child',
            'parent'
        ]

        # Run
        data_navigator_mock = Mock()
        data_navigator_mock.update_mapping = update_mock

        result = DataNavigator._get_relationships(data_navigator_mock, tables)

        # Asserts
        expect = 'child', 'parent', {('DEMO', 'DEMO_2'): ('DEMO_2_ID', 'a_field')}
        exp_args_list = [call({}, 'DEMO_2', 'DEMO'), call({}, 'DEMO', 'DEMO_2')]

        assert result == expect
        update_mock.call_args_list == exp_args_list

    def test_transform_data_default_transformers(self):
        """transform_data with default transformers."""

        # Setup
        default_transformers = ['NumberTransformer', 'DTTransformer', 'CatTransformer']

        ht_mock = Mock()
        ht_mock.fit_transform.return_value = {
            'some': 'data'
        }

        # Run
        data_navigator_mock = Mock()
        data_navigator_mock.DEFAULT_TRANSFORMERS = default_transformers
        data_navigator_mock.ht = ht_mock

        result = DataNavigator.transform_data(data_navigator_mock, None)

        # Asserts
        expect = {'some': 'data'}
        expect_transformers = ['NumberTransformer', 'DTTransformer', 'CatTransformer']

        assert result == expect

        ht_mock.fit_transform.assert_called_once_with(transformer_list=expect_transformers)

    def test_transform_data_with_transformers(self):
        """transform_data with transformers from parameters"""

        # Setup
        transformers = ['NumberTransformer', 'DTTransformer']

        ht_mock = Mock()
        ht_mock.fit_transform.return_value = {
            'some': 'data'
        }

        # Run
        data_navigator_mock = Mock()
        data_navigator_mock.ht = ht_mock

        result = DataNavigator.transform_data(data_navigator_mock, transformers=transformers)

        # Asserts
        expect = {'some': 'data'}
        expect_transformers = ['NumberTransformer', 'DTTransformer']

        assert result == expect

        ht_mock.fit_transform.assert_called_once_with(transformer_list=expect_transformers)


if __name__ == '__main__':
    unittest.main()
