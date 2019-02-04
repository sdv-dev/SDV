from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from sdv.data_navigator import CSVDataLoader, DataNavigator, Table


class TestDataNavigator(TestCase):

    def setUp(self):
        data_loader = CSVDataLoader('tests/data/meta.json')
        self.data_navigator = data_loader.load_data()

    @patch('sdv.data_navigator.DataNavigator._get_relationships', autospec=True)
    @patch('sdv.data_navigator.HyperTransformer')
    def test__anonymize_data(self, hypertransformer_mock, relations_mock):
        """If there are pii fields, their tables are anonymized."""
        # Setup
        meta_filename = ''
        table_data = pd.DataFrame([
            {
                'primary_key': 1,
                'credit_card_number': '1111-2222-3333-4444'
            },
            {
                'primary_key': 2,
                'credit_card_number': '0000-0000-0000-0000'
            },
            {
                'primary_key': 3,
                'credit_card_number': '2222-2222-2222-2222'
            },
        ])

        table_meta = {
            'fields': [
                {
                    'name': 'credit_card_number',
                    'type': 'categorical',
                    'pii': True,
                    'pii_category': 'credit_card_number'
                },
                {
                    'name': 'primary_key',
                    'subtype': 'integer',
                    'type': 'number',
                    'regex': '^[0-9]{10}$'
                },
            ],
            'headers': True,
            'name': 'table',
            'path': 'table.csv',
            'primary_key': 'primary_key',
            'use': True
        }

        tables = {
            'table': Table(table_data, table_meta)
        }

        meta = {
            'path': '',
            'tables': [
                table_meta
            ]
        }

        ht_instance = hypertransformer_mock.return_value
        ht_instance.table_dict = {
            'table': ('anonymized_table', 'anonymized_table_metadata')
        }
        relations_mock.return_value = ('child_map', 'parent_map', 'foreign_keys')

        data_navigator = DataNavigator(meta_filename, meta, tables)

        # We reset the mocks as `anonymize_data` is also called on __init__
        relations_mock.reset_mock()
        hypertransformer_mock.reset_mock()
        ht_instance.reset_mock()

        # Run
        result = data_navigator._anonymize_data()

        # Check
        assert result is None
        assert data_navigator.tables['table'].data == 'anonymized_table'
        assert data_navigator.tables['table'].meta == table_meta

        assert len(relations_mock.call_args_list) == 0
        assert len(hypertransformer_mock.call_args_list) == 0
        ht_instance._get_pii_fields.assert_called_once_with('anonymized_table_metadata')

    @patch('sdv.data_navigator.DataNavigator._get_relationships', autospec=True)
    @patch('sdv.data_navigator.DataNavigator._anonymize_data', autospec=True)
    @patch('sdv.data_navigator.HyperTransformer')
    def test___init__(self, hypertransformer_mock, anon_mock, relations_mock):
        """On init, relationships are built."""
        # Setup
        meta_filename = ''
        meta = {'meta': 'data'}
        tables = {'table_name': Table('data', 'meta')}

        relations_mock.return_value = ('child_map', 'parent_map', 'foreign_keys')

        # Run
        data_navigator = DataNavigator(meta_filename, meta, tables)

        # Check
        assert data_navigator.meta == {'meta': 'data'}
        assert data_navigator.tables == tables
        assert data_navigator.transformed_data is None
        assert data_navigator.ht == hypertransformer_mock.return_value
        assert data_navigator.child_map == 'child_map'
        assert data_navigator.parent_map == 'parent_map'
        assert data_navigator.foreign_keys == 'foreign_keys'

        hypertransformer_mock.assert_called_once_with('', missing=None)
        anon_mock.assert_called_once_with(data_navigator)
        relations_mock.assert_called_once_with(data_navigator, tables)

    def test_get_children(self):
        """get_children returns the relational children of a table."""
        # Setup
        expected_result = {'DEMO_ORDERS'}

        # Run
        result = self.data_navigator.get_children('DEMO_CUSTOMERS')

        # Check
        assert expected_result == result

        parent_pk = self.data_navigator.get_meta_data('DEMO_CUSTOMERS')['primary_key']

        for item in expected_result:
            item_table = self.data_navigator.get_data(item)
            item_meta = self.data_navigator.get_meta_data(item)
            ref = item_meta['fields'][parent_pk].get('ref')

            # The parent_pk column is in the children table.
            assert parent_pk in item_table.columns

            # The parent_pk is referencing to the parent table on the children meta.
            assert ref['table'] == 'DEMO_CUSTOMERS'

    def test_get_parents(self):
        """get_children returns the relational parent of a table."""

        # Setup
        expected_result = {'DEMO_ORDERS'}

        # Run
        result = self.data_navigator.get_parents('DEMO_ORDER_ITEMS')

        # Check
        assert expected_result == result

        children_meta = self.data_navigator.get_meta_data('DEMO_ORDER_ITEMS')
        children_data = self.data_navigator.get_data('DEMO_ORDER_ITEMS')

        for item in expected_result:
            item_meta = self.data_navigator.get_meta_data(item)
            parent_pk = item_meta['primary_key']
            ref = children_meta['fields'][parent_pk].get('ref')

            # The parent_pk column is in the children table.
            assert parent_pk in children_data.columns

            # The parent_pk is referencing to the parent table on the children meta.
            assert ref['table'] == 'DEMO_ORDERS'

    def test__get_relashionships(self):
        """_get_relashionships returns parents, children and foreign_keys dicts."""
        # Setup
        expected_children = {
            'DEMO_CUSTOMERS': {'DEMO_ORDERS'},
            'DEMO_ORDERS': {'DEMO_ORDER_ITEMS'}
        }
        expected_parents = {
            'DEMO_ORDERS': {'DEMO_CUSTOMERS'},
            'DEMO_ORDER_ITEMS': {'DEMO_ORDERS'}
        }
        expected_foreign_keys = {
            ('DEMO_ORDERS', 'DEMO_CUSTOMERS'): ('CUSTOMER_ID', 'CUSTOMER_ID'),
            ('DEMO_ORDER_ITEMS', 'DEMO_ORDERS'): ('ORDER_ID', 'ORDER_ID')
        }

        # Run
        children, parents, foreign_keys = self.data_navigator._get_relationships(
            self.data_navigator.tables)

        # Check
        assert children == expected_children
        assert parents == expected_parents
        assert foreign_keys == expected_foreign_keys

        # All children are referenced in the parent map too.
        for parent_name, childs in children.items():
            for child in childs:
                assert parent_name in parents[child]
                assert(child, parent_name) in foreign_keys

    def test_transform_data(self):
        """transform_data turns all data into numeric values."""

        # Run
        result = self.data_navigator.transform_data()

        # Check
        assert result.keys() == self.data_navigator.tables.keys()

        for name, table in result.items():
            raw_table = self.data_navigator.tables[name].data

            assert (table.columns == raw_table.columns).all()
            assert table.shape == raw_table.shape
            assert 'object' not in table.dtypes
