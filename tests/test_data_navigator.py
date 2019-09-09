import unittest
from io import StringIO
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from tests import utils
from sdv.data_navigator import DataNavigator, Table


class TestDataNavigator(TestCase):

    def setUp(self):
        meta = utils.build_meta()
        tables = utils.build_tables()

        with patch('sdv.data_navigator.DataNavigator._get_relationships',
                   autospec=True) as relations_mock:
            with patch('sdv.data_navigator.HyperTransformer') as ht_mock:
                relations_mock.return_value = utils.get_relations_tuple()
                ht_instance_mock = ht_mock.return_value
                ht_instance_mock.table_dict = utils.get_ht_table_dict()
                ht_instance_mock.fit_transform.return_value = utils.get_ht_fit_transform()

                self.relations_mock = relations_mock
                self.ht_mock = ht_mock
                self.ht_instance_mock = ht_instance_mock  # save HyperTransformer mock
                self.data_navigator = utils.DummyDataNavigator('some_meta.json', meta, tables)

    def test__anonymize_data(self):
        """If there are pii fields, their tables are anonymized."""
        assert self.data_navigator.tables['DEMO_CUSTOMERS'].data.equals(
            utils.get_table_customers_data())
        assert self.data_navigator.tables['DEMO_ORDERS'].data.equals(
            utils.get_table_orders_data())
        assert self.data_navigator.tables['DEMO_ORDER_ITEMS'].data.equals(
            utils.get_table_order_items_data())

        assert self.ht_instance_mock._get_pii_fields.call_count == 0

        super(utils.DummyDataNavigator, self.data_navigator)._anonymize_data()

        assert not self.data_navigator.tables['DEMO_CUSTOMERS'].data.equals(
            utils.get_table_customers_data())
        assert self.data_navigator.tables['DEMO_ORDERS'].data.equals(
            utils.get_table_orders_data())
        assert self.data_navigator.tables['DEMO_ORDER_ITEMS'].data.equals(
            utils.get_table_order_items_data())

        assert self.ht_instance_mock._get_pii_fields.call_count == len(self.data_navigator.tables)

    def test___init__(self):
        """On init, relationships are built."""
        self.ht_mock.assert_called_once_with('some_meta.json', missing=None)
        self.relations_mock.assert_called_once_with(
            self.data_navigator,
            self.data_navigator.tables
        )

    def test_get_children(self):
        """get_children returns the relational children of a table."""
        # Setup
        expected_result = {'DEMO_ORDERS'}

        # Run
        result = self.data_navigator.get_children('DEMO_CUSTOMERS')

        # Check
        assert expected_result == result

    def test_get_parents(self):
        """get_children returns the relational parent of a table."""
        # Setup
        expected_result = {'DEMO_ORDERS'}

        # Run
        result = self.data_navigator.get_parents('DEMO_ORDER_ITEMS')

        # Check
        assert expected_result == result

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

            # Transformed tables must own the same columns and data dimension
            assert (table.columns == raw_table.columns).all()
            assert table.shape == raw_table.shape
            assert 'object' not in table.dtypes


if __name__ == '__main__':
    unittest.main()
