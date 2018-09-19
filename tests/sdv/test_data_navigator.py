from unittest import TestCase

from sdv.data_navigator import CSVDataLoader


class TestDataNavigator(TestCase):

    def setUp(self):
        data_loader = CSVDataLoader('tests/data/meta.json')
        self.data_navigator = data_loader.load_data()

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
