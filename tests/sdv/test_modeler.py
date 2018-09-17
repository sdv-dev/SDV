from unittest import TestCase

import pandas as pd

from sdv.data_navigator import CSVDataLoader
from sdv.modeler import Modeler


class ModelerTest(TestCase):

    def setUp(self):
        """Set up test fixtures, if any."""
        dl = CSVDataLoader('tests/data/meta.json')
        self.dn = dl.load_data()
        self.dn.transform_data()
        self.modeler = Modeler(self.dn)

    def test_create_extension(self):
        """Tests that the create extension method returns correct parameters."""
        # Setup
        child_table = self.dn.get_data('DEMO_ORDERS')
        user = child_table[child_table['CUSTOMER_ID'] == 50]
        expected = pd.Series([
            1.500000e+00, 0.000000e+00, -1.269991e+00,
            0.000000e+00, 0.000000e+00, 0.000000e+00,
            -1.269991e+00, 0.000000e+00, 1.500000e+00,
            0.000000e+00, 0.000000e+00, -7.401487e-17,
            1.000000e+00, 7.000000e+00, 2.449490e+00,
            4.000000e+00, 5.000000e+01, 5.000000e+01,
            1.000000e-03, 5.000000e+01, 7.300000e+02,
            2.380000e+03, 7.618545e+02, 1.806667e+03
        ])

        # Run
        parameters = self.modeler._create_extension(user, child_table)

        # Check
        assert expected.subtract(parameters).all() < 10E-3

    def test_get_extensions(self):
        """Tests that get extensions works for table with child"""
        # Setup
        pk = 'ORDER_ID'
        table = 'DEMO_ORDERS'
        children = self.dn.get_children(table)

        # Run
        result = self.modeler._get_extensions(pk, children, table)

        # Check
        assert len(result) == 1
        assert result[0].shape == (10, 50)

    def test_get_extensions_no_children(self):
        """Tests that get extensions works for table with no children."""
        # Setup
        pk = 'ORDER_ITEM_ID'
        table = 'DEMO_ORDER_ITEMS'
        children = self.dn.get_children(table)
        expected_result = []

        # Run
        result = self.modeler._get_extensions(pk, children, table)

        # Check
        assert result == expected_result

    def test_CPA(self):
        """ """
        # Setup
        self.modeler.model_database()
        table_name = 'DEMO_CUSTOMERS'

        # Run
        self.modeler.CPA(table_name)

        # Check
        result = self.modeler.tables[table_name]

        # When we run Conditional Parameter Aggregation we add a key on Modeler.tables
        # It contains a not null pandas DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape != (0, 0)
