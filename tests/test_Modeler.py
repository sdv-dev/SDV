from unittest import TestCase, skipIf
import pandas as pd

from sdv.DataNavigator import CSVDataLoader
from sdv.Modeler import Modeler


class ModelerTest(TestCase):

    def setUp(self):
        """Set up test fixtures, if any."""
        dl = CSVDataLoader('tests/manual_data/meta.json')
        self.dn = dl.load_data()
        self.dn.transform_data()
        self.Modeler = Modeler(
            self.dn, model_type='GaussianMultivariate',
            model_params=['GaussianUnivariate'])

    def test_create_extension(self):
        """Tests that the create extension method returns correct parameters"""
        child_table = self.dn.get_data('DEMO_ORDERS')
        user = child_table[child_table['CUSTOMER_ID'] == 50]
        transformer_child_table = self.dn.transformed_data['DEMO_ORDERS']
        parameters = self.Modeler._create_extension(
            user, 'DEMO_ORDERS', child_table)

        expected = pd.Series([1.500000e+00, 0.000000e+00, -1.269991e+00,
                              0.000000e+00, 0.000000e+00, 0.000000e+00,
                              -1.269991e+00, 0.000000e+00, 1.500000e+00,
                              0.000000e+00, 0.000000e+00, -7.401487e-17,
                              1.000000e+00, 7.000000e+00, 2.449490e+00,
                              4.000000e+00, 5.000000e+01, 5.000000e+01,
                              1.000000e-03, 5.000000e+01, 7.300000e+02,
                              2.380000e+03, 7.618545e+02, 1.806667e+03])
        assert expected.subtract(parameters).all() < 10E-3

    def test_get_extensions(self):
        """Tests that get extensions works for table with child"""
        pk = 'ORDER_ID'
        table = 'DEMO_ORDERS'
        children = self.dn.get_children(table)
        result = self.Modeler._get_extensions(pk, children, table)
        # expected dimensions of output
        expected_len = 1
        expected_num_colums = 50
        expected_num_rows = 10

        assert len(result) == 1
        assert result[0].shape == (expected_num_rows, expected_num_colums)

    def test_get_extensions_no_children(self):
        """Tests that get extensions works for table with no children"""
        pk = 'ORDER_ITEM_ID'
        table = 'DEMO_ORDER_ITEMS'
        children = self.dn.get_children(table)
        result = self.Modeler._get_extensions(pk, children, table)
        # should be empty
        expected = []

        assert result == expected
