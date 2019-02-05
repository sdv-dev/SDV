from unittest import TestCase, mock

import numpy as np
import pandas as pd
from copulas.multivariate import GaussianMultivariate, VineCopula
from copulas.univariate.kde import KDEUnivariate

from sdv.data_navigator import CSVDataLoader, Table
from sdv.modeler import Modeler


class ModelerTest(TestCase):

    def setUp(self):
        """Set up test fixtures, if any."""
        dl = CSVDataLoader('tests/data/meta.json')
        self.dn = dl.load_data()
        self.dn.transform_data()
        self.modeler = Modeler(self.dn)

    def test__create_extension(self):
        """Tests that the create extension method returns correct parameters."""
        # Setup
        data_navigator = mock.MagicMock()
        modeler = Modeler(data_navigator)
        table = pd.DataFrame({
            'foreign': [0, 1, 0, 1, 0, 1],
            'a': [0, 1, 0, 1, 0, 1],
            'b': [1, 2, 3, 4, 5, 6]
        })
        group = table[table.a == 0]
        table_info = ('foreign', '')

        expected_result = pd.Series({
            'covariance__0__0': 0.0,
            'covariance__0__1': 0.0,
            'covariance__1__0': 0.0,
            'covariance__1__1': 1.4999999999999991,
            'distribs__a__mean': 0.0,
            'distribs__a__std': 0.001,
            'distribs__b__mean': 3.0,
            'distribs__b__std': 1.632993161855452
        })

        # Run
        result = modeler._create_extension(group, table, table_info)

        # Check
        assert result.equals(expected_result)

    def test__create_extension_wrong_index_return_none(self):
        """_create_extension return None if transformed_child_table can't be indexed by df."""
        # Setup
        data_navigator = mock.MagicMock()
        modeler = Modeler(data_navigator)
        transformed_child_table = pd.DataFrame(np.eye(3), columns=['A', 'B', 'C'])
        table_info = ('', '')
        df = pd.DataFrame(index=range(5, 10))

        # Run
        result = modeler._create_extension(df, transformed_child_table, table_info)

        # Check
        assert result is None

    @mock.patch('sdv.modeler.Modeler._create_extension')
    @mock.patch('sdv.modeler.Modeler.get_foreign_key')
    def test__get_extensions(self, get_foreign_mock, extension_mock):
        """_get_extensions return the conditional modelling parameters for each children."""
        # Setup
        data_navigator = mock.MagicMock()

        first_table_data = pd.DataFrame({'foreign_key': [0, 1]})
        first_table_meta = {'fields': []}

        data_navigator.tables = {
            'first_children': Table(first_table_data, first_table_meta),
            'second_children': Table(first_table_data, first_table_meta),
        }
        data_navigator.get_children.return_value = {}
        modeler = Modeler(data_navigator)
        modeler.tables = {}

        extension_mock.side_effect = lambda x, y, z: None

        get_foreign_mock.return_value = 'foreign_key'

        pk = 'primary_key'
        children = ['first_children', 'second_children']

        expected_result = [
            pd.DataFrame([{
                '__first_children_column_1': 1,
                '__first_children_column_2': 2
            }]),
            pd.DataFrame([{
                '__second_children_column_1': 1,
                '__second_children_column_2': 2
            }])
        ]

        # Run
        result = modeler._get_extensions(pk, children)

        # Check
        assert all([result[index].equals(expected_result[index]) for index in range(len(result))])

    def test_get_extensions_no_children(self):
        """_get_extensions return an empty list if children is empty."""
        # Setup
        pk = 'primary_key'
        children = {}

        expected_result = []

        # Run
        result = self.modeler._get_extensions(pk, children)

        # Check
        assert result == expected_result

    def test_CPA(self):
        """CPA will append extensions to the original table."""
        # Setup
        self.modeler.model_database()
        table_name = 'DEMO_CUSTOMERS'

        # Run
        self.modeler.CPA(table_name)

        # Check
        for name, table in self.modeler.tables.items():
            with self.subTest(table=name):
                raw_table = self.modeler.dn.tables[name].data

                # When we run Conditional Parameter Aggregation we add a key on Modeler.tables
                # for each table. It contains a not null pandas DataFrame with the computed
                # extension.
                assert isinstance(table, pd.DataFrame)

                assert raw_table.shape[0] == table.shape[0]
                assert (raw_table.index == table.index).all()
                assert all([column in table.columns for column in raw_table.columns])

    def test_flatten_model(self):
        """flatten_model returns a pandas.Series with all the params to recreate a model."""
        # Setup
        model = GaussianMultivariate()
        X = np.eye(3)
        model.fit(X)

        expected_result = pd.Series({
            'covariance__0__0': 1.5000000000000004,
            'covariance__0__1': -0.7500000000000003,
            'covariance__0__2': -0.7500000000000003,
            'covariance__1__0': -0.7500000000000003,
            'covariance__1__1': 1.5000000000000004,
            'covariance__1__2': -0.7500000000000003,
            'covariance__2__0': -0.7500000000000003,
            'covariance__2__1': -0.7500000000000003,
            'covariance__2__2': 1.5000000000000007,
            'distribs__0__mean': 0.33333333333333331,
            'distribs__0__std': 0.47140452079103168,
            'distribs__1__mean': 0.33333333333333331,
            'distribs__1__std': 0.47140452079103168,
            'distribs__2__mean': 0.33333333333333331,
            'distribs__2__std': 0.47140452079103168
        })

        # Run
        result = Modeler.flatten_model(model)

        # Check
        assert np.isclose(result, expected_result).all()

    def test_impute_table(self):
        """impute_table fills all NaN values with 0 or the mean of values."""
        # Setup
        table = pd.DataFrame([
            {'A': np.nan, 'B': 10., 'C': 20.},
            {'A': 5., 'B': np.nan, 'C': 20.},
            {'A': 5., 'B': 10., 'C': np.nan},
        ])
        expected_result = pd.DataFrame([
            {'A': 5., 'B': 10., 'C': 20.},
            {'A': 5., 'B': 10., 'C': 20.},
            {'A': 5., 'B': 10., 'C': 20.},
        ])

        # Run
        result = self.modeler.impute_table(table)

        # Check
        assert result.equals(expected_result)

        # No null values are left
        assert not result.isnull().all().all()

        # Averages are computed on every column
        for column in result:
            assert 0 not in result[column].values

    def test_model_database(self):
        """model_database computes conditions between tables and models them."""

        # Run
        self.modeler.model_database()

        # Check
        assert self.modeler.tables.keys() == self.modeler.models.keys()

    def test_get_foreign_key(self):
        """get_foreign_key returns the foreign key from a metadata and a primary key."""
        # Setup
        fields = self.modeler.dn.get_meta_data('DEMO_ORDERS')['fields']
        primary = 'CUSTOMER_ID'
        expected_result = 'CUSTOMER_ID'

        # Run
        result = self.modeler.get_foreign_key(fields, primary)

        # Check
        assert result == expected_result

    def test_fit_model_distribution_arg(self):
        """fit_model will pass self.distribution FQN to modeler."""
        # Setup
        model_mock = mock.MagicMock()
        model_mock.__eq__.return_value = True
        model_mock.__ne__.return_value = False
        modeler = Modeler(data_navigator='navigator', model=model_mock, distribution=KDEUnivariate)
        data = pd.DataFrame({
            'column': [0, 1, 1, 1, 0],
        })

        # Run
        modeler.fit_model(data)

        # Check
        model_mock.assert_called_once_with(distribution='copulas.univariate.kde.KDEUnivariate')

    def test_model_database_kde_distribution(self):
        """model_database works fine with kde distribution."""
        # Setup
        modeler = Modeler(data_navigator=self.dn, distribution=KDEUnivariate)

        # Run
        modeler.model_database()

    def test_model_database_vine_modeler(self):
        """model_database works fine with vine modeler."""
        # Setup
        modeler = Modeler(data_navigator=self.dn, model=VineCopula)

        # Run
        modeler.model_database()

    def test__flatten_dict_flat_dict(self):
        """_flatten_dict don't modify flat dicts."""
        # Setup
        nested_dict = {
            'a': 1,
            'b': 2
        }
        expected_result = {
            'a': 1,
            'b': 2
        }

        # Run
        result = Modeler._flatten_dict(nested_dict)

        # Check
        assert result == expected_result

    def test__flatten_dict_nested_dict(self):
        """_flatten_dict flatten nested dicts respecting the prefixes."""
        # Setup
        nested_dict = {
            'first_key': {
                'a': 1,
                'b': 2
            },
            'second_key': {
                'x': 0
            }
        }

        expected_result = {
            'first_key__a': 1,
            'first_key__b': 2,
            'second_key__x': 0
        }

        # Run
        result = Modeler._flatten_dict(nested_dict)

        # Check
        assert result == expected_result

    def test__flatten_array_ndarray(self):
        """_flatten_array return a dict formed from the input np.array"""
        # Setup
        nested = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        expected_result = {
            '0__0': 1,
            '0__1': 0,
            '0__2': 0,
            '1__0': 0,
            '1__1': 1,
            '1__2': 0,
            '2__0': 0,
            '2__1': 0,
            '2__2': 1
        }

        # Run
        result = Modeler._flatten_array(nested)

        # Check
        assert result == expected_result

    def test__flatten_array_list(self):
        """_flatten_array return a dict formed from the input list"""
        # Setup
        nested = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
        expected_result = {
            '0__0': 1,
            '0__1': 0,
            '0__2': 0,
            '1__0': 0,
            '1__1': 1,
            '1__2': 0,
            '2__0': 0,
            '2__1': 0,
            '2__2': 1
        }

        # Run
        result = Modeler._flatten_array(nested)

        # Check
        assert result == expected_result
