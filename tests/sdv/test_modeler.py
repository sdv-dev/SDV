from collections import OrderedDict
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from copulas import EPSILON
from copulas.multivariate import GaussianMultivariate, VineCopula
from copulas.univariate import KDEUnivariate

from sdv.data_navigator import CSVDataLoader, DataNavigator, HyperTransformer, Table
from sdv.modeler import Modeler


class TestModeler(TestCase):

    def setUp(self):
        """Set up test fixtures, if any."""
        dl = CSVDataLoader('tests/data/meta.json')
        self.dn = dl.load_data()
        self.dn.transform_data()
        self.modeler = Modeler(self.dn)

    @patch('sdv.modeler.Modeler._get_model_dict')
    @patch('sdv.modeler.Modeler.impute_table')
    def test__create_extension(self, impute_mock, model_mock):
        """Tests that the create extension method returns correct parameters."""
        # Setup
        data_navigator = MagicMock()
        modeler = Modeler(data_navigator)
        table = pd.DataFrame({
            'foreign': [0, 1, 0, 1, 0, 1],
            'a': [0, 1, 0, 1, 0, 1],
            'b': [1, 2, 3, 4, 5, 6]
        })
        foreign = table[table.a == 0]
        table_info = ('foreign', 'child')

        impute_mock.return_value = pd.DataFrame({
            'A': [1, 2],
            'B': [3, 4]
        })

        model_mock.return_value = pd.Series({
            'covariance__0__0': 0.0,
            'covariance__1__0': 0.0,
            'covariance__1__1': 1.4999999999999991,
            'distribs__a__mean': 0.0,
            'distribs__a__std': 0.001,
            'distribs__b__mean': 3.0,
            'distribs__b__std': 1.632993161855452
        })

        # Run
        result = modeler._create_extension(foreign, table, table_info)

        # Check
        assert result.equals(model_mock.return_value)

        df = pd.DataFrame({
            'a': [0, 1, 0, 1, 0, 1],
            'b': [1, 2, 3, 4, 5, 6]
        })
        df = df.loc[foreign.index]

        assert len(impute_mock.call_args_list)
        call_args = impute_mock.call_args_list[0]
        assert len(call_args[0]) == 1
        assert call_args[0][0].equals(df)
        assert call_args[1] == {}

        model_mock.assert_called_once_with(impute_mock.return_value)

    def test__create_extension_wrong_index_return_none(self):
        """_create_extension return None if transformed_child_table can't be indexed by df."""
        # Setup
        data_navigator = MagicMock()
        modeler = Modeler(data_navigator)
        transformed_child_table = pd.DataFrame(np.eye(3), columns=['A', 'B', 'C'])
        table_info = ('', '')
        df = pd.DataFrame(index=range(5, 10))

        # Run
        result = modeler._create_extension(df, transformed_child_table, table_info)

        # Check
        assert result is None

    @patch('sdv.modeler.Modeler._create_extension')
    @patch('sdv.modeler.Modeler.get_foreign_key')
    def test__get_extensions(self, get_foreign_mock, extension_mock):
        """_get_extensions return the conditional modelling parameters for each children."""
        # Setup
        data_navigator = MagicMock()

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

    @patch('sdv.modeler.pd.DataFrame.merge', autospec=True)
    @patch('sdv.modeler.Modeler._get_extensions', autospec=True)
    def test_CPA(self, extensions_mock, merge_mock):
        """CPA will append extensions to the original table."""
        # Setup
        data_navigator = MagicMock(spec=DataNavigator)
        table = Table(
            pd.DataFrame({'table_pk': range(5)}),
            {'primary_key': 'table_pk'}
        )
        data_navigator.tables = {
            'table': table
        }

        transformed_table = pd.DataFrame({'table_pk': range(5)})
        data_navigator.transformed_data = {
            'table': transformed_table
        }

        data_navigator.get_children.return_value = 'children of table'
        modeler = Modeler(data_navigator)

        extension = MagicMock()
        extensions_mock.return_value = [extension]
        extended_table = MagicMock()
        merge_mock.return_value = extended_table

        table_name = 'table'

        # Run
        modeler.CPA(table_name)

        # Check
        assert modeler.tables[table_name] == extended_table

        extensions_mock.assert_called_once_with(modeler, 'table_pk', 'children of table')
        merge_mock.assert_called_once_with(
            transformed_table, extension.reset_index.return_value, how='left', on='table_pk')

        data_navigator.get_children.assert_called_once_with('table')
        extension.reset_index.assert_called_once_with()
        extended_table.drop.assert_not_called()
        call_args_list = extended_table.__setitem__.call_args_list
        assert len(call_args_list) == 1
        args, kwargs = call_args_list[0]
        assert kwargs == {}
        assert len(args) == 2
        assert args[0] == 'table_pk'
        assert args[1].equals(transformed_table['table_pk'])

    @patch('sdv.modeler.Modeler._get_extensions')
    def test_CPA_transformed_index(self, extension_mock):
        """CPA is able to merge extensions in tables with transformed index. """
        # Setup
        data_navigator = MagicMock(spec=DataNavigator)
        modeler = Modeler(data_navigator)

        # Setup - Mock
        parent_data = pd.DataFrame([
            {'parent_id': 'A', 'values': 1},
            {'parent_id': 'B', 'values': 2},
            {'parent_id': 'C', 'values': 3},
        ])
        parent_meta = {
            'name': 'parent',
            'primary_key': 'parent_id',
            'fields': {
                'parent_id': {
                    'name': 'parent_id',
                    'type': 'categorical',
                    'regex': '^[A-Z]$'
                },
                'values': {
                    'name': 'values',
                    'type': 'number',
                    'subtype': 'integer'
                }
            }
        }

        child_data = pd.DataFrame([
            {'child_id': 1, 'parent_id': 'A', 'value': 0.1},
            {'child_id': 2, 'parent_id': 'A', 'value': 0.2},
            {'child_id': 3, 'parent_id': 'A', 'value': 0.3},
            {'child_id': 4, 'parent_id': 'B', 'value': 0.4},
            {'child_id': 5, 'parent_id': 'B', 'value': 0.5},
            {'child_id': 6, 'parent_id': 'B', 'value': 0.6},
            {'child_id': 7, 'parent_id': 'C', 'value': 0.7},
            {'child_id': 8, 'parent_id': 'C', 'value': 0.8},
            {'child_id': 9, 'parent_id': 'C', 'value': 0.9},
        ])
        child_meta = {
            'name': 'child',
            'primary_key': 'child_id',
            'fields': {
                'child_id': {
                    'name': 'child_id',
                    'type': 'number'
                },
                'parent_id': {
                    'name': 'parent_id',
                    'type': 'category',
                    'ref': {
                        'table': 'parent',
                        'field': 'parent_id'
                    }
                },
                'value': {
                    'name': 'value',
                    'type': 'number'
                }
            }
        }

        data_navigator.tables = {
            'parent': Table(parent_data, parent_meta),
            'child': Table(child_data, child_meta)
        }

        children_map = {'parent': {'child'}}
        parent_map = {'child': {'parent'}}

        data_navigator.get_children.side_effect = lambda x: children_map.get(x, set())
        data_navigator.get_parents.side_effect = lambda x: parent_map.get(x, set())

        transformed_parent = pd.DataFrame([
            {'parent_id': 0.1, 'values': 1},
            {'parent_id': 0.4, 'values': 2},
            {'parent_id': 0.8, 'values': 3},
        ])
        transformed_child = pd.DataFrame([
            {'child_id': 1, 'parent_id': 0.15, 'value': 0.1},
            {'child_id': 2, 'parent_id': 0.10, 'value': 0.2},
            {'child_id': 3, 'parent_id': 0.20, 'value': 0.3},
            {'child_id': 4, 'parent_id': 0.35, 'value': 0.4},
            {'child_id': 5, 'parent_id': 0.50, 'value': 0.5},
            {'child_id': 6, 'parent_id': 0.55, 'value': 0.6},
            {'child_id': 7, 'parent_id': 0.70, 'value': 0.7},
            {'child_id': 8, 'parent_id': 0.80, 'value': 0.8},
            {'child_id': 9, 'parent_id': 0.85, 'value': 0.9},
        ])

        data_navigator.transformed_data = {
            'parent': transformed_parent,
            'child': transformed_child
        }
        extension = pd.DataFrame(**{
            'data': [
                {'param_1': 0.5, 'param_2': 0.4},
                {'param_1': 0.7, 'param_2': 0.2},
                {'param_1': 0.2, 'param_2': 0.1},
            ],
            'index': list('ABC')
        })
        extension.index.name = 'parent_id'
        extension_mock.return_value = [extension]

        expected_extended_parent = pd.DataFrame(
            [
                {'parent_id': 0.1, 'values': 1, 'param_1': 0.5, 'param_2': 0.4},
                {'parent_id': 0.4, 'values': 2, 'param_1': 0.7, 'param_2': 0.2},
                {'parent_id': 0.8, 'values': 3, 'param_1': 0.2, 'param_2': 0.1},
            ],
            columns=['parent_id', 'values', 'param_1', 'param_2']
        )

        # Run
        modeler.CPA('parent')

        # Check
        'parent' in modeler.tables
        assert modeler.tables['parent'].equals(expected_extended_parent)

        data_navigator.get_children.assert_called_once_with('parent')
        extension_mock.assert_called_once_with('parent_id', {'child'})

    def test__get_model_dict(self):
        """_get_model_dict returns a pandas.Series with all the params to recreate a model."""
        # Setup
        X = np.eye(3)

        expected_result = {
            'covariance__0__0': 1.5000000000000009,
            'covariance__1__0': -0.7500000000000003,
            'covariance__1__1': 1.5000000000000009,
            'covariance__2__0': -0.7500000000000003,
            'covariance__2__1': -0.7500000000000003,
            'covariance__2__2': 1.5000000000000007,
            'distribs__0__mean': 0.33333333333333331,
            'distribs__0__std': -0.7520386983881371,
            'distribs__1__mean': 0.33333333333333331,
            'distribs__1__std': -0.7520386983881371,
            'distribs__2__mean': 0.33333333333333331,
            'distribs__2__std': -0.7520386983881371,
        }
        data_navigator = MagicMock()
        modeler = Modeler(data_navigator)

        # Run
        result = modeler._get_model_dict(X)

        # Check
        assert result == expected_result

    def test_impute_table_with_mean(self):
        """impute_table fills all NaN values the mean of values when possible."""
        # Setup
        table = pd.DataFrame([
            {'A': np.nan, 'B': 2., 'C': 4.},
            {'A': 4., 'B': np.nan, 'C': 2.},
            {'A': 2., 'B': 4., 'C': np.nan},
        ])
        expected_result = pd.DataFrame([
            {'A': 3., 'B': 2., 'C': 4.},
            {'A': 4., 'B': 3., 'C': 2.},
            {'A': 2., 'B': 4., 'C': 3.},
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

    def test_impute_table_with_mean_default(self):
        """If a column only has NaN, impute_table fills it with 0.(+EPSILON).

        If a column has no mean (all values are null), then the NaN values are replaced with 0.
        Then, it will transform like a constant column, adding copulas.EPSILON at the
        first element.
        """
        # Setup
        table = pd.DataFrame([
            {'A': np.nan, 'B': 2., 'C': 2.},
            {'A': np.nan, 'B': 3., 'C': 3.},
            {'A': np.nan, 'B': 4., 'C': 4.},
        ])
        expected_result = pd.DataFrame([
            {'A': EPSILON, 'B': 2., 'C': 2.},
            {'A': 0., 'B': 3., 'C': 3.},
            {'A': 0., 'B': 4., 'C': 4.},
        ])

        # Run
        result = self.modeler.impute_table(table)

        # Check
        assert result.equals(expected_result)

        # No null values are left
        assert not result.isnull().all().all()

    def test_impute_table_constant_column(self):
        """impute_table adds EPSILON at the first element of a constant column."""
        # Setup
        table = pd.DataFrame([
            {'A': np.nan, 'B': 10., 'C': 20.},
            {'A': 5., 'B': np.nan, 'C': 20.},
            {'A': 5., 'B': 10., 'C': np.nan},
        ])
        expected_result = pd.DataFrame([
            {'A': 5. + EPSILON, 'B': 10. + EPSILON, 'C': 20. + EPSILON},
            {'A': 5., 'B': 10., 'C': 20.},
            {'A': 5., 'B': 10., 'C': 20.},
        ])

        # Run
        result = self.modeler.impute_table(table)

        # Check
        assert result.equals(expected_result)

        # No null values are left
        assert not result.isnull().all().all()

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
        model_mock = MagicMock()
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

    @patch('sdv.modeler.Modeler.fit_model')
    @patch('sdv.modeler.Modeler.impute_table')
    @patch('sdv.modeler.Modeler.RCPA')
    def test_model_database(self, rcpa_mock, impute_mock, fit_mock):
        """model_database computes conditions between tables and models them."""
        # Setup
        data_navigator = MagicMock(spec=DataNavigator)
        modeler = Modeler(data_navigator)

        data_navigator.tables = OrderedDict()
        data_navigator.tables['table_A'] = 'table_A_dataframe'
        data_navigator.tables['table_B'] = 'table_B_dataframe'
        data_navigator.tables['table_C'] = 'table_C_dataframe'

        parents = {
            'table_A': {},
            'table_B': {'table_A'},
            'table_C': {'table_B'}
        }
        data_navigator.get_parents.side_effect = lambda x: parents[x]

        def rcpa_side_effect(*args):
            modeler.tables = data_navigator.tables

        rcpa_mock.side_effect = rcpa_side_effect
        impute_mock.side_effect = ['TABLE_A', 'TABLE_B', 'TABLE_C']
        fit_mock.side_effect = lambda x: 'model_for_{}'.format(x)

        # Run
        modeler.model_database()

        # Check
        assert data_navigator.get_parents.call_args_list == [
            (('table_A',), ),
            (('table_B',), ),
            (('table_C',), ),
        ]

        rcpa_mock.assert_called_once_with('table_A')
        assert impute_mock.call_args_list == [
            (('table_A_dataframe', ), ),
            (('table_B_dataframe', ), ),
            (('table_C_dataframe', ), ),
        ]

        assert fit_mock.call_args_list == [
            (('TABLE_A', ), ),
            (('TABLE_B', ), ),
            (('TABLE_C', ), ),
        ]

        assert modeler.models == {
            'table_A': 'model_for_TABLE_A',
            'table_B': 'model_for_TABLE_B',
            'table_C': 'model_for_TABLE_C'
        }

    def test_model_database_gaussian_copula_single_table(self):
        """model_database can model a single table using the gausian copula model."""
        # Setup
        data_navigator = MagicMock(spec=DataNavigator)
        modeler = Modeler(data_navigator=data_navigator, model=GaussianMultivariate)

        # Setup - Mocks - DataNavigator
        table_data = pd.DataFrame({
            'column_A': list('abdc'),
            'column_B': range(4)
        })
        table_metadata = {
            'name': 'table_name',
            'fields': {
                'column_A': {
                    'name': 'column_A',
                    'type': 'categorical'
                },
                'column_B': {
                    'name': 'column_B',
                    'type': 'number',
                    'subtype': 'integer'
                }
            }
        }

        data_navigator.tables = {
            'table_name': Table(table_data, table_metadata)
        }
        data_navigator.get_parents.return_value = set()
        data_navigator.get_children.return_value = set()
        data_navigator.transformed_data = {
            'table_name': pd.DataFrame({
                'column_A': [0.1, 0.2, 0.5, 1.0],
                'column_B': range(4)
            })
        }
        metadata = {
            'name': 'table_name',
            'fields': [
                {
                    'name': 'column_A',
                    'type': 'categorical'
                },
                {
                    'name': 'column_B',
                    'type': 'number',
                    'subtype': 'integer'
                }
            ]
        }

        data_navigator.meta = {
            'tables': [metadata]
        }
        ht = MagicMock(spec=HyperTransformer)
        ht.transformers = {
            ('table_name', 'column_A'): None,
            ('table_name', 'column_B'): None
        }

        reverse_transform_dataframe = pd.DataFrame({
            'column_A': list('bcda'),
            'column_B': [1.0, 2.0, 3.0, 4.0]
        }, columns=['column_A', 'column_B'])
        ht.reverse_transform_table.return_value = reverse_transform_dataframe

        data_navigator.ht = ht

        # Run
        modeler.model_database()

        # Check
        assert len(modeler.models) == 1
        assert 'table_name' in modeler.models
        model = modeler.models['table_name']

        assert isinstance(model, GaussianMultivariate)
        assert model.distribution == 'copulas.univariate.gaussian.GaussianUnivariate'
        assert model.fitted is True

        assert data_navigator.get_parents.call_args_list == [(('table_name', ), )]
        assert data_navigator.get_children.call_args_list == [
            (('table_name', ), ),
            (('table_name', ), )
        ]
        assert modeler.tables['table_name'].equals(modeler.dn.transformed_data['table_name'])

    def test_model_database_kde_distribution(self):
        """model_database works fine with kde distribution."""
        # Setup
        data_navigator = MagicMock(spec=DataNavigator)
        modeler = Modeler(data_navigator=data_navigator, distribution=KDEUnivariate)

        # Setup - Mocks - DataNavigator
        table_data = pd.DataFrame({
            'column_A': list('abdc'),
            'column_B': range(4)
        })
        table_metadata = {
            'name': 'table_name',
            'fields': {
                'column_A': {
                    'name': 'column_A',
                    'type': 'categorical'
                },
                'column_B': {
                    'name': 'column_B',
                    'type': 'number',
                    'subtype': 'integer'
                }
            }
        }

        data_navigator.tables = {
            'table_name': Table(table_data, table_metadata)
        }
        data_navigator.get_parents.return_value = set()
        data_navigator.get_children.return_value = set()
        data_navigator.transformed_data = {
            'table_name': pd.DataFrame({
                'column_A': [0.1, 0.2, 0.5, 1.0],
                'column_B': range(4)
            })
        }
        metadata = {
            'name': 'table_name',
            'fields': [
                {
                    'name': 'column_A',
                    'type': 'categorical'
                },
                {
                    'name': 'column_B',
                    'type': 'number',
                    'subtype': 'integer'
                }
            ]
        }

        data_navigator.meta = {
            'tables': [metadata]
        }
        ht = MagicMock(spec=HyperTransformer)
        ht.transformers = {
            ('table_name', 'column_A'): None,
            ('table_name', 'column_B'): None
        }

        reverse_transform_dataframe = pd.DataFrame({
            'column_A': list('bcda'),
            'column_B': [1.0, 2.0, 3.0, 4.0]
        }, columns=['column_A', 'column_B'])
        ht.reverse_transform_table.return_value = reverse_transform_dataframe

        data_navigator.ht = ht

        # Run
        modeler.model_database()

        # Check
        assert len(modeler.models) == 1
        assert 'table_name' in modeler.models
        model = modeler.models['table_name']

        assert isinstance(model, GaussianMultivariate)
        assert model.distribution == 'copulas.univariate.kde.KDEUnivariate'
        assert model.fitted is True

        assert data_navigator.get_parents.call_args_list == [(('table_name', ), )]
        assert data_navigator.get_children.call_args_list == [
            (('table_name', ), ),
            (('table_name', ), )
        ]

    def test_model_database_vine_modeler_single_table(self):
        """model_database works fine with vine modeler."""
        # Setup
        data_navigator = MagicMock(spec=DataNavigator)
        modeler = Modeler(data_navigator=data_navigator, model=VineCopula)

        # Setup - Mock
        data = pd.DataFrame({
            'column_A': list('abdc'),
            'column_B': range(4)
        })
        meta = {
            'name': 'table_name',
            'fields': {
                'column_A': {
                    'name': 'column_A',
                    'type': 'categorical'
                },
                'column_B': {
                    'name': 'column_B',
                    'type': 'number',
                    'subtype': 'integer'
                }
            }
        }

        data_navigator.tables = {
            'table_name': Table(data, meta)
        }
        data_navigator.get_parents.return_value = set()
        data_navigator.get_children.return_value = set()
        data_navigator.transformed_data = {
            'table_name': pd.DataFrame({
                'column_A': [0.1, 0.2, 0.5, 1.0],
                'column_B': range(4)
            })
        }
        metadata = {
            'name': 'table_name',
            'fields': [
                {
                    'name': 'column_A',
                    'type': 'categorical'
                },
                {
                    'name': 'column_B',
                    'type': 'number',
                    'subtype': 'integer'
                }
            ]
        }

        data_navigator.meta = {
            'tables': [metadata]
        }

        ht = MagicMock(spec=HyperTransformer)
        ht.transformers = {
            ('table_name', 'column_A'): None,
            ('table_name', 'column_B'): None
        }

        reverse_transform_dataframe = pd.DataFrame({
            'column_A': list('bcda'),
            'column_B': [1.0, 2.0, 3.0, 4.0]
        }, columns=['column_A', 'column_B'])
        ht.reverse_transform_table.return_value = reverse_transform_dataframe

        data_navigator.ht = ht

        # Run
        modeler.model_database()

        # Check
        assert len(modeler.models) == 1
        model = modeler.models['table_name']
        assert isinstance(model, VineCopula)
        assert model.fitted is True

        assert data_navigator.get_parents.call_args_list == [(('table_name', ), )]
        assert data_navigator.get_children.call_args_list == [
            (('table_name', ), ),
            (('table_name', ), )
        ]
        assert modeler.tables['table_name'].equals(modeler.dn.transformed_data['table_name'])

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

    def test__flatten_dict_missing_keys_gh_89(self):
        """flatten_dict will only ignore keys that don't have dict or list values.

        https://github.com/HDI-Project/SDV/issues/89
        """
        # Setup
        nested_dict = {
            'covariance': [
                [1.4999999999999991, 1.4999999999999991, 1.4999999999999991],
                [1.4999999999999991, 1.4999999999999991, 1.4999999999999991],
                [1.4999999999999991, 1.4999999999999991, 1.4999999999999991]],
            'distribs': {
                'type': {
                    'type': 'copulas.univariate.gaussian.GaussianUnivariate',
                    'fitted': True,
                    'mean': 4.0,
                    'std': 2.449489742783178
                },
                'distribution': {
                    'type': 'copulas.univariate.gaussian.GaussianUnivariate',
                    'fitted': True,
                    'mean': 5.0,
                    'std': 2.449489742783178
                },
                'fitted': {
                    'type': 'copulas.univariate.gaussian.GaussianUnivariate',
                    'fitted': True,
                    'mean': 6.0,
                    'std': 2.449489742783178
                }
            },
            'type': 'copulas.multivariate.gaussian.GaussianMultivariate',
            'fitted': True,
            'distribution': 'copulas.univariate.gaussian.GaussianUnivariate'
        }
        expected_result = {
            'covariance__0__0': 1.4999999999999991,
            'covariance__0__1': 1.4999999999999991,
            'covariance__0__2': 1.4999999999999991,
            'covariance__1__0': 1.4999999999999991,
            'covariance__1__1': 1.4999999999999991,
            'covariance__1__2': 1.4999999999999991,
            'covariance__2__0': 1.4999999999999991,
            'covariance__2__1': 1.4999999999999991,
            'covariance__2__2': 1.4999999999999991,
            'distribs__type__mean': 4.0,
            'distribs__type__std': 2.449489742783178,
            'distribs__distribution__mean': 5.0,
            'distribs__distribution__std': 2.449489742783178,
            'distribs__fitted__mean': 6.0,
            'distribs__fitted__std': 2.449489742783178
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
