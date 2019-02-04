from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd

from sdv.data_navigator import CSVDataLoader, Table
from sdv.modeler import Modeler
from sdv.sampler import Sampler


class TestSampler(TestCase):

    @classmethod
    def setUpClass(cls):
        data_loader = CSVDataLoader('tests/data/meta.json')
        cls.data_navigator = data_loader.load_data()
        cls.data_navigator.transform_data()

        cls.modeler = Modeler(cls.data_navigator)
        cls.modeler.model_database()

    def setUp(self):
        self.sampler = Sampler(self.data_navigator, self.modeler)

    def test__rescale_values(self):
        """_rescale_values return and array satisfying  0 < array < 1."""
        # Setup
        data_navigator = MagicMock()
        modeler = MagicMock()
        sampler = Sampler(data_navigator, modeler)

        column = pd.Series([0.0, 5.0, 10], name='column')
        expected_result = pd.Series([0.0, 0.5, 1.0], name='column')

        # Run
        result = sampler._rescale_values(column)

        # Check
        assert (result == expected_result).all().all()
        assert len(data_navigator.call_args_list) == 0
        assert len(modeler.call_args_list) == 0

    @patch('sdv.sampler.Sampler._get_table_meta')
    def test_transform_synthesized_rows(self, get_table_meta_mock):
        """t_s_r will add the primary key and reverse transform rows."""
        # Setup
        data_navigator = MagicMock()

        table_metadata = {
            'fields': {
                'id': {
                    'regex': '[0-9]{5}',
                    'type': 'number',
                    'subtype': 'integer'

                }
            },
            'primary_key': 'id',
        }
        table_data = pd.DataFrame()
        test_table = Table(table_data, table_metadata)
        data_navigator.tables = {
            'table': test_table
        }
        modeler = MagicMock()
        sampler = Sampler(data_navigator, modeler)

        synthesized_rows = pd.DataFrame({

        })
        table_name = 'table'
        num_rows = 2

        expected_result = pd.DataFrame({

        })

        # Run
        result = sampler.transform_synthesized_rows(synthesized_rows, table_name, num_rows)

        # Check
        assert result.equals(expected_result)

    def test_sample_rows_parent_table(self):
        """sample_rows samples new rows for the given table."""
        # Setup
        raw_data = self.modeler.dn.tables['DEMO_CUSTOMERS'].data

        # Run
        result = self.sampler.sample_rows('DEMO_CUSTOMERS', 5)

        # Check
        assert result.shape[0] == 5
        assert (result.columns == raw_data.columns).all()

        # Primary key columns are sampled values
        assert len(result['CUSTOMER_ID'].unique()) != 1

    def test_sample_rows_children_table(self):
        """sample_rows samples new rows for the given table."""
        # Setup
        raw_data = self.modeler.dn.tables['DEMO_ORDERS'].data
        # Sampling parent table.
        self.sampler.sample_rows('DEMO_CUSTOMERS', 5)

        # Run
        result = self.sampler.sample_rows('DEMO_ORDERS', 5)

        # Check
        assert result.shape[0] == 5
        assert (result.columns == raw_data.columns).all()

        # Foreign key columns are all the same
        unique_foreign_keys = result['CUSTOMER_ID'].unique()
        sampled_parent = self.sampler.sampled['DEMO_CUSTOMERS'][0][1]
        assert len(unique_foreign_keys) == 1
        assert unique_foreign_keys[0] in sampled_parent['CUSTOMER_ID'].values

    def test_sample_all(self):
        """Check sample_all and returns some value."""

        # Run
        result = self.sampler.sample_all(num_rows=5)

        # Check
        assert result.keys() == self.sampler.dn.tables.keys()

        for name, table in result.items():
            with self.subTest(table=name):
                raw_data = self.modeler.dn.tables[name].data
                assert (table.columns == raw_data.columns).all()

                if not self.sampler.dn.get_parents(name):
                    primary_key = self.sampler.dn.get_meta_data(name)['primary_key']
                    assert len(table) == 5
                    assert len(table[primary_key].unique()) == 5
