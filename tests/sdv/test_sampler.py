from unittest import TestCase
from unittest.mock import MagicMock

from sdv.data_navigator import CSVDataLoader
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

    def test_unflatten_dict(self):
        """ """
        # Setup
        data_navigator = MagicMock()
        modeler = MagicMock()
        sampler = Sampler(data_navigator, modeler)
        flat = {
            'first_key__a': 1,
            'first_key__b': 2,
            'second_key__x': 0
        }

        expected_result = {
            'first_key': {
                'a': 1,
                'b': 2
            },
            'second_key': {
                'x': 0
            }
        }

        # Run
        result = sampler._unflatten_dict(flat)

        # Check
        assert result == expected_result
        data_navigator.assert_not_called()
        modeler.assert_not_called()

    def test_unflatten_dict_mixed_array(self):
        """unflatten_dict restruicture arrays"""
        # Setup
        data_navigator = MagicMock()
        modeler = MagicMock()
        sampler = Sampler(data_navigator, modeler)
        flat = {
            'first_key__0__0': 1,
            'first_key__0__1': 0,
            'first_key__1__0': 0,
            'first_key__1__1': 1,
            'second_key__0__std': 0.5,
            'second_key__0__mean': 0.5,
            'second_key__1__std': 0.25,
            'second_key__1__mean': 0.25
        }

        expected_result = {
            'first_key': [
                [1, 0],
                [0, 1]
            ],
            'second_key': [
                {
                    'std': 0.5,
                    'mean': 0.5
                },
                {
                    'std': 0.25,
                    'mean': 0.25
                }
            ]
        }

        # Run

        result = sampler._unflatten_dict(flat)

        # Check
        assert result == expected_result
        data_navigator.assert_not_called()
        modeler.assert_not_called()

    def test_unflatten_dict_extension(self):
        """ """
        # Setup
        data_navigator = MagicMock()
        data_navigator.get_children.return_value = ['CHILD_TABLE']
        modeler = MagicMock()
        sampler = Sampler(data_navigator, modeler)

        flat = {
            'first_key__a': 1,
            'first_key____CHILD_TABLE__model_param': 0,
            'distribs____CHILD_TABLE__distribs__UNIT_PRICE__std__mean': 0
        }
        expected_result = {
            'first_key': {
                'a': 1,
                'CHILD_TABLE': {
                    'model_param': 0
                }
            },
            'distribs': {
                'CHILD_TABLE__distribs__UNIT_PRICE__std': {
                    'mean': 0
                }
            }
        }

        # Run
        result = sampler._unflatten_dict(flat)

        # Check
        assert result == expected_result
