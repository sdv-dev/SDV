from unittest import TestCase, skip

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

    @skip
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

    @skip
    def test_sample_all(self):
        """Check sample_all and returns some value."""

        # Run
        result = self.sampler.sample_all()

        # Check
        assert result.keys() == self.sampler.dn.tables.keys()
