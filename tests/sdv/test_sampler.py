from unittest import TestCase, expectedFailure

from sdv.DataNavigator import CSVDataLoader
from sdv.Modeler import Modeler
from sdv.Sampler import Sampler


class TestSampler(TestCase):

    def setUp(self):
        data_loader = CSVDataLoader('tests/data/meta.json')
        data_navigator = data_loader.load_data()
        data_navigator.transform_data()

        modeler = Modeler(data_navigator)
        modeler.model_database()

        self.sampler = Sampler(data_navigator, modeler)

    @expectedFailure
    def test_sample_all(self):
        """Check sample_all and returns some value."""
        # Setup

        # Run
        result = self.sampler.sample_all()

        # Check
        assert result.keys() == self.sampler.dn.tables.keys()
