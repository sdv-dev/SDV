from unittest import TestCase, expectedFailure

from sdv.DataNavigator import CSVDataLoader


class TestDataNavigator(TestCase):

    def setUp(self):
        data_loader = CSVDataLoader('tests/data/meta.json')
        self.data_navigator = data_loader.load_data()


    @expectedFailure
    def test_transform_data(self):
        """transform_data turns all data into float values in [0,1]."""

        # Run
        result = self.data_navigator.transform_data()

        # Check
        assert result.keys() == self.data_navigator.tables.keys()

        for name, table in result.items():
            raw_table = self.data_navigator.tables[name].data
            assert (table.columns == raw_table.columns).all()
            assert table.shape == raw_table.shape
            assert not table.equals(raw_table)