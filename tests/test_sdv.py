from unittest import TestCase, mock

from copulas import NotFittedError

from sdv import SDV


class TestSDV(TestCase):

    def test__check_unsupported_raises(self):
        """_check_unsupported will raise a ValueError if a table has two parents."""
        # Setup
        instance = SDV(meta_file_name='meta.json')

        data_navigator_mock = mock.MagicMock()
        data_navigator_mock.tables.keys.return_value = ['A', 'B']
        data_navigator_mock.get_parents.return_value = ['X', 'Y']

        instance.dn = data_navigator_mock

        # Run / Check
        with self.assertRaises(ValueError):
            instance._check_unsupported_dataset_structure()

    def test_sample_rows_not_fitted(self):
        """Check that the sample_rows raise an exception when is not fitted."""

        # Setup

        # Run and asserts
        sdv_mock = mock.Mock()
        sdv_mock.sampler = None

        table_name = 'DEMO'
        num_rows = 5

        with self.assertRaises(NotFittedError):
            SDV.sample_rows(sdv_mock, table_name, num_rows)

    def test_sample_rows_fitted(self):
        """Check that the sample_rows is called."""

        # Setup

        # Run
        sdv_mock = mock.Mock()
        sdv_mock.sampler = mock.Mock()

        table_name = 'DEMO'
        num_rows = 5

        SDV.sample_rows(sdv_mock, table_name, num_rows)

        # Asserts
        sdv_mock.sampler.sample_rows.assert_called_once_with(
            'DEMO', 5, sample_children=True, reset_primary_keys=False)

    def test_sample_table_not_fitted(self):
        """Check that the sample_table raise an exception when is not fitted."""

        # Setup

        # Run and asserts
        sdv_mock = mock.Mock()
        sdv_mock.sampler = None

        table_name = 'DEMO'

        with self.assertRaises(NotFittedError):
            SDV.sample_table(sdv_mock, table_name)

    def test_sample_table_fitted(self):
        """Check that the sample_table is called."""

        # Setup

        # Run
        sdv_mock = mock.Mock()
        sdv_mock.sampler = mock.Mock()

        table_name = 'DEMO'

        SDV.sample_table(sdv_mock, table_name)

        # Asserts
        sdv_mock.sampler.sample_table.assert_called_once_with(
            'DEMO', reset_primary_keys=False)
