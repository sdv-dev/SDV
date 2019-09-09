from unittest import TestCase, mock

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
