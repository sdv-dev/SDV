import pandas as pd

from sdv.io.local import CSVHandler, ExcelHandler
from sdv.metadata import MultiTableMetadata


class TestCSVHandler:
    def test_integration_write_and_read(self, tmpdir):
        """Test end to end the write and read methods of ``CSVHandler``."""
        # Prepare synthetic data
        synthetic_data = {
            'table1': pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}),
            'table2': pd.DataFrame({'col3': [4, 5, 6], 'col4': ['d', 'e', 'f']}),
        }

        # Write synthetic data to CSV files
        handler = CSVHandler()
        handler.write(synthetic_data, tmpdir)

        # Read data from CSV files
        data = handler.read(tmpdir)

        # Detect metadata
        metadata = handler.create_metadata(data)

        # Check if data was read correctly
        assert len(data) == 2
        assert 'table1' in data
        assert 'table2' in data
        assert isinstance(metadata, MultiTableMetadata) is True

        # Check if the dataframes match the original synthetic data
        pd.testing.assert_frame_equal(data['table1'], synthetic_data['table1'])
        pd.testing.assert_frame_equal(data['table2'], synthetic_data['table2'])


class TestExcelHandler:
    def test_integration_write_and_read(self, tmpdir):
        """Test end to end the write and read methods of ``ExcelHandler``."""
        # Prepare synthetic data
        synthetic_data = {
            'table1': pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}),
            'table2': pd.DataFrame({'col3': [4, 5, 6], 'col4': ['d', 'e', 'f']}),
        }

        # Write synthetic data to xslx files
        handler = ExcelHandler()
        handler.write(synthetic_data, tmpdir / 'excel.xslx')

        # Read data from xslx file
        data = handler.read(tmpdir / 'excel.xslx')

        # Detect metadata
        metadata = handler.create_metadata(data)

        # Check if data was read correctly
        assert len(data) == 2
        assert 'table1' in data
        assert 'table2' in data
        assert isinstance(metadata, MultiTableMetadata) is True

        # Check if the dataframes match the original synthetic data
        pd.testing.assert_frame_equal(data['table1'], synthetic_data['table1'])
        pd.testing.assert_frame_equal(data['table2'], synthetic_data['table2'])

    def test_integration_write_and_read_append_mode(self, tmpdir):
        """Test end to end the write and read methods of ``ExcelHandler``."""
        # Prepare synthetic data
        synthetic_data = {
            'table1': pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}),
            'table2': pd.DataFrame({'col3': [4, 5, 6], 'col4': ['d', 'e', 'f']}),
        }

        # Write synthetic data to xslx files
        handler = ExcelHandler()
        handler.write(synthetic_data, tmpdir / 'excel.xslx')

        # Read data from xslx file
        data = handler.read(tmpdir / 'excel.xslx')

        # Write using append mode
        handler.write(synthetic_data, tmpdir / 'excel.xslx', mode='a')

        # Read data from xslx file
        data = handler.read(tmpdir / 'excel.xslx')

        # Detect metadata
        metadata = handler.create_metadata(data)

        # Check if data was read correctly
        assert len(data) == 2
        assert 'table1' in data
        assert 'table2' in data
        assert isinstance(metadata, MultiTableMetadata) is True

        # Check if the dataframes match the original synthetic data
        expected_table_one = pd.concat(
            [synthetic_data['table1'], synthetic_data['table1']], ignore_index=True
        )
        expected_table_two = pd.concat(
            [synthetic_data['table2'], synthetic_data['table2']], ignore_index=True
        )
        pd.testing.assert_frame_equal(data['table1'], expected_table_one)
        pd.testing.assert_frame_equal(data['table2'], expected_table_two)
