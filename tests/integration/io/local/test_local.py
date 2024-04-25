import pandas as pd

from sdv.io.local import CSVHandler
from sdv.metadata import MultiTableMetadata


class TestCSVHandler:

    def test_integration_read_write(self, tmpdir):
        """Test end to end the read and write methods of ``CSVHandler``."""
        # Prepare synthetic data
        synthetic_data = {
            'table1': pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}),
            'table2': pd.DataFrame({'col3': [4, 5, 6], 'col4': ['d', 'e', 'f']})
        }

        # Write synthetic data to CSV files
        handler = CSVHandler()
        handler.write(synthetic_data, tmpdir)

        # Read data from CSV files
        data, metadata = handler.read(tmpdir)

        # Check if data was read correctly
        assert len(data) == 2
        assert 'table1' in data
        assert 'table2' in data
        assert isinstance(metadata, MultiTableMetadata) is True

        # Check if the dataframes match the original synthetic data
        pd.testing.assert_frame_equal(data['table1'], synthetic_data['table1'])
        pd.testing.assert_frame_equal(data['table2'], synthetic_data['table2'])
