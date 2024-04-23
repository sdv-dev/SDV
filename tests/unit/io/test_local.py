"""Unit tests for local file handlers."""
import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from sdv.io.local import CSVHandler
from sdv.metadata.multi_table import MultiTableMetadata


class TestCSVHandler:

    def test___init__(self):
        """Test the dafault initialization of the class."""
        # Run
        instance = CSVHandler()

        # Assert
        assert instance.decimal == '.'
        assert instance.float_format is None
        assert instance.encoding == 'UTF'
        assert instance.sep == ','

    def test___init___custom(self):
        """Test custom initialization of the class."""
        # Run
        instance = CSVHandler(sep=';', encoding='utf-8', decimal=',', float_format='%.2f')

        # Assert
        assert instance.decimal == ','
        assert instance.float_format == '%.2f'
        assert instance.encoding == 'utf-8'
        assert instance.sep == ';'

    def test___init___error_encoding(self):
        """Test custom initialization of the class."""
        # Run and Assert
        error_msg = "The provided encoding 'sdvutf-8' is not available in your system."
        with pytest.raises(ValueError, match=error_msg):
            CSVHandler(sep=';', encoding='sdvutf-8', decimal=',', float_format='%.2f')

    @patch('sdv.io.local.Path.glob')
    @patch('pandas.read_csv')
    def test_read(self, mock_read_csv, mock_glob):
        """Test the read method of CSVHandler class with a folder."""
        # Setup
        mock_glob.return_value = [
            Path('/path/to/data/parent.csv'),
            Path('/path/to/data/child.csv')
        ]
        mock_read_csv.side_effect = [
            pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}),
            pd.DataFrame({'col3': [4, 5, 6], 'col4': ['d', 'e', 'f']})
        ]

        handler = CSVHandler()

        # Run
        data, metadata = handler.read('/path/to/data')

        # Assert
        assert len(data) == 2
        assert 'parent' in data
        assert 'child' in data
        assert isinstance(metadata, MultiTableMetadata)
        assert mock_read_csv.call_count == 2
        pd.testing.assert_frame_equal(
            data['parent'],
            pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        )
        pd.testing.assert_frame_equal(
            data['child'],
            pd.DataFrame({'col3': [4, 5, 6], 'col4': ['d', 'e', 'f']})
        )

    def test_read_files(self, tmpdir):
        """Test the read method of CSVHandler class with given ``file_names``."""
        # Setup
        file_path = Path(tmpdir)
        pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}).to_csv(
            file_path / 'parent.csv',
            index=False
        )
        pd.DataFrame({'col3': [4, 5, 6], 'col4': ['d', 'e', 'f']}).to_csv(
            file_path / 'child.csv',
            index=False
        )

        handler = CSVHandler()

        # Run
        data, metadata = handler.read(tmpdir, file_names='parent.csv')

        # Assert
        assert 'parent' in data
        pd.testing.assert_frame_equal(
            data['parent'],
            pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        )

    def test_read_files_missing(self, tmpdir):
        """Test the read method of CSVHandler with missing ``file_names``."""
        # Setup
        file_path = Path(tmpdir)
        pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}).to_csv(
            file_path / 'parent.csv',
            index=False
        )
        pd.DataFrame({'col3': [4, 5, 6], 'col4': ['d', 'e', 'f']}).to_csv(
            file_path / 'child.csv',
            index=False
        )

        handler = CSVHandler()

        # Run and Assert
        error_msg = 'The following files do not exist in the folder: grandchild.csv, parents.csv.'
        with pytest.raises(FileNotFoundError, match=error_msg):
            handler.read(tmpdir, file_names=['grandchild.csv', 'parents.csv'])

    def test_write(self, tmpdir):
        """Test the write functionality of a CSVHandler."""
        # Setup
        synthetic_data = {
            'table1': pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}),
            'table2': pd.DataFrame({'col3': [4, 5, 6], 'col4': ['d', 'e', 'f']})
        }
        handler = CSVHandler()

        assert os.path.exists(tmpdir / 'synthetic_data') is False

        # Run
        handler.write(synthetic_data, tmpdir / 'synthetic_data', file_name_suffix='_synthetic')

        # Assert
        assert os.listdir(tmpdir / 'synthetic_data') == [
            'table2_synthetic.csv',
            'table1_synthetic.csv'
        ]

    def test_write_file_exists(self, tmpdir):
        """Test that an error is raised when it exists and the mode is `x`."""
        # Setup
        synthetic_data = {
            'table1': pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}),
            'table2': pd.DataFrame({'col3': [4, 5, 6], 'col4': ['d', 'e', 'f']})
        }

        os.makedirs(tmpdir / 'synthetic_data')
        synthetic_data['table1'].to_csv(tmpdir / 'synthetic_data' / 'table1.csv', index=False)
        handler = CSVHandler()

        # Run
        with pytest.raises(FileExistsError, match=f"{tmpdir / 'synthetic_data' / 'table1.csv'}"):
            handler.write(synthetic_data, tmpdir / 'synthetic_data')

    def test_write_file_exists_mode_is_a(self, tmpdir):
        """Test the write functionality of a CSVHandler when the mode is ``a``."""
        # Setup
        synthetic_data = {
            'table1': pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}),
            'table2': pd.DataFrame({'col3': [4, 5, 6], 'col4': ['d', 'e', 'f']})
        }

        os.makedirs(tmpdir / 'synthetic_data')
        synthetic_data['table1'].to_csv(tmpdir / 'synthetic_data' / 'table1.csv', index=False)
        handler = CSVHandler()

        # Run
        handler.write(synthetic_data, tmpdir / 'synthetic_data', mode='a')

        # Assert
        dataframe = pd.read_csv(tmpdir / 'synthetic_data' / 'table1.csv')
        expected_dataframe = pd.DataFrame({
            'col1': ['1', '2', '3', 'col1', '1', '2', '3'],
            'col2': ['a', 'b', 'c', 'col2', 'a', 'b', 'c']
        })
        pd.testing.assert_frame_equal(dataframe, expected_dataframe)

    def test_write_file_exists_mode_is_w(self, tmpdir):
        """Test the write functionality of a CSVHandler when the mode is ``w``."""
        # Setup
        synthetic_data = {
            'table1': pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}),
            'table2': pd.DataFrame({'col3': [4, 5, 6], 'col4': ['d', 'e', 'f']})
        }

        os.makedirs(tmpdir / 'synthetic_data')
        synthetic_data['table1'].to_csv(tmpdir / 'synthetic_data' / 'table1.csv', index=False)
        handler = CSVHandler()

        # Run
        handler.write(synthetic_data, tmpdir / 'synthetic_data', mode='w')

        # Assert
        dataframe = pd.read_csv(tmpdir / 'synthetic_data' / 'table1.csv')
        expected_dataframe = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        pd.testing.assert_frame_equal(dataframe, expected_dataframe)
