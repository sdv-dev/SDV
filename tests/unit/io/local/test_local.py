"""Unit tests for local file handlers."""
import os
from pathlib import Path
from unittest.mock import Mock, call, patch

import pandas as pd
import pytest

from sdv.io.local.local import CSVHandler, ExcelHandler
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
        assert instance.quotechar == '"'
        assert instance.quoting == 0

    def test___init___custom(self):
        """Test custom initialization of the class."""
        # Run
        instance = CSVHandler(
            sep=';',
            encoding='utf-8',
            decimal=',',
            float_format='%.2f',
            quotechar="'",
            quoting=2
        )

        # Assert
        assert instance.decimal == ','
        assert instance.float_format == '%.2f'
        assert instance.encoding == 'utf-8'
        assert instance.sep == ';'
        assert instance.quotechar == "'"
        assert instance.quoting == 2

    def test___init___error_encoding(self):
        """Test custom initialization of the class."""
        # Run and Assert
        error_msg = "The provided encoding 'sdvutf-8' is not available in your system."
        with pytest.raises(ValueError, match=error_msg):
            CSVHandler(sep=';', encoding='sdvutf-8', decimal=',', float_format='%.2f')

    @patch('sdv.io.local.local.Path.glob')
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
        data, metadata = handler.read(tmpdir, file_names=['parent.csv'])

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
        assert 'table1_synthetic.csv' in os.listdir(tmpdir / 'synthetic_data')
        assert 'table2_synthetic.csv' in os.listdir(tmpdir / 'synthetic_data')

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
        with pytest.raises(FileExistsError):
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


class TestExcelHandler:

    def test___init__(self):
        """Test the init parameters with default values."""
        # Run
        instance = ExcelHandler()

        # Assert
        assert instance.decimal == '.'
        assert instance.float_format is None

    def test___init___custom(self):
        """Test custom initialization of the class."""
        # Run
        instance = ExcelHandler(decimal=',', float_format='%.2f')

        # Assert
        assert instance.decimal == ','
        assert instance.float_format == '%.2f'

    @patch('sdv.io.local.local.pd')
    def test_read(self, mock_pd):
        """Test the read method of ExcelHandler class."""
        # Setup
        file_path = 'test_file.xlsx'
        mock_pd.ExcelFile.return_value = Mock(sheet_names=['Sheet1', 'Sheet2'])
        mock_pd.read_excel.side_effect = [
            pd.DataFrame({'A': [1, 2], 'B': [3, 4]}),
            pd.DataFrame({'C': [5, 6], 'D': [7, 8]})
        ]

        instance = ExcelHandler()

        # Run
        data, metadata = instance.read(file_path)

        # Assert
        sheet_1_call = call(
            'test_file.xlsx',
            sheet_name='Sheet1',
            parse_dates=False,
            decimal='.',
            index_col=None
        )
        sheet_2_call = call(
            'test_file.xlsx',
            sheet_name='Sheet2',
            parse_dates=False,
            decimal='.',
            index_col=None
        )
        pd.testing.assert_frame_equal(
            data['Sheet1'],
            pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        )
        pd.testing.assert_frame_equal(
            data['Sheet2'],
            pd.DataFrame({'C': [5, 6], 'D': [7, 8]})
        )
        assert isinstance(metadata, MultiTableMetadata)
        assert mock_pd.read_excel.call_args_list == [sheet_1_call, sheet_2_call]

    @patch('sdv.io.local.local.pd')
    def test_read_sheet_names(self, mock_pd):
        """Test the read method when provided sheet names."""
        # Setup
        file_path = 'test_file.xlsx'
        sheet_names = ['Sheet1']
        mock_pd.ExcelFile.return_value = Mock(sheet_names=['Sheet1', 'Sheet2'])
        mock_pd.read_excel.side_effect = [
            pd.DataFrame({'A': [1, 2], 'B': [3, 4]}),
            pd.DataFrame({'C': [5, 6], 'D': [7, 8]})
        ]

        instance = ExcelHandler()

        # Run
        data, metadata = instance.read(file_path, sheet_names)

        # Assert
        sheet_1_call = call(
            'test_file.xlsx',
            sheet_name='Sheet1',
            parse_dates=False,
            decimal='.',
            index_col=None
        )
        pd.testing.assert_frame_equal(
            data['Sheet1'],
            pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        )
        assert isinstance(metadata, MultiTableMetadata)
        assert mock_pd.read_excel.call_args_list == [sheet_1_call]
        assert list(data) == ['Sheet1']

    def test_read_sheet_names_string(self):
        """Test the read method when provided sheet names but they are string."""
        # Setup
        file_path = 'test_file.xlsx'
        sheet_names = 'Sheet1'
        instance = ExcelHandler()

        # Run and Assert
        error_msg = "'sheet_names' must be None or a list of strings."
        with pytest.raises(ValueError, match=error_msg):
            instance.read(file_path, sheet_names)

    @patch('sdv.io.local.local.pd')
    def test_write(self, mock_pd):
        """Test the write functionality of the ExcelHandler."""
        # Setup
        sheet_one = Mock()
        sheet_two = Mock()
        synthetic_data = {'Sheet1': sheet_one, 'Sheet2': sheet_two}

        file_name = 'output_file.xlsx'
        sheet_name_suffix = '_synthetic'
        instance = ExcelHandler()

        # Run
        instance.write(synthetic_data, file_name, sheet_name_suffix)

        # Assert
        sheet_one.to_excel.assert_called_once_with(
            mock_pd.ExcelWriter.return_value,
            sheet_name='Sheet1_synthetic',
            float_format=None,
            index=False
        )
        sheet_two.to_excel.assert_called_once_with(
            mock_pd.ExcelWriter.return_value,
            sheet_name='Sheet2_synthetic',
            float_format=None,
            index=False
        )
        mock_pd.ExcelWriter.return_value.close.assert_called_once_with()

    @patch('sdv.io.local.local.pd')
    def test_write_mode_append(self, mock_pd):
        """Test the write functionality of the ExcelHandler when mode is `a``."""
        # Setup
        sheet_one = Mock()
        sheet_two = Mock()
        synth_sheet_one = Mock()
        synth_sheet_two = Mock()
        synthetic_data = {'Sheet1': synth_sheet_one, 'Sheet2': synth_sheet_two}

        file_name = 'output_file.xlsx'
        sheet_name_suffix = '_synthetic'
        instance = ExcelHandler()
        instance._read_excel = Mock(return_value={'Sheet1': sheet_one, 'Sheet2': sheet_two})

        # Run
        instance.write(synthetic_data, file_name, sheet_name_suffix, mode='a')

        # Assert
        sheet_one.to_excel.assert_called_once_with(
            mock_pd.ExcelWriter.return_value,
            sheet_name='Sheet1',
            float_format=None,
            index=False
        )
        sheet_two.to_excel.assert_called_once_with(
            mock_pd.ExcelWriter.return_value,
            sheet_name='Sheet2',
            float_format=None,
            index=False
        )
        synth_sheet_one.to_excel.assert_called_once_with(
            mock_pd.ExcelWriter.return_value,
            sheet_name='Sheet1_synthetic',
            float_format=None,
            index=False
        )
        synth_sheet_two.to_excel.assert_called_once_with(
            mock_pd.ExcelWriter.return_value,
            sheet_name='Sheet2_synthetic',
            float_format=None,
            index=False
        )
        mock_pd.ExcelWriter.return_value.close.assert_called_once_with()

    @patch('sdv.io.local.local.pd')
    def test_write_mode_append_no_suffix(self, mock_pd):
        """Test the write functionality of the ExcelHandler when mode is `a`` and no suffix."""
        # Setup
        sheet_one = Mock()
        sheet_two = Mock()
        synth_sheet_one = Mock()
        synthetic_data = {'Sheet1': synth_sheet_one}
        file_name = 'output_file.xlsx'
        instance = ExcelHandler()
        instance._read_excel = Mock(return_value={'Sheet1': sheet_one, 'Sheet2': sheet_two})

        # Run
        instance.write(synthetic_data, file_name, mode='a')

        # Assert
        mock_pd.concat.assert_called_once_with([sheet_one, synth_sheet_one], ignore_index=True)
        mock_pd.concat.return_value.to_excel.assert_called_once_with(
            mock_pd.ExcelWriter.return_value,
            sheet_name='Sheet1',
            float_format=None,
            index=False
        )

        sheet_two.to_excel.assert_called_once_with(
            mock_pd.ExcelWriter.return_value,
            sheet_name='Sheet2',
            float_format=None,
            index=False
        )
        mock_pd.ExcelWriter.return_value.close.assert_called_once_with()
