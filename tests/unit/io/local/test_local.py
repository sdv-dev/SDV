"""Unit tests for local file handlers."""

import os
import re
from pathlib import Path
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from sdv.io.local.local import BaseLocalHandler, CSVHandler, ExcelHandler
from sdv.metadata import Metadata


class TestBaseLocalHandler:
    def test___init__(self):
        """Test the default initialization of the class."""
        # Run
        instance = BaseLocalHandler()

        # Assert
        assert instance.decimal == '.'
        assert instance.float_format is None

    def test_create_metadata(self):
        """Test that ``create_metadata`` will infer the metadata."""
        # Setup
        data = {
            'hotel': pd.DataFrame({'hotel_id': [1, 2, 3, 4, 5], 'stars': [3, 4, 5, 3, 4]}),
            'guests': pd.DataFrame({'guest_id': [1, 2, 3, 4, 5], 'hotel_id': [1, 1, 3, 2, 3]}),
        }
        instance = BaseLocalHandler()

        # Run
        metadata = instance.create_metadata(data)

        # Assert
        assert isinstance(metadata, Metadata)
        assert metadata.to_dict() == {
            'METADATA_SPEC_VERSION': 'V1',
            'relationships': [
                {
                    'child_foreign_key': 'hotel_id',
                    'child_table_name': 'guests',
                    'parent_primary_key': 'hotel_id',
                    'parent_table_name': 'hotel',
                },
            ],
            'tables': {
                'guests': {
                    'columns': {
                        'guest_id': {'sdtype': 'id'},
                        'hotel_id': {'sdtype': 'id'},
                    },
                    'primary_key': 'guest_id',
                },
                'hotel': {
                    'columns': {
                        'hotel_id': {'sdtype': 'id'},
                        'stars': {'sdtype': 'numerical'},
                    },
                    'primary_key': 'hotel_id',
                },
            },
        }


class TestCSVHandler:
    def test___init__(self):
        """Test the dafault initialization of the class."""
        # Run
        instance = CSVHandler()

        # Assert
        assert not hasattr(instance, 'decimal')
        assert not hasattr(instance, 'float_format')
        assert not hasattr(instance, 'encoding')
        assert not hasattr(instance, 'sep')
        assert not hasattr(instance, 'quotechar')
        assert not hasattr(instance, 'quoting')

    @patch('sdv.io.local.local.Path.glob')
    @patch('pandas.read_csv')
    def test_read(self, mock_read_csv, mock_glob):
        """Test the read method of CSVHandler class with a folder."""
        # Setup
        mock_glob.return_value = [Path('/path/to/data/parent.csv'), Path('/path/to/data/child.csv')]
        mock_read_csv.side_effect = [
            pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}),
            pd.DataFrame({'col3': [4, 5, 6], 'col4': ['d', 'e', 'f']}),
        ]

        handler = CSVHandler()

        # Run
        data = handler.read('/path/to/data', keep_leading_zeros=False)

        # Assert
        assert len(data) == 2
        assert 'parent' in data
        assert 'child' in data
        assert mock_read_csv.call_count == 2
        pd.testing.assert_frame_equal(
            data['parent'], pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        )
        pd.testing.assert_frame_equal(
            data['child'], pd.DataFrame({'col3': [4, 5, 6], 'col4': ['d', 'e', 'f']})
        )

    def test_read_keep_leading_zeros_default(self, tmpdir):
        """Test that leading zeros are preserved by default."""
        # Setup
        file_path = Path(tmpdir)
        data = pd.DataFrame({
            'zip_code': ['02116', '10110'],
            'age': [30, 25],
        })
        data.to_csv(file_path / 'users.csv', index=False)

        handler = CSVHandler()

        # Run
        out = handler.read(tmpdir, file_names=['users.csv'])

        # Assert
        pd.testing.assert_frame_equal(out['users'], data)

    def test_read_keep_leading_zeros_multiple_files_mixed_types(self, tmpdir):
        """Test leading zeros with multiple files and mixed dtypes."""
        # Setup
        file_path = Path(tmpdir)
        users = pd.DataFrame({
            'user_id': [1, 2, None],
            'zip_code': ['00123', '98765', np.nan],
            'age': [30, 25, None],
            'is_active': [True, False, None],
            'joined_at': ['2024-01-01', '2024-01-02', None],
        })
        orders = pd.DataFrame({
            'order_id': [10, 20, np.nan],
            'tracking_code': ['000045', '123450', None],
            'amount': [10.5, 20.0, None],
            'discount_rate': [0.0001, 0.0015, np.nan],
            'notes': ['first', 'second', np.nan],
        })
        users.to_csv(file_path / 'users.csv', index=False)
        orders.to_csv(file_path / 'orders.csv', index=False)

        handler = CSVHandler()

        # Run
        out = handler.read(tmpdir, file_names=['users.csv', 'orders.csv'])

        # Assert
        users_expected = users.where(users.notna(), np.nan)
        orders_expected = orders.where(orders.notna(), np.nan)
        pd.testing.assert_frame_equal(out['users'], users_expected)
        pd.testing.assert_frame_equal(out['orders'], orders_expected)

    def test_read_keep_leading_zeros_false(self, tmpdir):
        """Test that leading zeros can be ignored when requested."""
        # Setup
        file_path = Path(tmpdir)
        pd.DataFrame({
            'zip_code': ['02116', '10110'],
            'age': [30, 25],
        }).to_csv(file_path / 'users.csv', index=False)

        handler = CSVHandler()

        # Run
        data = handler.read(tmpdir, file_names=['users.csv'], keep_leading_zeros=False)

        # Assert
        expected = pd.DataFrame({'zip_code': [2116, 10110], 'age': [30, 25]})
        pd.testing.assert_frame_equal(data['users'], expected)

    def test_read_files(self, tmpdir):
        """Test the read method of CSVHandler class with given ``file_names``."""
        # Setup
        file_path = Path(tmpdir)
        pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}).to_csv(
            file_path / 'parent.csv', index=False
        )
        pd.DataFrame({'col3': [4, 5, 6], 'col4': ['d', 'e', 'f']}).to_csv(
            file_path / 'child.csv', index=False
        )

        handler = CSVHandler()

        # Run
        data = handler.read(tmpdir, file_names=['parent.csv'])

        # Assert
        assert 'parent' in data
        pd.testing.assert_frame_equal(
            data['parent'], pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        )

    def test_read_files_missing(self, tmpdir):
        """Test the read method of CSVHandler with missing ``file_names``."""
        # Setup
        file_path = Path(tmpdir)
        pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}).to_csv(
            file_path / 'parent.csv', index=False
        )
        pd.DataFrame({'col3': [4, 5, 6], 'col4': ['d', 'e', 'f']}).to_csv(
            file_path / 'child.csv', index=False
        )

        handler = CSVHandler()

        # Run and Assert
        error_msg = 'The following files do not exist in the folder: grandchild.csv, parents.csv.'
        with pytest.raises(FileNotFoundError, match=error_msg):
            handler.read(tmpdir, file_names=['grandchild.csv', 'parents.csv'])

    def test_read_files_custom_parameters(self, tmpdir):
        """Test the read method of CSVHandler class with custom read parameters."""
        # Setup
        file_path = Path(tmpdir)
        read_csv_parameters = {
            'encoding': 'latin-1',
            'nrows': 1,
            'escapechar': '\\',
            'quotechar': '"',
            'sep': ';',
        }
        pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}).to_csv(
            file_path / 'parent.csv', index=False, sep=';'
        )
        pd.DataFrame({'col3': [4, 5, 6], 'col4': ['d', 'e', 'f']}).to_csv(
            file_path / 'child.csv', index=False, sep=';'
        )

        handler = CSVHandler()

        # Run
        data = handler.read(
            tmpdir, file_names=['parent.csv'], read_csv_parameters=read_csv_parameters
        )

        # Assert
        assert 'parent' in data
        pd.testing.assert_frame_equal(data['parent'], pd.DataFrame({'col1': [1], 'col2': ['a']}))

    def test_read_files_bad_parameters(self, tmpdir):
        """Test the read method of CSVHandler class with custom read parameters."""
        # Setup
        file_path = Path(tmpdir)
        read_csv_parameters = {
            'filepath_or_buffer': 'myfile',
            'nrows': 1,
            'sep': ';',
        }
        pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}).to_csv(
            file_path / 'parent.csv', index=False, sep=';'
        )
        pd.DataFrame({'col3': [4, 5, 6], 'col4': ['d', 'e', 'f']}).to_csv(
            file_path / 'child.csv', index=False, sep=';'
        )

        handler = CSVHandler()

        # Run and Assert
        error_msg = re.escape(
            "The CSVHandler is unable to use the parameter 'filepath_or_buffer' because it can "
            "read multiple files at once. Please use the 'folder_name' and 'file_names' "
            'parameters instead.'
        )
        with pytest.raises(ValueError, match=error_msg):
            handler.read(tmpdir, file_names=['parent.csv'], read_csv_parameters=read_csv_parameters)

    def test_write(self, tmpdir):
        """Test the write functionality of a CSVHandler."""
        # Setup
        synthetic_data = {
            'table1': pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}),
            'table2': pd.DataFrame({'col3': [4, 5, 6], 'col4': ['d', 'e', 'f']}),
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
            'table2': pd.DataFrame({'col3': [4, 5, 6], 'col4': ['d', 'e', 'f']}),
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
            'table2': pd.DataFrame({'col3': [4, 5, 6], 'col4': ['d', 'e', 'f']}),
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
            'col2': ['a', 'b', 'c', 'col2', 'a', 'b', 'c'],
        })
        pd.testing.assert_frame_equal(dataframe, expected_dataframe)

    def test_write_file_exists_mode_is_w(self, tmpdir):
        """Test the write functionality of a CSVHandler when the mode is ``w``."""
        # Setup
        synthetic_data = {
            'table1': pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}),
            'table2': pd.DataFrame({'col3': [4, 5, 6], 'col4': ['d', 'e', 'f']}),
        }

        os.makedirs(tmpdir / 'synthetic_data')
        synthetic_data['table1'].to_csv(tmpdir / 'synthetic_data' / 'table1.csv', index=False)
        handler = CSVHandler()

        # Run
        handler.write(synthetic_data, tmpdir / 'synthetic_data', mode='w')

        # Assert
        dataframe = pd.read_csv(tmpdir / 'synthetic_data' / 'table1.csv')
        expected_dataframe = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        pd.testing.assert_frame_equal(dataframe, expected_dataframe)

    def test_write_file_with_custom_params(self, tmpdir):
        """Test the write functionality of a CSVHandler when the mode is ``w``."""
        # Setup
        table_one_mock = Mock()
        table_two_mock = Mock()

        synthetic_data = {'table1': table_one_mock, 'table2': table_two_mock}

        os.makedirs(tmpdir / 'synthetic_data')
        handler = CSVHandler()
        write_parameters = {'index': True, 'sep': ';'}

        # Run
        handler.write(synthetic_data, tmpdir / 'synthetic_data', to_csv_parameters=write_parameters)

        # Assert
        table_one_mock.to_csv.assert_called_once_with(
            tmpdir / 'synthetic_data' / 'table1.csv', index=True, sep=';', mode='x'
        )
        table_two_mock.to_csv.assert_called_once_with(
            tmpdir / 'synthetic_data' / 'table2.csv', index=True, sep=';', mode='x'
        )


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
            pd.DataFrame({'C': [5, 6], 'D': [7, 8]}),
        ]

        instance = ExcelHandler()

        # Run
        data = instance.read(file_path)

        # Assert
        sheet_1_call = call(
            'test_file.xlsx', sheet_name='Sheet1', parse_dates=False, decimal='.', index_col=None
        )
        sheet_2_call = call(
            'test_file.xlsx', sheet_name='Sheet2', parse_dates=False, decimal='.', index_col=None
        )
        pd.testing.assert_frame_equal(data['Sheet1'], pd.DataFrame({'A': [1, 2], 'B': [3, 4]}))
        pd.testing.assert_frame_equal(data['Sheet2'], pd.DataFrame({'C': [5, 6], 'D': [7, 8]}))
        assert mock_pd.read_excel.call_args_list == [sheet_1_call, sheet_2_call]

    @patch('sdv.io.local.local.pd')
    def test_read_sheet_names(self, mock_pd):
        """Test the read method when provided sheet names."""
        # Setup
        filepath = 'test_file.xlsx'
        sheet_names = ['Sheet1']
        mock_pd.ExcelFile.return_value = Mock(sheet_names=['Sheet1', 'Sheet2'])
        mock_pd.read_excel.side_effect = [
            pd.DataFrame({'A': [1, 2], 'B': [3, 4]}),
            pd.DataFrame({'C': [5, 6], 'D': [7, 8]}),
        ]

        instance = ExcelHandler()

        # Run
        data = instance.read(filepath, sheet_names)

        # Assert
        sheet_1_call = call(
            'test_file.xlsx', sheet_name='Sheet1', parse_dates=False, decimal='.', index_col=None
        )
        pd.testing.assert_frame_equal(data['Sheet1'], pd.DataFrame({'A': [1, 2], 'B': [3, 4]}))
        assert mock_pd.read_excel.call_args_list == [sheet_1_call]
        assert list(data) == ['Sheet1']

    def test_read_sheet_names_string(self):
        """Test the read method when provided sheet names but they are string."""
        # Setup
        filepath = 'test_file.xlsx'
        sheet_names = 'Sheet1'
        instance = ExcelHandler()

        # Run and Assert
        error_msg = "'sheet_names' must be None or a list of strings."
        with pytest.raises(ValueError, match=error_msg):
            instance.read(filepath, sheet_names)

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
            index=False,
        )
        sheet_two.to_excel.assert_called_once_with(
            mock_pd.ExcelWriter.return_value,
            sheet_name='Sheet2_synthetic',
            float_format=None,
            index=False,
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
            mock_pd.ExcelWriter.return_value, sheet_name='Sheet1', float_format=None, index=False
        )
        sheet_two.to_excel.assert_called_once_with(
            mock_pd.ExcelWriter.return_value, sheet_name='Sheet2', float_format=None, index=False
        )
        synth_sheet_one.to_excel.assert_called_once_with(
            mock_pd.ExcelWriter.return_value,
            sheet_name='Sheet1_synthetic',
            float_format=None,
            index=False,
        )
        synth_sheet_two.to_excel.assert_called_once_with(
            mock_pd.ExcelWriter.return_value,
            sheet_name='Sheet2_synthetic',
            float_format=None,
            index=False,
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
            mock_pd.ExcelWriter.return_value, sheet_name='Sheet1', float_format=None, index=False
        )

        sheet_two.to_excel.assert_called_once_with(
            mock_pd.ExcelWriter.return_value, sheet_name='Sheet2', float_format=None, index=False
        )
        mock_pd.ExcelWriter.return_value.close.assert_called_once_with()
