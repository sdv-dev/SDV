"""Local file handlers."""

import inspect
import os
from pathlib import Path

import pandas as pd

from sdv.metadata.metadata import Metadata

CSV_DEFAULT_READ_ARGS = {'parse_dates': False, 'low_memory': False, 'on_bad_lines': 'warn'}
CSV_DEFAULT_WRITE_ARGS = {'index': False}

UNSUPPORTED_ARGS = frozenset(['filepath_or_buffer', 'path_or_buf'])


class BaseLocalHandler:
    """Base class for local handlers."""

    def __init__(self, decimal='.', float_format=None):
        self.decimal = decimal
        self.float_format = float_format

    def create_metadata(self, data):
        """Detect the metadata for all tables in a dictionary of dataframes.

        Args:
            data (dict):
                Dictionary of table names to dataframes.

        Returns:
            Metadata:
                An ``sdv.metadata.Metadata`` object with the detected metadata
                properties from the data.
        """
        metadata = Metadata.detect_from_dataframes(data)
        return metadata

    def read(self):
        """Read data from files and return it along with metadata.

        This method must be implemented by subclasses.

        Returns:
            dict:
                The dictionary maps table names to pandas DataFrames.
        """
        raise NotImplementedError()

    def write(self):
        """Write data to files.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError()


class CSVHandler(BaseLocalHandler):
    """A class for handling CSV files."""

    def __init__(self):
        pass

    def _get_leading_zero_columns(self, file_path, read_csv_parameters):
        """Detect columns that contain numeric strings with leading zeros.

        Args:
            file_path (Path):
                Path to the CSV file being read.
            read_csv_parameters (dict):
                Parameters used for the read.

        Returns:
            list:
                Column names that contain values with leading zeros.
        """
        sample_parameters = read_csv_parameters.copy()
        for key in ('chunksize', 'iterator'):
            sample_parameters.pop(key, None)

        sample_parameters['dtype'] = str
        sampled_rows = sample_parameters.get('nrows', 100_000)
        sample_parameters['nrows'] = min(sampled_rows, 100_000)
        sample_data = pd.read_csv(file_path, **sample_parameters)

        leading_zero_columns = []
        for column in sample_data.columns:
            series = sample_data[column].dropna().astype(str).str.strip()
            if series.str.match(r'^0\d+$').any():
                leading_zero_columns.append(column)

        return leading_zero_columns

    def _read_table_data(self, file_path, read_csv_parameters, keep_leading_zeros):
        """Read a CSV file and apply leading-zero handling when requested."""
        if keep_leading_zeros:
            leading_zero_columns = self._get_leading_zero_columns(file_path, read_csv_parameters)
            leading_zero_parameters = read_csv_parameters.copy()
            if leading_zero_columns:
                dtype = leading_zero_parameters.get('dtype')
                if isinstance(dtype, dict):
                    dtype = {**dtype, **{col: str for col in leading_zero_columns}}
                elif dtype is None:
                    dtype = {col: str for col in leading_zero_columns}
                # if dtype is a str, don't override it
                leading_zero_parameters['dtype'] = dtype

            return pd.read_csv(file_path, **leading_zero_parameters)

        return pd.read_csv(file_path, **read_csv_parameters)

    def read(self, folder_name, file_names=None, read_csv_parameters=None, keep_leading_zeros=True):
        """Read data from CSV files and return it along with metadata.

        Args:
            folder_name (str):
                The name of the folder containing CSV files.
            file_names (list of str, optional):
                The names of CSV files to read. If None, all files ending with '.csv'
                in the folder are read.
            read_csv_parameters (dict):
                A dictionary with additional parameters to use when reading the CSVs.
                The keys are any of the parameter names of the pandas.read_csv function
                and the values are your inputs. Defaults to
                `{'parse_dates': False, 'low_memory': False, 'on_bad_lines': 'warn'}`
            keep_leading_zeros (bool):
                Whether to keep leading zeros by detecting numeric columns that have
                string values with leading zeros and loading those columns as strings.
                Defaults to ``True``.

        Returns:
            dict:
                The dictionary maps table names to pandas DataFrames.

        Raises:
            FileNotFoundError:
                If the specified files do not exist in the folder.

            ValueError:
                If a provided parameter in `read_csv_parameters` is not supported by the
                `CSVHandler`.
        """
        data = {}
        folder_path = Path(folder_name)
        read_csv_parameters = read_csv_parameters or {}
        for key, value in CSV_DEFAULT_READ_ARGS.items():
            read_csv_parameters.setdefault(key, value)

        for key in UNSUPPORTED_ARGS:
            if key in read_csv_parameters:
                raise ValueError(
                    f"The CSVHandler is unable to use the parameter '{key}' "
                    'because it can read multiple files at once. Please use the '
                    "'folder_name' and 'file_names' parameters instead."
                )

        if file_names is None:
            # If file_names is None, read all files in the folder ending with ".csv"
            file_paths = folder_path.glob('*.csv')
        else:
            # Validate if the given files exist in the folder
            missing_files = [file for file in file_names if not (folder_path / file).exists()]
            if missing_files:
                raise FileNotFoundError(
                    f'The following files do not exist in the folder: {", ".join(missing_files)}.'
                )

            file_paths = [folder_path / file for file in file_names]

        # Read CSV files
        args = inspect.getfullargspec(pd.read_csv)
        if 'on_bad_lines' not in args.kwonlyargs:
            read_csv_parameters.pop('on_bad_lines')
            read_csv_parameters['error_bad_lines'] = False

        for file_path in file_paths:
            table_name = file_path.stem  # Remove file extension to get table name
            table_data = self._read_table_data(file_path, read_csv_parameters, keep_leading_zeros)
            data[table_name] = table_data

        return data

    def write(
        self, synthetic_data, folder_name, file_name_suffix=None, mode='x', to_csv_parameters=None
    ):
        """Write synthetic data to CSV files.

        Args:
            synthetic_data (dict):
                A dictionary mapping table names to pandas DataFrames containing synthetic data.
            folder_name (str):
                The name of the folder to write CSV files to.
            file_name_suffix (str, optional):
                An optional suffix to add to each file name. If ``None``, no suffix is added.
            mode (str, optional):
                The mode of writing to use. Defaults to 'x'.
                'x': Write to new files, raising errors if existing files exist with the same name.
                'w': Write to new files, clearing any existing files that exist.
                'a': Append the new CSV rows to any existing files.

            to_csv_parameters (dict):
                A dictionary with additional parameters to use when writing the CSVs.
                The keys are any of the parameter names of the pandas.to_csv function and
                the values are your input. Defaults to `{ 'index': False }`.
        """
        folder_path = Path(folder_name)
        to_csv_parameters = to_csv_parameters or {}
        for key, value in CSV_DEFAULT_WRITE_ARGS.items():
            to_csv_parameters.setdefault(key, value)

        to_csv_parameters['mode'] = mode
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for table_name, table_data in synthetic_data.items():
            file_name = f'{table_name}{file_name_suffix}' if file_name_suffix else f'{table_name}'
            file_path = f'{folder_path / file_name}.csv'
            table_data.to_csv(file_path, **to_csv_parameters)


class ExcelHandler(BaseLocalHandler):
    """A class for handling Excel files."""

    def _read_excel(self, filepath, sheet_names=None):
        """Read data from Excel File and return just the data as a dictionary."""
        data = {}
        if sheet_names is None:
            xl_file = pd.ExcelFile(filepath)
            sheet_names = xl_file.sheet_names

        for sheet_name in sheet_names:
            data[sheet_name] = pd.read_excel(
                filepath,
                sheet_name=sheet_name,
                parse_dates=False,
                decimal=self.decimal,
                index_col=None,
            )

        return data

    def read(self, filepath, sheet_names=None):
        """Read data from Excel files and return it along with metadata.

        Args:
            filepath (str):
                The path to the Excel file to read.
            sheet_names (list of str, optional):
                The names of sheets to read. If None, all sheets are read.

        Returns:
            dict:
                The dictionary maps table names to pandas DataFrames.
        """
        if sheet_names is not None and not isinstance(sheet_names, list):
            raise ValueError("'sheet_names' must be None or a list of strings.")

        return self._read_excel(filepath, sheet_names)

    def write(self, synthetic_data, filepath, sheet_name_suffix=None, mode='w'):
        """Write synthetic data to an Excel File.

        Args:
            synthetic_data (dict):
                A dictionary mapping table names to pandas DataFrames containing synthetic data.
            filepath (str):
                The path to the Excel file to write.
            sheet_name_suffix (str, optional):
                A suffix to add to each sheet name.
            mode (str, optional):
                The mode of writing to use. Defaults to 'w'.
                'w': Write sheets to a new file, clearing any existing file that may exist.
                'a': Append new sheets within the existing file.
                     Note: You cannot append data to existing sheets.
        """
        temp_data = synthetic_data
        suffix_added = False

        if mode == 'a':
            temp_data = self._read_excel(filepath)
            for table_name, table in synthetic_data.items():
                sheet_name = table_name
                if sheet_name_suffix:
                    sheet_name = f'{table_name}{sheet_name_suffix}'
                    suffix_added = True

                if temp_data.get(sheet_name) is not None:
                    temp_data[sheet_name] = pd.concat(
                        [temp_data[sheet_name], synthetic_data[sheet_name]], ignore_index=True
                    )

                else:
                    temp_data[sheet_name] = table

        writer = pd.ExcelWriter(filepath)
        for table_name, table_data in temp_data.items():
            if sheet_name_suffix and not suffix_added:
                table_name += sheet_name_suffix

            table_data.to_excel(
                writer, sheet_name=table_name, float_format=self.float_format, index=False
            )

        writer.close()
