"""Local file handlers."""

import codecs
import inspect
import os
from pathlib import Path

import pandas as pd

from sdv.metadata.metadata import Metadata


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
            MultiTableMetadata:
                An ``sdv.metadata.MultiTableMetadata`` object with the detected metadata
                properties from the data.
        """
        metadata = Metadata()
        metadata.detect_from_dataframes(data)
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
    """A class for handling CSV files.

    Args:
        sep (str):
            The separator used for reading and writing CSV files. Defaults to ``,``.
        encoding (str):
            The character encoding to use for reading and writing CSV files. Defaults to ``UTF``.
        decimal (str):
            The character used to denote the decimal point. Defaults to ``.``.
        float_format (str or None):
            The formatting string for floating-point numbers. Optional.
        quotechar (str):
            Character used to denote the start and end of a quoted item.
            Quoted items can include the delimiter and it will be ignored. Defaults to '"'.
        quoting (int or None):
            Control field quoting behavior. Default is 0.

    Raises:
        ValueError:
            If the provided encoding is not available in the system.
    """

    def __init__(
        self, sep=',', encoding='UTF', decimal='.', float_format=None, quotechar='"', quoting=0
    ):
        super().__init__(decimal, float_format)
        try:
            codecs.lookup(encoding)
        except LookupError as error:
            raise ValueError(
                f"The provided encoding '{encoding}' is not available in your system."
            ) from error

        self.sep = sep
        self.encoding = encoding
        self.quotechar = quotechar
        self.quoting = quoting

    def read(self, folder_name, file_names=None):
        """Read data from CSV files and return it along with metadata.

        Args:
            folder_name (str):
                The name of the folder containing CSV files.
            file_names (list of str, optional):
                The names of CSV files to read. If None, all files ending with '.csv'
                in the folder are read.

        Returns:
            dict:
                The dictionary maps table names to pandas DataFrames.

        Raises:
            FileNotFoundError:
                If the specified files do not exist in the folder.
        """
        data = {}
        folder_path = Path(folder_name)

        if file_names is None:
            # If file_names is None, read all files in the folder ending with ".csv"
            file_paths = folder_path.glob('*.csv')
        else:
            # Validate if the given files exist in the folder
            file_names = file_names
            missing_files = [file for file in file_names if not (folder_path / file).exists()]
            if missing_files:
                raise FileNotFoundError(
                    f"The following files do not exist in the folder: {', '.join(missing_files)}."
                )

            file_paths = [folder_path / file for file in file_names]

        # Read CSV files
        kwargs = {
            'sep': self.sep,
            'encoding': self.encoding,
            'parse_dates': False,
            'low_memory': False,
            'decimal': self.decimal,
            'on_bad_lines': 'warn',
            'quotechar': self.quotechar,
            'quoting': self.quoting,
        }

        args = inspect.getfullargspec(pd.read_csv)
        if 'on_bad_lines' not in args.kwonlyargs:
            kwargs.pop('on_bad_lines')
            kwargs['error_bad_lines'] = False

        for file_path in file_paths:
            table_name = file_path.stem  # Remove file extension to get table name
            data[table_name] = pd.read_csv(file_path, **kwargs)

        return data

    def write(self, synthetic_data, folder_name, file_name_suffix=None, mode='x'):
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
        """
        folder_path = Path(folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for table_name, table_data in synthetic_data.items():
            file_name = f'{table_name}{file_name_suffix}' if file_name_suffix else f'{table_name}'
            file_path = f'{folder_path / file_name}.csv'
            table_data.to_csv(
                file_path,
                sep=self.sep,
                encoding=self.encoding,
                index=False,
                float_format=self.float_format,
                quotechar=self.quotechar,
                quoting=self.quoting,
                mode=mode,
            )


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
