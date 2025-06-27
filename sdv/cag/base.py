"""Base constraint."""

import logging

import numpy as np
import pandas as pd

from sdv._utils import _format_invalid_values_string
from sdv.data_processing.datetime_formatter import DatetimeFormatter
from sdv.data_processing.numerical_formatter import NumericalFormatter
from sdv.errors import NotFittedError

LOGGER = logging.getLogger(__name__)


class BaseConstraint:
    """Base Constraint Class."""

    _is_single_table = True

    def __init__(self):
        self.metadata = None
        self._fitted = False
        self._single_table = False
        self._dtypes = None
        self._formatters = {}

    def _convert_data_to_dictionary(self, data, metadata, copy=False):
        """Helper to handle converting single dataframes into dictionaries.

        This method takes in data, metadata, and, optionally, a flag indiciating if the
        returned data should be a copy of the original input data. If the data is a single
        dataframe, it converts it into a dictionary of dataframes.
        """
        if isinstance(data, pd.DataFrame):
            if copy:
                data = data.copy()

            if self._single_table:
                data = {self._table_name: data}
            else:
                table_name = self._get_single_table_name(metadata)
                data = {table_name: data}
        elif copy:
            data = {table_name: table_data.copy() for table_name, table_data in data.items()}

        return data

    def _get_single_table_name(self, metadata):
        if not hasattr(self, 'table_name'):
            raise ValueError('No ``table_name`` attribute has been set.')

        return metadata._get_single_table_name() if self.table_name is None else self.table_name

    def _format_error_message_constraint(self, invalid_data, table_name):
        """Format the error message for the constraints."""
        invalid_rows_str = _format_invalid_values_string(invalid_data, 5)
        return (
            f"Data is not valid for the '{self.__class__.__name__}' constraint in "
            f"table '{table_name}':\n{invalid_rows_str}"
        )

    def _validate_constraint_with_metadata(self, metadata):
        raise NotImplementedError()

    def _validate_constraint_with_data(self, data, metadata):
        raise NotImplementedError()

    def validate(self, data=None, metadata=None):
        """Validate the data/metadata meets the constraint requirements.

        Args:
            data (dict[str, pd.DataFrame], optional)
                The data dictionary. If `None`, ``validate`` will skip data validation.
            metadata (sdv.Metadata, optional)
                The input metadata. If `None`, constraint must have been fitted and ``validate``
                will use the metadata saved during fitting.
        """
        if metadata is None:
            if self.metadata is None:
                raise NotFittedError('Constraint must be fit before validating without metadata.')

            metadata = self.metadata

        self._validate_constraint_with_metadata(metadata)

        if data is not None:
            data = self._convert_data_to_dictionary(data, metadata)
            self._validate_constraint_with_data(data, metadata)

    def _get_updated_metadata(self, metadata):
        return metadata

    def get_updated_metadata(self, metadata):
        """Get the updated metadata after applying the constraint to the input metadata.

        Args:
            metadata (sdv.Metadata):
                The input metadata to apply the constraint to.
        """
        self.validate(metadata=metadata)
        return self._get_updated_metadata(metadata)

    def _fit(self, data, metadata):
        raise NotImplementedError

    def _fit_constraint_column_formatters(self, data):
        """Fit formatters for columns that are dropped by constraints before data processing."""
        updated_metadata = self._get_updated_metadata(self.metadata)
        for table_name in self.metadata.tables:
            self._formatters[table_name] = {}
            primary_key = self.metadata.tables[table_name].primary_key
            input_columns = self._original_data_columns[table_name]
            table_metdadata = updated_metadata.tables.get(table_name)
            if table_metdadata:
                columns_to_format = set(input_columns) - set(table_metdadata.get_column_names())
            else:
                columns_to_format = set(input_columns)

            for column_name in columns_to_format:
                column_metadata = self.metadata.tables[table_name].columns.get(column_name)
                sdtype = column_metadata.get('sdtype')
                if sdtype == 'numerical' and column_name != primary_key:
                    representation = column_metadata.get('computer_representation', 'Float')
                    self._formatters[table_name][column_name] = NumericalFormatter(
                        computer_representation=representation,
                        enforce_rounding=True,
                        enforce_min_max_values=True,
                    )
                    self._formatters[table_name][column_name].learn_format(
                        data[table_name][column_name]
                    )

                elif sdtype == 'datetime' and column_name != primary_key:
                    datetime_format = column_metadata.get('datetime_format')
                    self._formatters[table_name][column_name] = DatetimeFormatter(
                        datetime_format=datetime_format
                    )
                    self._formatters[table_name][column_name].learn_format(
                        data[table_name][column_name]
                    )

    def _format_constraint_columns(self, data):
        """Format columns skipped by the data processor due to being dropped by constraints."""
        if not hasattr(self, '_formatters'):
            # Backwards compatibility for constraints saved without formatters.
            return data

        for table_name, table_data in data.items():
            column_order = [
                column for column in self._original_data_columns[table_name] if column in table_data
            ]
            table_formatters = self._formatters.get(table_name, {})
            for column_name, formatter in table_formatters.items():
                column_data = table_data[column_name]
                table_data[column_name] = formatter.format_data(column_data)

            data[table_name] = table_data[column_order]

        return data

    def fit(self, data, metadata):
        """Fit the constraint with data and metadata.

        Args:
            data (dict[pd.DataFrame]):
                The data dictionary to fit the constraint on.
            metadata (sdv.Metadata):
                The metadata to fit the constraint on.
        """
        self._validate_constraint_with_metadata(metadata)
        if isinstance(data, pd.DataFrame):
            self._single_table = True
            self._table_name = self._get_single_table_name(metadata)
            data = self._convert_data_to_dictionary(data, metadata)

        self._validate_constraint_with_data(data, metadata)
        self._fit(data, metadata)
        self.metadata = metadata
        self._dtypes = {table: data[table].dtypes.to_dict() for table in metadata.tables}
        self._original_data_columns = {
            table: metadata.tables[table].get_column_names() for table in metadata.tables
        }
        self._fit_constraint_column_formatters(data)
        self._fitted = True

    def _transform(self, data):
        raise NotImplementedError

    def transform(self, data):
        """Transform the data.

        Args:
            data (dict[str, pd.DataFrame])
                The input data dictionary to be transformed.
        """
        if not self._fitted:
            raise NotFittedError('Constraint must be fit using ``fit`` before transforming.')

        self.validate(data)
        data = self._convert_data_to_dictionary(data, self.metadata, copy=True)
        transformed_data = self._transform(data)
        if self._single_table:
            return transformed_data[self._table_name]

        return transformed_data

    def _reverse_transform(self, data):
        raise NotImplementedError

    def _table_as_type_by_col(self, reverse_transformed, table, table_name):
        """Cast table to given types on a column by column basis.

        Args:
            reverse_transformed (dict[str, pd.DataFrame])
                The reverse transformed data dictionary
            table (pd.DataFrame)
                The reverse transformed table
            table_name (str)
                The name of the table
        """
        for col in table:
            try:
                reverse_transformed[table_name][col] = table[col].astype(
                    self._dtypes[table_name][col]
                )
            except pd.errors.IntCastingNaNError:
                LOGGER.info(
                    "Column '%s' is being converted to float because it contains NaNs.", col
                )
                self._dtypes[table_name][col] = np.dtype('float64')
                reverse_transformed[table_name][col] = table[col].astype(
                    self._dtypes[table_name][col]
                )

    def reverse_transform(self, data):
        """Reverse transform the data back into the original space.

        Args:
            data (dict[str, pd.DataFrame])
                The transformed data dictionary to be reverse transformed.
        """
        data = self._convert_data_to_dictionary(data, self.metadata, copy=True)
        reverse_transformed = self._reverse_transform(data)
        reverse_transformed = self._format_constraint_columns(reverse_transformed)
        for table_name, table in reverse_transformed.items():
            valid_columns = [
                column
                for column in self._original_data_columns[table_name]
                if column in table.columns
            ]
            dtypes = {col: self._dtypes[table_name][col] for col in valid_columns}
            table = table[valid_columns]
            try:
                reverse_transformed[table_name] = table.astype(dtypes)
            except pd.errors.IntCastingNaNError:
                # iterate over the columns and cast individually
                self._table_as_type_by_col(reverse_transformed, table, table_name)

        if self._single_table:
            return reverse_transformed[self._table_name]

        return reverse_transformed

    def _is_valid(self, data, metadata):
        raise NotImplementedError

    def is_valid(self, data, metadata=None):
        """Say whether the given table rows are valid.

        Args:
            data (pd.DataFrame or dict[pd.DataFrame]):
                Table data.

        Returns:
            pd.Series or dict[pd.Series]:
                Series of boolean values indicating if the row is valid for the constraint or not.
        """
        if not self._fitted and metadata is None:
            raise NotFittedError(
                'Constraint must be fit using ``fit`` before determining if data is valid '
                'without providing metadata.'
            )

        metadata = self.metadata if metadata is None else metadata
        data_dict = self._convert_data_to_dictionary(data, metadata)
        is_valid_data = self._is_valid(data_dict, metadata)
        if isinstance(data, pd.DataFrame) or self._single_table:
            table_name = (
                self._get_single_table_name(metadata)
                if getattr(self, '_table_name', None) is None
                else self._table_name
            )
            return is_valid_data[table_name]

        return is_valid_data
