"""FixedIncrements constraint."""

import pandas as pd

from sdv._utils import _create_unique_name
from sdv.cag._errors import ConstraintNotMetError
from sdv.cag._utils import (
    _get_invalid_rows,
    _get_is_valid_dict,
    _remove_columns_from_metadata,
    _validate_table_and_column_names,
    _validate_table_name_if_defined,
)
from sdv.cag.base import BaseConstraint


class FixedIncrements(BaseConstraint):
    """Ensure every value in a column is a multiple of the specified increment.

    Args:
        column_name (str):
            Name of the column.
        increment_value (int):
            The increment that each value in the column must be a multiple of. Must be greater
            than 0 and a whole number.
        table_name (str, optional):
            The name of the table that contains the column. Optional if the
            data is only a single table. Defaults to None.
    """

    @staticmethod
    def _validate_init_inputs(column_name, increment_value, table_name):
        if not isinstance(column_name, str):
            raise ValueError('`column_name` must be a string.')
        if not isinstance(increment_value, (int, float)):
            raise ValueError('`increment_value` must be an integer or float.')
        if table_name and not isinstance(table_name, str):
            raise ValueError('`table_name` must be a string if not None.')

        if increment_value <= 0:
            raise ValueError('`increment_value` must be greater than 0.')
        if increment_value % 1 != 0:
            raise ValueError('`increment_value` must be a whole number.')

        _validate_table_name_if_defined(table_name)

    def __init__(self, column_name, increment_value, table_name=None):
        super().__init__()
        self._validate_init_inputs(column_name, increment_value, table_name)
        self.column_name = column_name
        self.table_name = table_name
        self._fixed_increments_column_name = f'{self.column_name}#increment'
        self.increment_value = increment_value
        self._dtype = None

    def _validate_constraint_with_metadata(self, metadata):
        """Validate the constraint is compatible with the provided Metadata.

        Validates that:
            - If no table_name is set, checks that the Metadata only contains a single table
            - The column_name exist in the table in the Metadata
            - The column_name is a numerical sdtype
        Args:
            metadata (sdv.metadata.Metadata):
                The input Metadata to validate.
        """
        _validate_table_and_column_names(
            self.table_name, columns=[self.column_name], metadata=metadata
        )
        table_name = self._get_single_table_name(metadata)
        col_sdtype = metadata.tables[table_name].columns[self.column_name]['sdtype']
        if col_sdtype != 'numerical':
            raise ConstraintNotMetError(
                f"Column '{self.column_name}' has an incompatible sdtype ('{col_sdtype}')."
                " The column sdtype must be 'numerical'."
            )

    def _check_if_divisible(self, data, table_name, column_name, increment_value):
        """Check if a column is divisible by a given increment value.

        Args:
            data (dict[pd.DataFrame]):
                The data.

            table_name (str):
                Name of the table.

            column_name (str):
                Name of the table to check divisibility.

            increment_value (int):
                the number with which divisibility needs to be checked.
        """
        isnan = pd.isna(data[table_name][column_name])
        is_divisible = data[table_name][column_name] % increment_value == 0
        return isnan | is_divisible

    def _validate_constraint_with_data(self, data, metadata):
        """Validate the data is compatible with the constraint.

        Args:
            data (dict[pd.DataFrame]):
                The data to validate

            metadata (sdv.metadata.Metadata):
                The input Metadata to use to validate the data.

        Returns:
            None
        """
        valid = self._check_if_divisible(
            data, self._get_single_table_name(metadata), self.column_name, self.increment_value
        )
        if not valid.all():
            invalid_rows_str = _get_invalid_rows(valid)
            raise ConstraintNotMetError(
                'The fixed increments requirement has not been met because the data is not '
                f"evenly divisible by '{self.increment_value}' for row indices: "
                f'[{invalid_rows_str}]'
            )

    def _get_updated_metadata(self, metadata):
        """Get the updated metadata after applying the constraint to the metadata.

        Args:
            metadata (sdv.metadata.Metadata):
                The input Metadata to apply the constraint to.

        Returns:
            (sdv.metadata.Metadata): The updated Metadata with the constraint applied.
        """
        table_name = self._get_single_table_name(metadata)
        increments_column = _create_unique_name(
            self._fixed_increments_column_name, metadata.tables[table_name].columns.keys()
        )
        metadata = metadata.to_dict()
        metadata['tables'][table_name]['columns'][increments_column] = {'sdtype': 'numerical'}
        return _remove_columns_from_metadata(
            metadata, table_name, columns_to_drop=[self.column_name]
        )

    def _fit(self, data, metadata):
        """Learn the dtype of the column.

        Args:
            data (dict[pd.DataFrame]):
                The data.

            metadata (sdv.metadata.Metadata):
                The input Metadata.
        """
        table_name = self._get_single_table_name(metadata)
        self._dtype = data[table_name][self.column_name].dtype

    def _is_valid(self, data):
        """Determine if the data is evenly divisible by the increment.

        Args:
            data (dict[pd.DataFrame]):
                The data.

        Returns:
            (dict[pd.DataFrame]):
                For the specified table and column, returns a Series
                which specifies if that row is evenly divisible or
                not by the increment. The length of the Series
                will be equal to the length of the input column.
                The length of the dictionary will be equal to the
                number of tables in the data and contain the same
                table names.
        """
        table_name = self._get_single_table_name(self.metadata)
        is_valid = _get_is_valid_dict(data, table_name)
        valid = self._check_if_divisible(data, table_name, self.column_name, self.increment_value)
        is_valid[table_name] = valid
        return is_valid

    def _transform(self, data):
        """Transform the data.

        The transformation works by dividing each value by the increment.

        Args:
            data (dict[pd.DataFrame]):
                The data.

        Returns:
            (dict[pd.DataFrame]):
                Transformed data.
        """
        table_name = self._get_single_table_name(self.metadata)
        table_data = data[table_name]
        self._fixed_increments_column_name = _create_unique_name(
            self._fixed_increments_column_name, table_data.columns
        )
        table_data[self._fixed_increments_column_name] = (
            table_data[self.column_name] / self.increment_value
        ).astype(self._dtype)
        data[table_name] = table_data.drop(columns=self.column_name)
        return data

    def _reverse_transform(self, data):
        """Reverse transform the data.

        Convert column to a multiple of the increment.

        Args:
            data (dict[pd.DataFrame)]:
                Transformed data.

        Returns:
            dict[pd.DataFrame]:
                Reverse transformed data.
        """
        table_name = self._get_single_table_name(self.metadata)
        table_data = data[table_name]

        column_data = table_data[self._fixed_increments_column_name].round()
        table_data[self.column_name] = (column_data * self.increment_value).astype(self._dtype)
        data[table_name] = table_data.drop(columns=[self._fixed_increments_column_name])
        return data
