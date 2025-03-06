"""FixedIncrements CAG pattern."""

from copy import deepcopy

import pandas as pd

from sdv._utils import _create_unique_name
from sdv.cag._errors import PatternNotMetError
from sdv.cag._utils import (
    _get_invalid_rows,
    _remove_columns_from_metadata,
    _validate_columns_in_metadata,
    _validate_table_name,
    _validate_table_name_if_defined,
)
from sdv.cag.base import BasePattern


class FixedIncrements(BasePattern):
    """Ensure that the combinations of values across several columns are the same after sampling.

    Args:
        column_name (str or list[str]):
            Name of the column or a list of column names.
        increment_value (int):
            The increment that each value in the column must be a multiple of. Must be greater
            than 0.
        table_name (str, optional):
            The name of the table that contains the columns. Optional if the
            data is only a single table. Defaults to None.
    """

    @staticmethod
    def _validate_init_inputs(column_name, increment_value, table_name):
        if not isinstance(column_name, str):
            raise ValueError('`column_name` must be a string.')
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

    def _validate_pattern_with_metadata(self, metadata):
        """Validate the pattern is compatible with the provided Metadata.

        Validates that:
            - If no table_name is set, the Metadata must only contain a single table
            - The column_name exist in the table in the Metadata.
            - The column is a numerical sdtype
        """
        _validate_table_name(
            table_name=self.table_name,
            metadata=metadata,
        )
        _validate_columns_in_metadata(
            self._get_single_table_name(metadata),
            columns=[self.column_name],
            metadata=metadata,
        )
        table_name = self._get_single_table_name(metadata)
        col_sdtype = metadata.tables[table_name].columns[self.column_name]['sdtype']
        if col_sdtype != 'numerical':
            raise PatternNotMetError(
                f"Column '{self.column_name}' has an incompatible sdtype ('{col_sdtype}')."
                "The column sdtype must be 'numerical'."
            )

    def _validate_pattern_with_data(self, data, metadata):
        """Validate the data is compatible with the pattern.

        Args:
            data (dict[pd.DataFrame]):
                The data.

            metadata (sdv.metadata.Metadata):
                The input Metadata
        """
        valid = self._check_if_divisible(
            data, self._get_single_table_name(metadata), self.column_name, self.increment_value
        )
        if not valid.all():
            invalid_rows_str = _get_invalid_rows(valid)
            raise PatternNotMetError(
                'The fixed increments requirement has been met because the data is not '
                f"evenly divisible by '{self.increment_value}' or contains NaNs "
                f'for row indices: [{invalid_rows_str}]'
            )

    def _get_updated_metadata(self, metadata):
        """Get the updated metadata after applying the pattern to the metadata."""
        table_name = self._get_single_table_name(metadata)
        original_columns = list(metadata.tables[table_name].columns)
        updated_metadata = deepcopy(metadata)
        new_column_name = _create_unique_name(
            self._fixed_increments_column_name, metadata.tables[table_name].columns.keys()
        )
        updated_metadata.add_column(
            column_name=new_column_name,
            sdtype='numerical',
            table_name=table_name,
        )
        updated_metadata = _remove_columns_from_metadata(
            updated_metadata,
            table_name,
            columns_to_drop=original_columns,
        )
        return updated_metadata

    def _fit(self, data, metadata):
        """Learn the dtype of the column.

        Args:
            data (dict[pd.DataFrame]):
                The data.
        """
        table_name = self._get_single_table_name(metadata)
        self._dtype = data[table_name][self.column_name].dtype

    def _check_if_divisible(self, data, table_name, column_name, increment_value):
        isnan = pd.isna(data[table_name][column_name])
        is_divisible = data[table_name][column_name] % increment_value == 0
        return isnan | is_divisible

    def _is_valid(self, data):
        """Determine if the data is evenly divisible by the increment.

        Args:
            data (dict[pd.DataFrame]):
                The data.

        Returns:
            pandas.Series:
                Whether each row is valid.
        """
        table_name = self._get_single_table_name(self.metadata)
        is_valid = {
            table: pd.Series(True, index=table_data.index)
            for table, table_data in data.items()
            if table != table_name
        }
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

        Convert column(s) to multiples of the increment.

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
        data[table_name] = table_data.drop(columns=self._fixed_increments_column_name)
        return data
