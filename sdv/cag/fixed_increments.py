"""FixedIncrements CAG pattern."""

import uuid

import numpy as np
import pandas as pd

from sdv._utils import _create_unique_name
from sdv.cag._errors import PatternNotMetError
from sdv.cag._utils import (
    _validate_table_and_column_names,
    _validate_table_name,
    _validate_columns_in_metadata,
    _remove_columns_from_metadata
)
from copy import deepcopy
from sdv.cag.base import BasePattern
from sdv.cag._utils import _validate_table_name_if_defined
from sdv.constraints.utils import get_mappable_combination
from sdv.metadata import Metadata
from sdv._utils import _convert_to_timedelta, _create_unique_name

class FixedIncrements(BasePattern):
    """
    Args:
        column_name (str or list[str]):
            Name of the column(s).
        increment_value (int):
            The increment that each value in the column must be a multiple of. Must be greater
            than 0.
        table_name (str, optional):
            The name of the table that contains the columns. Optional if the
            data is only a single table. Defaults to None.
    """
    def __init__(self, column_name, increment_value, table_name=None):
        super().__init__()

        if isinstance(column_name, str):
            column_name = [column_name]
        elif not isinstance(column_name, list) or not all(isinstance(col, str) for col in column_name):
            raise ValueError('`column_name` must be a string or list of strings.')

        if increment_value <= 0:
            raise ValueError('`increment_value` must be greater than 0.')

        if increment_value % 1 != 0:
            raise ValueError('`increment_value` must be a whole number.')

        _validate_table_name_if_defined(table_name)

        self.column_name = column_name
        self.increment_value = increment_value
        self.table_name = table_name
        self._dtype = {}

    def _validate_pattern_with_metadata(self, metadata):
        """Validate the pattern is compatible with the provided metadata.

        Validates that:
        - If no table_name is set, the metadata must only contain a single table
        - The column(s) in column_name exist in the table in the metadata.
        - All columns have the numerical sdtype
        """
        _validate_table_name(
            table_name=self.table_name,
            metadata=metadata,
        )
        _validate_columns_in_metadata(
            self._get_single_table_name(metadata),
            columns=self.column_name,
            metadata=metadata,
        )
        table_name = self._get_single_table_name(metadata)
        for column in self.column_name:
            col_sdtype = metadata.tables[table_name].columns[column]['sdtype']
            if col_sdtype != 'numerical':
                raise PatternNotMetError(
                    f"Column '{column}' has an incompatible sdtype ('{col_sdtype}'). The column "
                    "sdtype must be 'numerical'."
                )


    def _validate_pattern_with_data(self, data, metadata):
        """Validate the data is compatible with the pattern.
        Args:
            data (dict[pd.DataFrame]):
                The Table data.

            metadata (sdv.metadata.Metadata):
                The input Metadata
        """
        is_column_valid = []
        for column in self.column_name:
            is_column_valid = self._check_if_divisible(data,
                                                       self._get_single_table_name(metadata),
                                                       column,
                                                       self.increment_value)
        return all(is_column_valid)

    def _get_updated_metadata(self, metadata):
        """Get the updated metadata after applying the pattern to the metadata"""
        table_name = self._get_single_table_name(metadata)
        original_columns = list(metadata.tables[table_name].columns)
        updated_metadata = deepcopy(metadata)
        for column_name in original_columns:
            new_column_name = _create_unique_name(
                f'{column_name}#increment',
                metadata.tables[table_name].columns.keys()
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
            data (pandas.DataFrame):
                The Table data.
        """
        table_name = self._get_single_table_name(metadata)
        for column in self.column_name:
            self._dtype[column] = data[table_name][column].dtype

    def _check_if_divisible(self, data, table_name, column_name, increment_value):
        isnan = pd.isna(data[table_name][column_name])
        is_divisible = data[table_name][column_name] % increment_value == 0
        return is_divisible | isnan

    def _is_valid(self, data):
        """Determine if the data is evenly divisible by the increment.

        Args:
            data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.Series:
                Whether each row is valid.
        """
        return self._check_if_divisible(data,
                                        self.table_name,
                                        self.column_name,
                                        self.increment_value)

    def _transform(self, data):
        """Transform the data.

        The transformation works by dividing each value by the increment.

        Args:
            data (pandas.DataFrame):
                Table data.

        Returns:
            (dict[pd.DataFrame]):
                Transformed data.
        """
        table_name = self._get_single_table_name(self.metadata)
        data[table_name][self.column_name] = (data[table_name][self.column_name] / self.increment_value).astype(
            self._dtype
        )
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

        for column in self.column_name:
            column_data = table_data[column].round()
            new_dtype = self._dtype[column]
            table_data[column] = (column_data * self.increment_value).astype(new_dtype)
        return data