"""FixedCombinations CAG pattern."""

import uuid

import numpy as np
import pandas as pd

from sdv._utils import _convert_to_timedelta, _create_unique_name
from sdv.cag._errors import PatternNotMetError
from sdv.cag._utils import _validate_table_and_column_names
from sdv.cag.base import BasePattern
from sdv.constraints.utils import cast_to_datetime64, compute_nans_column, get_datetime_diff, get_mappable_combination, match_datetime_precision, revert_nans_columns
from sdv.metadata import Metadata


class Inequality(BasePattern):
    """Pattern that ensures `high_column_name` is greater than `low_column_name` .

    The transformation works by creating a column with the difference between the
    `high_column_name` and `low_column_name` columns and storing it in the
    `high_column_name`'s place. The reverse transform adds the difference column
    and the `low_column_name` to reconstruct the `high_column_name`.

    Args:
        low_column_name (str):
            Name of the column that contains the low values.
        high_column_name (str):
            Name of the column that contains the high values.
        strict_boundaries (bool):
            Whether the comparison of the values should be strict ``>=`` or
            not ``>``. Defaults to False.
        table_name (str, optional):
            The name of the table that contains the columns. Optional if the
            data is only a single table. Defaults to None.
    """

    @staticmethod
    def _validate_init_inputs(low_column_name, high_column_name, strict_boundaries):
        if not (isinstance(low_column_name, str) and isinstance(high_column_name, str)):
            raise ValueError('`low_column_name` and `high_column_name` must be strings.')

        if not isinstance(strict_boundaries, bool):
            raise ValueError('`strict_boundaries` must be a boolean.')

    def __init__(self, low_column_name, high_column_name, strict_boundaries=False, table_name=None):
        super().__init__()
        self._validate_init_inputs(low_column_name, high_column_name, strict_boundaries)
        self._low_column_name = low_column_name
        self._high_column_name = high_column_name
        self._diff_column_name = f'{self._low_column_name}#{self._high_column_name}'
        self._operator = np.greater if strict_boundaries else np.greater_equal
        self.constraint_columns = (low_column_name, high_column_name)
        self._dtype = None
        self._is_datetime = None
        self._low_datetime_format = None
        self._high_datetime_format = None
        self._nan_column_name = None
        self.table_name = table_name

    def _validate_pattern_with_metadata(self, metadata):
        """Validate the pattern is compatible with the provided metadata.

        Validates that:
        - If no table_name is provided, the metadata must only contain a single table (this should be considered the target table)
        - Validate that both the low_column_name and high_column_name columns exist in the table in the metadata.
        - Validate that both the low_column_name and high_column_name have the same sdtype, and that it is either numerical or datetime

        Args:
            metadata (Metadata):
                The metadata to validate against.

        Raises:
            ValueError:
                If any of the validations fail.
        """
        columns = [self._low_column_name, self._high_column_name]
        _validate_table_and_column_names(self.table_name, columns, metadata)
        table_name = self._get_single_table_name(metadata)
        for column in columns:
            col_sdtype = metadata.tables[table_name].columns[column]['sdtype']
            if col_sdtype not in ['numerical', 'datetime']:
                raise PatternNotMetError(
                    f"Column '{column}' has an incompatible sdtype ('{col_sdtype}'). The column "
                    "sdtype must be either 'numerical' or 'datetime'."
                )

        low_column_sdtype = metadata.tables[table_name].columns[self._low_column_name]['sdtype']
        high_column_sdtype = metadata.tables[table_name].columns[self._high_column_name]['sdtype']
        if low_column_sdtype != high_column_sdtype:
            raise PatternNotMetError(
                f"Columns {self._low_column_name} and {self._high_column_name} must have the same sdtype. "
                f"Found {low_column_sdtype} and {high_column_sdtype}."
            )

    def _get_data(self, table_data):
        low = table_data[self._low_column_name].to_numpy()
        high = table_data[self._high_column_name].to_numpy()
        return low, high

    def _validate_pattern_with_data(self, data, metadata):
        """Validate the data is compatible with the pattern.

        Check whether `high` is greater than `low` in each row.
        """
        low, high = self._get_data(data)
        if self._is_datetime and self._dtype == 'O':
            low = cast_to_datetime64(low, self._low_datetime_format)
            high = cast_to_datetime64(high, self._high_datetime_format)

            format_matches = bool(self._low_datetime_format == self._high_datetime_format)
            if not format_matches:
                low, high = match_datetime_precision(
                    low=low,
                    high=high,
                    low_datetime_format=self._low_datetime_format,
                    high_datetime_format=self._high_datetime_format,
                )

        valid = pd.isna(low) | pd.isna(high) | self._operator(high, low)

        return valid

    def _get_updated_metadata(self, metadata):
        """Get the new output metadata after applying the pattern to the input metadata."""
        table_name = self._get_single_table_name(metadata)
        diff_column = _create_unique_name(
            f'{self._low_column_name}#{self._high_column_name}',
            metadata.tables[table_name].columns.keys()
        )

        metadata = metadata.to_dict()
        metadata['tables'][table_name]['columns'][diff_column] = {'sdtype': 'numerical'}
        del metadata['tables'][table_name]['columns'][self._high_column_name]

        column_set = {self._high_column_name}
        metadata['tables'][table_name]['relationships'] = [
            rel
            for rel in metadata['tables'][table_name].get('relationships', {})
            if {rel['parent_table_name'], rel['child_table_name']}.isdisjoint(column_set)
        ]

        return Metadata.load_from_dict(metadata)

    def _fit(self, data, metadata):
        """Learn the ``dtype`` of ``_high_column_name`` and whether the data is datetime.

        Args:
            data (pandas.DataFrame):
                The table data.
        """
        table_name = self._get_single_table_name(metadata)
        table_data = data[table_name]
        self._validate_columns_exist(table_data)
        self._dtype = table_data[self._high_column_name].dtypes
        self._is_datetime = self._get_is_datetime()
        if self._is_datetime:
            self._low_datetime_format = metadata.tables[table_name].columns[self._low_column_name].get(
                'datetime_format'
            )
            self._high_datetime_format = metadata.tables[table_name].columns[self._high_column_name].get(
                'datetime_format'
            )

    def _transform(self, data):
        """Transform the table data.

        The transformation consists on replacing the ``high_column_name`` values with the
        difference between it and the ``low_column_name`` values.

        Afterwards, a logarithm is applied to the difference + 1 to ensure that the
        value stays positive when reverted afterwards using an exponential.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        table_name = self._get_single_table_name(self.metadata)
        table_data = data[table_name]
        low, high = self._get_data(table_data)
        if self._is_datetime:
            diff_column = get_datetime_diff(
                high=high,
                low=low,
                high_datetime_format=self._high_datetime_format,
                low_datetime_format=self._low_datetime_format,
            )
        else:
            diff_column = high - low

        self._diff_column_name = _create_unique_name(self._diff_column_name, table_data.columns)
        table_data[self._diff_column_name] = np.log(diff_column + 1)

        nan_col = compute_nans_column(table_data, [self._low_column_name, self._high_column_name])
        if nan_col is not None:
            self._nan_column_name = _create_unique_name(nan_col.name, table_data.columns)
            table_data[self._nan_column_name] = nan_col
            if self._is_datetime:
                mean_value_low = table_data[self._low_column_name].mode()[0]
            else:
                mean_value_low = table_data[self._low_column_name].mean()
            table_data = table_data.fillna({
                self._low_column_name: mean_value_low,
                self._diff_column_name: table_data[self._diff_column_name].mean(),
            })

        return table_data.drop(self._high_column_name, axis=1)

    def _reverse_transform(self, data):
        """Reverse transform the table data.

        The transformation is reversed by computing an exponential of the difference value,
        subtracting 1 and converting it to the original dtype. Finally, the obtained column
        is added to the ``low_column_name`` column to get back the original ``high_column_name``
        value.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        table_name = self._get_single_table_name(self.metadata)
        table_data = data[table_name]
        diff_column = np.exp(table_data[self._diff_column_name]) - 1
        if self._dtype != np.dtype('float'):
            diff_column = diff_column.round()

        if self._is_datetime:
            diff_column = _convert_to_timedelta(diff_column)

        low = table_data[self._low_column_name].to_numpy()
        if self._is_datetime and self._dtype == 'O':
            low = cast_to_datetime64(low)

        table_data[self._high_column_name] = pd.Series(diff_column + low).astype(self._dtype)

        if self._nan_column_name and self._nan_column_name in table_data.columns:
            table_data = revert_nans_columns(table_data, self._nan_column_name)

        return table_data.drop(self._diff_column_name, axis=1)

    def is_valid(self, table_data):
        """Check whether `high` is greater than `low` in each row.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.Series:
                Whether each row is valid.
        """
        low, high = self._get_data(table_data)
        if self._is_datetime and self._dtype == 'O':
            low = cast_to_datetime64(low, self._low_datetime_format)
            high = cast_to_datetime64(high, self._high_datetime_format)

            format_matches = bool(self._low_datetime_format == self._high_datetime_format)
            if not format_matches:
                low, high = match_datetime_precision(
                    low=low,
                    high=high,
                    low_datetime_format=self._low_datetime_format,
                    high_datetime_format=self._high_datetime_format,
                )

        valid = pd.isna(low) | pd.isna(high) | self._operator(high, low)
        return valid
