"""Range CAG pattern."""

import operator

import numpy as np
import pandas as pd

from sdv._utils import _convert_to_timedelta, _create_unique_name
from sdv.cag._errors import PatternNotMetError
from sdv.cag._utils import _validate_table_and_column_names
from sdv.cag.base import BasePattern
from sdv.constraints.utils import (
    cast_to_datetime64,
    compute_nans_column,
    get_datetime_diff,
    revert_nans_columns,
)
from sdv.metadata import Metadata


class Range(BasePattern):
    """Ensure that the `middle_column_name` is between `low_column_name` and `high_column_name`.

    The transformation strategy works the same as the Inequality constraint but with two
    columns instead of one. We compute the difference between the `middle_column_name`
    and the `low_column_name` column and then apply a logarithm to the difference + 1 to ensure
    that the value stays positive when reverted afterwards using an exponential.
    We do the same for the difference between the `high_column_name` and `middle_column_name`.

    Args:
        low_column_name (str):
            Name of the column which will be the lower bound.
        middle_column_name (str):
            Name of the column that has to be between the lower bound and upper bound.
        high_column_name (str):
            Name of the column which will be the higher bound.
        strict_boundaries (bool):
            Whether the comparison of the values should be strict `>=` or
            not `>` when comparing them.
            Defaults to True.
        table_name (str, optional):
            The name of the table that contains the columns. Optional if the
            data is only a single table. Defaults to None.
    """

    @staticmethod
    def _validate_init_inputs(
        low_column_name,
        middle_column_name,
        high_column_name,
        strict_boundaries,
        table_name,
    ):
        if not (
            isinstance(low_column_name, str)
            and isinstance(middle_column_name, str)
            and isinstance(high_column_name, str)
        ):
            raise ValueError(
                '`low_column_name`, `middle_column_name` and `high_column_name` must be strings.'
            )

        if not isinstance(strict_boundaries, bool):
            raise ValueError('`strict_boundaries` must be a boolean.')

        if table_name and not isinstance(table_name, str):
            raise ValueError('`table_name` must be a string or None.')

    def __init__(
        self,
        low_column_name,
        middle_column_name,
        high_column_name,
        strict_boundaries=True,
        table_name=None,
    ):
        super().__init__()
        self._validate_init_inputs(
            low_column_name,
            middle_column_name,
            high_column_name,
            strict_boundaries,
            table_name,
        )
        self._low_column_name = low_column_name
        self._middle_column_name = middle_column_name
        self._high_column_name = high_column_name
        self._low_diff_column_name = f'{self._low_column_name}#{self._middle_column_name}'
        self._high_diff_column_name = f'{self._middle_column_name}#{self._high_column_name}'
        self._operator = operator.lt if strict_boundaries else operator.le
        self.table_name = table_name

        # Set during fit
        self._is_datetime = None
        self._dtype = None
        self._low_datetime_format = None
        self._middle_datetime_format = None
        self._high_datetime_format = None

        # Set during transform
        self._nan_column_name = None

    def _validate_pattern_with_metadata(self, metadata):
        """Validate the pattern is compatible with the provided metadata.

        Validates that:
        - If no table_name is provided the metadata contains a single table
        - All input columns exist in the table in the metadata.
        - All columns have the same sdtype, and that it is either numerical or datetime

        Args:
            metadata (Metadata):
                The metadata to validate against.

        Raises:
            ValueError:
                If any of the validations fail.
        """
        columns = [self._low_column_name, self._middle_column_name, self._high_column_name]
        _validate_table_and_column_names(self.table_name, columns, metadata)
        table_name = self._get_single_table_name(metadata)
        for column in columns:
            col_sdtype = metadata.tables[table_name].columns[column]['sdtype']
            if col_sdtype not in ['numerical', 'datetime']:
                raise PatternNotMetError(
                    f"Column '{column}' has an incompatible sdtype '{col_sdtype}'. The column "
                    "sdtype must be either 'numerical' or 'datetime'."
                )

        low_column_sdtype = metadata.tables[table_name].columns[self._low_column_name]['sdtype']
        mid_column_sdtype = metadata.tables[table_name].columns[self._middle_column_name]['sdtype']
        high_column_sdtype = metadata.tables[table_name].columns[self._high_column_name]['sdtype']
        if low_column_sdtype != mid_column_sdtype or mid_column_sdtype != high_column_sdtype:
            raise PatternNotMetError(
                f"Columns '{self._low_column_name}', '{self._middle_column_name}' and "
                f"'{self._high_column_name}' must have the same sdtype. Found '{low_column_sdtype}'"
                f", '{mid_column_sdtype}' and '{high_column_sdtype}'."
            )

    def _get_data(self, data):
        low = data[self._low_column_name].to_numpy()
        mid = data[self._middle_column_name].to_numpy()
        high = data[self._high_column_name].to_numpy()
        return low, mid, high

    def _validate_pattern_with_data(self, data, metadata):
        """Validate the data is compatible with the pattern."""
        table_name = self._get_single_table_name(metadata)
        data = data[table_name]
        low, mid, high = self._get_data(data)

        low_is_nan = pd.isna(low)
        mid_is_nan = pd.isna(mid)
        high_is_nan = pd.isna(high)

        low_lt_mid = self._operator(low, mid) | low_is_nan | mid_is_nan
        mid_lt_high = self._operator(mid, high) | mid_is_nan | high_is_nan
        low_lt_high = self._operator(low, high) | low_is_nan | high_is_nan

        valid = low_lt_mid & mid_lt_high & low_lt_high

        if not valid.all():
            invalid_rows = np.where(~valid)[0]
            if len(invalid_rows) <= 5:
                invalid_rows_str = ', '.join(str(i) for i in invalid_rows)
            else:
                first_five = ', '.join(str(i) for i in invalid_rows[:5])
                remaining = len(invalid_rows) - 5
                invalid_rows_str = f'{first_five}, +{remaining} more'

            raise PatternNotMetError(
                f'The range requirement is not met for row indices: [{invalid_rows_str}]'
            )

    def _get_updated_metadata(self, metadata):
        """Get the new output metadata after applying the pattern to the input metadata."""
        table_name = self._get_single_table_name(metadata)
        low_diff_column = _create_unique_name(
            self._low_diff_column_name, metadata.tables[table_name].columns.keys()
        )
        high_diff_column = _create_unique_name(
            self._high_diff_column_name, metadata.tables[table_name].columns.keys()
        )

        metadata = metadata.to_dict()
        metadata['tables'][table_name]['columns'][low_diff_column] = {'sdtype': 'numerical'}
        metadata['tables'][table_name]['columns'][high_diff_column] = {'sdtype': 'numerical'}
        del metadata['tables'][table_name]['columns'][self._high_column_name]
        del metadata['tables'][table_name]['columns'][self._middle_column_name]

        metadata['tables'][table_name]['column_relationships'] = [
            rel
            for rel in metadata['tables'][table_name].get('column_relationships', [])
            if self._high_column_name not in rel['column_names']
            and self._middle_column_name not in rel['column_names']
        ]

        return Metadata.load_from_dict(metadata)

    def _get_is_datetime(self, metadata, table_name):
        return metadata.tables[table_name].columns[self._low_column_name]['sdtype'] == 'datetime'

    def _get_datetime_format(self, metadata, table_name, column_name):
        return metadata.tables[table_name].columns[column_name].get('datetime_format')

    def _fit(self, data, metadata):
        """Fit the pattern.

        Args:
            data (dict[str, pd.DataFrame]):
                Table data.
            metadata (Metadata):
                Metadata.
        """
        table_name = self._get_single_table_name(metadata)
        table_data = data[table_name]
        self._dtype = table_data[self._high_column_name].dtypes
        self._is_datetime = self._get_is_datetime(metadata, table_name)
        if self._is_datetime:
            self._low_datetime_format = self._get_datetime_format(
                metadata, table_name, self._low_column_name
            )
            self._middle_datetime_format = self._get_datetime_format(
                metadata, table_name, self._middle_column_name
            )
            self._high_datetime_format = self._get_datetime_format(
                metadata, table_name, self._high_column_name
            )

        self._low_diff_column_name = _create_unique_name(
            self._low_diff_column_name, table_data.columns
        )
        self._high_diff_column_name = _create_unique_name(
            self._high_diff_column_name, table_data.columns
        )

    def _transform(self, data):
        """Transform the data.

        The transformation consists in replacing `middle` and `high` by the difference
        between them and `low` and `middle` respectively. To avoid negative values, the
        logarithm of the difference + 1 is taken.

        Args:
            data (dict[str, pd.DataFrame]):
                Table data.

        Returns:
            dict[str, pd.DataFrame]:
                Transformed data.
        """
        table_name = self._get_single_table_name(self.metadata)
        table_data = data[table_name]
        low, mid, high = self._get_data(table_data)
        if self._is_datetime:
            low_diff_column = get_datetime_diff(
                mid,
                low,
                high_datetime_format=self._middle_datetime_format,
                low_datetime_format=self._low_datetime_format,
                dtype=self._dtype,
            )
            high_diff_column = get_datetime_diff(
                high,
                mid,
                high_datetime_format=self._high_datetime_format,
                low_datetime_format=self._middle_datetime_format,
                dtype=self._dtype,
            )
        else:
            low_diff_column = mid - low
            high_diff_column = high - mid

        table_data[self._low_diff_column_name] = np.log(low_diff_column + 1)
        table_data[self._high_diff_column_name] = np.log(high_diff_column + 1)

        list_columns = [self._low_column_name, self._middle_column_name, self._high_column_name]
        nan_column = compute_nans_column(table_data, list_columns)
        if nan_column is not None:
            self._nan_column_name = _create_unique_name(nan_column.name, table_data.columns)
            table_data[self._nan_column_name] = nan_column
            if self._is_datetime:
                mean_value_low = table_data[self._low_column_name].mode()[0]
            else:
                mean_value_low = table_data[self._low_column_name].mean()

            table_data = table_data.fillna({
                self._low_column_name: mean_value_low,
                self._low_diff_column_name: table_data[self._low_diff_column_name].mean(),
                self._high_diff_column_name: table_data[self._high_diff_column_name].mean(),
            })

        data[table_name] = table_data.drop(
            [self._middle_column_name, self._high_column_name], axis=1
        )

        return data

    def _reverse_transform(self, data):
        """Reverse transform the table data.

        The reverse transformation consists in replacing `low_diff_column` and
        `high_diff_column` by the sum of `low` and `low_diff_column` and `middle`
        and `high_diff_column` respectively.

        Args:
            data (dict[str, pd.DataFrame]):
                Table data.

        Returns:
            dict[str, pd.DataFrame]:
                Transformed data.
        """
        table_name = self._get_single_table_name(self.metadata)
        table_data = data[table_name]
        low_diff_column = np.exp(table_data[self._low_diff_column_name]) - 1
        high_diff_column = np.exp(table_data[self._high_diff_column_name]) - 1
        if self._dtype != np.dtype('float'):
            low_diff_column = low_diff_column.round()
            high_diff_column = high_diff_column.round()

        if self._is_datetime:
            low_diff_column = _convert_to_timedelta(low_diff_column)
            high_diff_column = _convert_to_timedelta(high_diff_column)

        low = table_data[self._low_column_name].to_numpy()
        if self._is_datetime and self._dtype == 'O':
            low = cast_to_datetime64(low, self._low_datetime_format)

        middle = pd.Series(low_diff_column + low).astype(self._dtype)
        table_data[self._middle_column_name] = middle
        table_data[self._high_column_name] = pd.Series(high_diff_column + middle.to_numpy()).astype(
            self._dtype
        )

        if self._nan_column_name in table_data.columns:
            table_data = revert_nans_columns(table_data, self._nan_column_name)

        data[table_name] = table_data.drop(
            [self._low_diff_column_name, self._high_diff_column_name], axis=1
        )

        return data

    def _is_valid(self, data):
        """Check whether the `middle` column is between the `low` and `high` columns.

        Args:
            data (dict[str, pd.DataFrame]):
                Table data.

        Returns:
            dict[str, pd.Series]:
                Whether each row is valid.
        """
        table_name = self._get_single_table_name(self.metadata)
        is_valid = {
            table: pd.Series(True, index=table_data.index)
            for table, table_data in data.items()
            if table != table_name
        }

        table_data = data[table_name]
        low, mid, high = self._get_data(table_data)
        low_is_nan = pd.isna(low)
        mid_is_nan = pd.isna(mid)
        high_is_nan = pd.isna(high)

        low_lt_middle = self._operator(low, mid) | low_is_nan | mid_is_nan
        mid_lt_high = self._operator(mid, high) | mid_is_nan | high_is_nan
        low_lt_high = self._operator(low, high) | low_is_nan | high_is_nan

        valid = low_lt_middle & mid_lt_high & low_lt_high
        is_valid[table_name] = valid

        return is_valid
