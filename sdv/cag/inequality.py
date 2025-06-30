"""Inequality constraint."""

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype

from sdv._utils import _convert_to_timedelta, _create_unique_name
from sdv.cag._errors import ConstraintNotMetError
from sdv.cag._utils import (
    _get_is_valid_dict,
    _is_list_of_type,
    _remove_columns_from_metadata,
    _validate_table_and_column_names,
    _validate_table_name_if_defined,
)
from sdv.cag.base import BaseConstraint
from sdv.constraints.utils import (
    _warn_if_timezone_aware_formats,
    cast_to_datetime64,
    compute_nans_column,
    get_datetime_diff,
    match_datetime_precision,
    revert_nans_columns,
)


class Inequality(BaseConstraint):
    """Constraint that ensures `high_column_name` is greater than `low_column_name` .

    The transformation works by creating a column with the difference between the
    `high_column_name` and `low_column_name` columns and storing it in the
    `high_column_name`'s place. The reverse transform adds the difference column
    to the `low_column_name` to reconstruct the `high_column_name`.

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
    def _validate_init_inputs(low_column_name, high_column_name, strict_boundaries, table_name):
        if not _is_list_of_type([low_column_name, high_column_name], str):
            raise ValueError('`low_column_name` and `high_column_name` must be strings.')

        if not isinstance(strict_boundaries, bool):
            raise ValueError('`strict_boundaries` must be a boolean.')

        _validate_table_name_if_defined(table_name)

    def __init__(self, low_column_name, high_column_name, strict_boundaries=False, table_name=None):
        super().__init__()
        self._validate_init_inputs(low_column_name, high_column_name, strict_boundaries, table_name)
        self._low_column_name = low_column_name
        self._high_column_name = high_column_name
        self._fillna_low_column_name = f'{low_column_name}.fillna'
        self._diff_column_name = f'{self._low_column_name}#{self._high_column_name}'
        self._nan_column_name = f'{self._diff_column_name}.nan_component'
        self._operator = np.greater if strict_boundaries else np.greater_equal
        self.table_name = table_name

        # Set during fit
        self._is_datetime = None
        self._dtype = None
        self._low_datetime_format = None
        self._high_datetime_format = None

    def _validate_constraint_with_metadata(self, metadata):
        """Validate the constraint is compatible with the provided metadata.

        Validates that:
        - If no table_name is provided, the metadata contains a single table
        - Both the low_column_name and high_column_name columns exist in the table in the metadata
        - Both the low_column_name and high_column_name columns have the same sdtype,
        and that it is either numerical or datetime

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
                raise ConstraintNotMetError(
                    f"Column '{column}' has an incompatible sdtype '{col_sdtype}'. The column "
                    "sdtype must be either 'numerical' or 'datetime'."
                )

        low_column_sdtype = metadata.tables[table_name].columns[self._low_column_name]['sdtype']
        high_column_sdtype = metadata.tables[table_name].columns[self._high_column_name]['sdtype']
        if low_column_sdtype != high_column_sdtype:
            raise ConstraintNotMetError(
                f"Columns '{self._low_column_name}' and '{self._high_column_name}' must have the "
                f"same sdtype. Found '{low_column_sdtype}' and '{high_column_sdtype}'."
            )

    def _get_data(self, data):
        low = data[self._low_column_name].to_numpy()
        high = data[self._high_column_name].to_numpy()
        return low, high

    def _get_is_datetime(self, metadata, table_name):
        return metadata.tables[table_name].columns[self._low_column_name]['sdtype'] == 'datetime'

    def _get_datetime_format(self, metadata, table_name, column_name):
        datetime_format = metadata.tables[table_name].columns[column_name].get('datetime_format')
        return datetime_format

    def _get_valid_table_data(self, table_data, metadata, table_name):
        low, high = self._get_data(table_data)
        is_datetime = self._get_is_datetime(metadata, table_name)
        if is_datetime and is_object_dtype(table_data[self._low_column_name]):
            low_format = self._get_datetime_format(metadata, table_name, self._low_column_name)
            high_format = self._get_datetime_format(metadata, table_name, self._high_column_name)
            low = cast_to_datetime64(low, low_format)
            high = cast_to_datetime64(high, high_format)

            format_matches = bool(low_format == high_format)
            if not format_matches:
                low, high = match_datetime_precision(
                    low=low,
                    high=high,
                    low_datetime_format=low_format,
                    high_datetime_format=high_format,
                )

        return pd.isna(low) | pd.isna(high) | self._operator(high, low)

    def _validate_constraint_with_data(self, data, metadata):
        """Validate the data is compatible with the constraint.

        Validate that the inequality requirement is met between the high and low columns.
        """
        table_name = self._get_single_table_name(metadata)
        valid = self._get_valid_table_data(data[table_name], metadata, table_name)
        if not valid.all():
            invalid_rows = data[table_name].loc[
                ~valid, [self._low_column_name, self._high_column_name]
            ]
            error_message = self._format_error_message_constraint(invalid_rows, table_name)
            raise ConstraintNotMetError(error_message)

    def _get_diff_and_nan_column_names(self, metadata, column_name, table_name):
        """Get the column names for the difference and NaN columns.

        Args:
            metadata (Metadata):
                The metadata to get the column names from.
            column_name (str):
                The input name of the column to be added.
            table_name (str):
                The name of the table that contains the columns.
        """
        column_names = metadata.tables[table_name].columns.keys()
        fillna_low_column = _create_unique_name(f'{self._low_column_name}.fillna', column_names)
        diff_column = _create_unique_name(column_name, column_names)
        nan_diff_column = _create_unique_name(diff_column + '.nan_component', column_names)

        return fillna_low_column, diff_column, nan_diff_column

    def _get_updated_metadata(self, metadata):
        """Get the new output metadata after applying the constraint to the input metadata."""
        table_name = self._get_single_table_name(metadata)
        fillna_low_column, diff_column, nan_diff_column = self._get_diff_and_nan_column_names(
            metadata, self._diff_column_name, table_name
        )

        metadata = metadata.to_dict()
        original_col_metadata = metadata['tables'][table_name]['columns'][self._low_column_name]
        metadata['tables'][table_name]['columns'][fillna_low_column] = original_col_metadata
        metadata['tables'][table_name]['columns'][diff_column] = {'sdtype': 'numerical'}
        metadata['tables'][table_name]['columns'][nan_diff_column] = {'sdtype': 'categorical'}
        return _remove_columns_from_metadata(
            metadata, table_name, columns_to_drop=[self._low_column_name, self._high_column_name]
        )

    def _fit(self, data, metadata):
        """Fit the constraint.

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
        self._fillna_low_column_name, self._diff_column_name, self._nan_column_name = (
            self._get_diff_and_nan_column_names(metadata, self._diff_column_name, table_name)
        )
        if self._is_datetime:
            self._low_datetime_format = self._get_datetime_format(
                metadata, table_name, self._low_column_name
            )
            self._high_datetime_format = self._get_datetime_format(
                metadata, table_name, self._high_column_name
            )
            formats = [self._low_datetime_format, self._high_datetime_format]
            _warn_if_timezone_aware_formats(formats)

    def _transform(self, data):
        """Transform the data.

        The transformation consists on replacing the `high_column_name` values with the
        difference between it and the `low_column_name` values.

        Afterwards, a logarithm is applied to the difference + 1 to ensure that the
        value stays positive when reverted afterwards using an exponential.

        Args:
            data (dict[str, pd.DataFrame]):
                Table data.

        Returns:
            dict[str, pd.DataFrame]:
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

        table_data[self._diff_column_name] = np.log(diff_column + 1)
        nan_col = compute_nans_column(table_data, [self._low_column_name, self._high_column_name])
        table_data[self._nan_column_name] = nan_col
        if self._is_datetime:
            mean_value_low = table_data[self._low_column_name].mode()[0]
        else:
            mean_value_low = table_data[self._low_column_name].mean()

        table_data = table_data.fillna({
            self._low_column_name: mean_value_low,
            self._diff_column_name: table_data[self._diff_column_name].mean(),
        }).rename(columns={self._low_column_name: self._fillna_low_column_name})

        data[table_name] = table_data.drop(self._high_column_name, axis=1)

        return data

    def _reverse_transform(self, data):
        """Reverse transform the table data.

        The transformation is reversed by computing an exponential of the difference value,
        subtracting 1 and converting it to the original dtype. Finally, the obtained column
        is added to the `low_column_name` column to get back the original `high_column_name`
        value.

        Args:
            data (dict[str, pd.DataFrame]):
                Table data.

        Returns:
            dict[str, pd.DataFrame]:
                Transformed data.
        """
        table_name = self._get_single_table_name(self.metadata)
        table_data = data[table_name].rename(
            columns={self._fillna_low_column_name: self._low_column_name}
        )
        diff_column = np.exp(table_data[self._diff_column_name]) - 1
        if self._dtype != np.dtype('float'):
            diff_column = diff_column.round()

        if self._is_datetime:
            diff_column = _convert_to_timedelta(diff_column)

        low = table_data[self._low_column_name].to_numpy()
        if self._is_datetime and is_object_dtype(self._dtype):
            table_data[self._low_column_name] = low = cast_to_datetime64(low)

        table_data[self._high_column_name] = pd.Series(diff_column + low).astype(self._dtype)
        table_data = revert_nans_columns(table_data, self._nan_column_name)

        data[table_name] = table_data.drop(self._diff_column_name, axis=1)

        return data

    def _is_valid(self, data, metadata):
        """Check whether `high` is greater than `low` in each row.

        Args:
            data (dict[str, pd.DataFrame]):
                Table data.

        Returns:
            dict[str, pd.Series]:
                Whether each row is valid.
        """
        table_name = self._get_single_table_name(metadata)
        is_valid = _get_is_valid_dict(data, table_name)
        valid_table_rows = self._get_valid_table_data(data[table_name], metadata, table_name)
        is_valid[table_name] = valid_table_rows

        return is_valid
