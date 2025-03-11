"""One Hot Encoding CAG pattern."""

import numpy as np
import pandas as pd

from sdv.cag._errors import PatternNotMetError
from sdv.cag._utils import _get_invalid_rows, _is_list_of_strings, _validate_table_and_column_names
from sdv.cag.base import BasePattern


class OneHotEncoding(BasePattern):
    """Ensure the appropriate columns are one hot encoded.

    This constraint allows the user to specify a list of columns where each row
    is a one hot vector. During the reverse transform, the output of the model
    is transformed so that the column with the largest value is set to 1 while
    all other columns are set to 0.

    Args:
        column_names (list[str]):
            Names of the columns containing one hot rows.
        table_name (str, optional):
            The name of the table that contains the columns. Optional if the
            data is only a single table. Defaults to None.
    """

    @staticmethod
    def _validate_init_inputs(column_names, table_name):
        if not _is_list_of_strings(column_names):
            raise ValueError('`column_names` must be a list of strings.')

        if table_name and not isinstance(table_name, str):
            raise ValueError('`table_name` must be a string or None.')

    def __init__(self, column_names, table_name=None):
        super().__init__()
        self._validate_init_inputs(column_names, table_name)
        self._column_names = column_names
        self.table_name = table_name

    def _validate_pattern_with_metadata(self, metadata):
        """Validate the pattern is compatible with the provided metadata.

        Validates that:
        - If no table_name is provided the metadata contains a single table
        - All input columns exist in the table in the metadata.

        Args:
            metadata (Metadata):
                The metadata to validate against.

        Raises:
            PatternNotMetError:
                If any of the validations fail.
        """
        _validate_table_and_column_names(self.table_name, self._column_names, metadata)

    def _get_valid_table_data(self, table_data):
        one_hot_data = table_data[self._column_names]

        sum_one = one_hot_data.sum(axis=1) == 1.0
        max_one = one_hot_data.max(axis=1) == 1.0
        min_zero = one_hot_data.min(axis=1) == 0.0
        no_nans = ~one_hot_data.isna().any(axis=1)

        return sum_one & max_one & min_zero & no_nans

    def _validate_pattern_with_data(self, data, metadata):
        """Validate the data is compatible with the pattern."""
        table_name = self._get_single_table_name(metadata)
        valid = self._get_valid_table_data(data[table_name])
        if not valid.all():
            invalid_rows_str = _get_invalid_rows(valid)
            raise PatternNotMetError(
                f'The one hot encoding requirement is not met for row indices: [{invalid_rows_str}]'
            )

    def _fit(self, data, metadata):
        """Fit the pattern.

        Args:
            data (dict[str, pd.DataFrame]):
                Table data.
            metadata (Metadata):
                Metadata.
        """
        pass

    def _transform(self, data):
        """Transform the data.

        Args:
            data (dict[str, pd.DataFrame]):
                Table data.

        Returns:
            dict[str, pd.DataFrame]:
                Transformed data.
        """
        return data

    def _reverse_transform(self, data):
        """Reverse transform the table data.

        Set the column with the largest value to one, set all other columns to zero.

        Args:
            data (dict[str, pd.DataFrame]):
                Table data.

        Returns:
            dict[str, pd.DataFrame]:
                Transformed data.
        """
        table_name = self._get_single_table_name(self.metadata)
        table_data = data[table_name]
        one_hot_data = table_data[self._column_names]
        transformed_data = np.zeros_like(one_hot_data.to_numpy())
        max_category_indices = np.argmax(one_hot_data.to_numpy(), axis=1)
        transformed_data[np.arange(len(one_hot_data)), max_category_indices] = 1
        table_data[self._column_names] = transformed_data
        data[table_name] = table_data

        return data

    def _is_valid(self, data):
        """Check whether the data satisfies the one-hot constraint.

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
        is_valid[table_name] = self._get_valid_table_data(data[table_name])

        return is_valid
