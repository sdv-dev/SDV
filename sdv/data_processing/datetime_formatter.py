"""Formatter for datetime data."""

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from sdv._utils import _get_datetime_format


class DatetimeFormatter:
    """Formatter for datetime data.

    Args:
        datetime_format (str):
            The strftime to use for parsing time. For more information, see
            https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior.
            If ``None`` it will attempt to learn it by itself. Defaults to ``None``.
    """

    def __init__(self, datetime_format=None):
        self.datetime_format = datetime_format

    def learn_format(self, column):
        """Learn the format of a column.

        Args:
            column (pandas.Series):
                Data to learn the format.
        """
        self._dtype = column.dtype
        if self.datetime_format is None:
            self.datetime_format = _get_datetime_format(column)

    def format_data(self, column):
        """Format a column according to the learned format.

        Args:
            column (pd.Series):
                Data to format.

        Returns:
            numpy.ndarray:
                containing the formatted data.
        """
        if self._dtype == column.dtype and is_datetime64_any_dtype(column):
            return column

        if self.datetime_format:
            try:
                datetime_column = pd.to_datetime(column, format=self.datetime_format)
                column = datetime_column.dt.strftime(self.datetime_format)
            except ValueError:
                column = pd.to_datetime(column).dt.strftime(self.datetime_format)

        return column.astype(self._dtype)
