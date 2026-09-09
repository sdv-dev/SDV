"""Formatter for numerical data."""

import logging
import sys

import pandas as pd
from rdt.transformers.utils import learn_rounding_digits

LOGGER = logging.getLogger(__name__)

MAX_DECIMALS = sys.float_info.dig - 1


class NumericalFormatter:
    """Formatter for numerical data.

    Args:
        enforce_rounding (bool):
            Whether or not to learn what place to round to based on the data seen during ``fit``.
            If ``True``, the data returned by ``reverse_transform`` will be rounded to that place.
            Defaults to ``False``.
        enforce_min_max_values (bool):
            Whether or not to clip the data returned by ``reverse_transform`` to the min and
            max values seen during ``fit``.
            Defaults to ``False``.
    """

    _dtype = None
    _min_value = None
    _max_value = None
    _rounding_digits = None

    def __init__(
        self,
        enforce_rounding=False,
        enforce_min_max_values=False,
        decimal_places=None,
    ):
        self.enforce_rounding = enforce_rounding or (decimal_places is not None)
        self.enforce_min_max_values = enforce_min_max_values
        self.decimal_places = decimal_places

    def learn_format(self, column):
        """Learn the format of a column.

        Args:
            column (pandas.Series):
                Data to learn the format.
        """
        self._dtype = column.dtype
        if self.enforce_min_max_values:
            self._min_value = column.min()
            self._max_value = column.max()

        self._rounding_digits = self.decimal_places
        if self.enforce_rounding and self.decimal_places is None:
            self._rounding_digits = learn_rounding_digits(column)

    def format_data(self, column):
        """Format a column according to the learned format.

        Args:
            column (pd.Series):
                Data to format.

        Returns:
            numpy.ndarray:
                containing the formatted data.
        """
        column = column.copy()
        if self.enforce_min_max_values:
            column = column.clip(self._min_value, self._max_value)

        is_integer = pd.api.types.is_integer_dtype(self._dtype)
        np_integer_with_nans = (
            not pd.api.types.is_extension_array_dtype(self._dtype)
            and is_integer
            and pd.isna(column).any()
        )
        if self.enforce_rounding and self._rounding_digits is not None:
            column = column.round(self._rounding_digits)
        elif is_integer:
            column = column.round(0)

        return column.astype(self._dtype if not np_integer_with_nans else 'float64')
