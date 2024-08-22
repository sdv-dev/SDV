"""Formatter for numerical data."""

import logging
import sys

import pandas as pd
from rdt.transformers.utils import learn_rounding_digits

LOGGER = logging.getLogger(__name__)

MAX_DECIMALS = sys.float_info.dig - 1
INTEGER_BOUNDS = {
    'Int8': (-(2**7), 2**7 - 1),
    'Int16': (-(2**15), 2**15 - 1),
    'Int32': (-(2**31), 2**31 - 1),
    'Int64': (-(2**63), 2**63 - 1),
    'UInt8': (0, 2**8 - 1),
    'UInt16': (0, 2**16 - 1),
    'UInt32': (0, 2**32 - 1),
    'UInt64': (0, 2**64 - 1),
}


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
        computer_representation (dtype):
            Accepts ``'Int8'``, ``'Int16'``, ``'Int32'``, ``'Int64'``, ``'UInt8'``, ``'UInt16'``,
            ``'UInt32'``, ``'UInt64'``, ``'Float'``.
            Defaults to ``'Float'``.
    """

    _dtype = None
    _min_value = None
    _max_value = None
    _rounding_digits = None

    def __init__(
        self, enforce_rounding=False, enforce_min_max_values=False, computer_representation='Float'
    ):
        self.enforce_rounding = enforce_rounding
        self.enforce_min_max_values = enforce_min_max_values
        self.computer_representation = computer_representation

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

        if self.enforce_rounding:
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
        elif not self.computer_representation.startswith('Float'):
            min_bound, max_bound = INTEGER_BOUNDS[self.computer_representation]
            column = column.clip(min_bound, max_bound)

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
