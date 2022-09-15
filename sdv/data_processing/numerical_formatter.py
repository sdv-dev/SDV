"""Formatter for numerical data."""
import sys

import numpy as np
import pandas as pd

MAX_DECIMALS = sys.float_info.dig - 1
INTEGER_BOUNDS = {
    'Int8': (-2**7, 2**7 - 1),
    'Int16': (-2**15, 2**15 - 1),
    'Int32': (-2**31, 2**31 - 1),
    'Int64': (-2**63, 2**63 - 1),
    'UInt8': (0, 2**8 - 1),
    'UInt16': (0, 2**16 - 1),
    'UInt32': (0, 2**32 - 1),
    'UInt64': (0, 2**64 - 1),
}


class NumericalFormatter:
    """Formatter for numerical data.

    Args:
        learn_rounding_scheme (bool):
            Whether or not to learn what place to round to based on the data seen during ``fit``.
            If ``True``, the data returned by ``reverse_transform`` will be rounded to that place.
            Defaults to ``False``.
        enforce_min_max_values (bool):
            Whether or not to clip the data returned by ``reverse_transform`` to the min and
            max values seen during ``fit``.
            Defaults to ``False``.
        representation (dtype):
            Accepts ``'Int8'``, ``'Int16'``, ``'Int32'``, ``'Int64'``, ``'UInt8'``, ``'UInt16'``,
            ``'UInt32'``, ``'UInt64'``, ``'Float'``.
            Defaults to ``'Float'``.
    """

    _dtype = None
    _min_value = None
    _max_value = None
    _rounding_digits = None

    def __init__(self, learn_rounding_scheme=False, enforce_min_max_values=False,
                 representation='Float'):
        self.learn_rounding_scheme = learn_rounding_scheme
        self.enforce_min_max_values = enforce_min_max_values
        self.representation = representation

    @staticmethod
    def _learn_rounding_digits(data):
        # check if data has any decimals
        data = np.array(data)
        roundable_data = data[~(np.isinf(data) | pd.isna(data))]
        if ((roundable_data % 1) != 0).any():
            if (roundable_data == roundable_data.round(MAX_DECIMALS)).all():
                for decimal in range(MAX_DECIMALS + 1):
                    if (roundable_data == roundable_data.round(decimal)).all():
                        return decimal

        return None

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

        if self.learn_rounding_scheme:
            self._rounding_digits = self._learn_rounding_digits(column)

    def format_data(self, column):
        """Format a column according to the learned format.

        Args:
            column (pd.Series):
                Data to format.

        Returns:
            numpy.ndarray containing the formatted data.
        """
        column = column.copy()
        if self.enforce_min_max_values:
            column = column.clip(self._min_value, self._max_value)
        elif self.representation != 'Float':
            min_bound, max_bound = INTEGER_BOUNDS[self.representation]
            column = column.clip(min_bound, max_bound)

        is_integer = np.dtype(self._dtype).kind == 'i'
        if self.learn_rounding_scheme or is_integer:
            column = column.round(self._rounding_digits or 0)

        if pd.isna(column).any() and is_integer:
            return column

        return column.astype(self._dtype)
