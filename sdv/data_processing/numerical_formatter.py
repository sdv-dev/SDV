"""Formatter for numerical data."""
import logging
import sys

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

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

    def __init__(self, enforce_rounding=False, enforce_min_max_values=False,
                 computer_representation='Float'):
        self.enforce_rounding = enforce_rounding
        self.enforce_min_max_values = enforce_min_max_values
        self.computer_representation = computer_representation

    @staticmethod
    def _learn_rounding_digits(data):
        """Check if data has any decimals."""
        name = data.name
        data = np.array(data)
        roundable_data = data[~(np.isinf(data) | pd.isna(data))]

        # Doesn't contain numbers
        if len(roundable_data) == 0:
            return None

        # Doesn't contain decimal digits
        if ((roundable_data % 1) == 0).all():
            return 0

        # Try to round to fewer digits
        if (roundable_data == roundable_data.round(MAX_DECIMALS)).all():
            for decimal in range(MAX_DECIMALS + 1):
                if (roundable_data == roundable_data.round(decimal)).all():
                    return decimal

        # Can't round, not equal after MAX_DECIMALS digits of precision
        LOGGER.info(
            f"No rounding scheme detected for column '{name}'."
            ' Synthetic data will not be rounded.'
        )
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

        if self.enforce_rounding:
            self._rounding_digits = self._learn_rounding_digits(column)

    def format_data(self, column):
        """Format a column according to the learned format.

        Args:
            column (pd.Series):
                Data to format.

        Returns:
            numpy.ndarray:
                containing the formatted data.
        """
        column = column.copy().to_numpy()
        if self.enforce_min_max_values:
            column = column.clip(self._min_value, self._max_value)
        elif self.computer_representation != 'Float':
            min_bound, max_bound = INTEGER_BOUNDS[self.computer_representation]
            column = column.clip(min_bound, max_bound)

        is_integer = np.dtype(self._dtype).kind == 'i'
        if self.enforce_rounding and self._rounding_digits is not None:
            column = column.round(self._rounding_digits)
        elif is_integer:
            column = column.round(0)

        if pd.isna(column).any() and is_integer:
            return column

        return column.astype(self._dtype)
