"""Constraint utility functions."""

from datetime import datetime

import pandas as pd


def is_datetime_type(val):
    """Determine if the input is a datetime type or not.

    Args:
        val (pandas.DataFrame, int or datetime):
            Input to evaluate.

    Returns:
        bool:
            True if the input is a datetime type, False if not.
    """
    return (
        pd.api.types.is_datetime64_any_dtype(val)
        or isinstance(val, pd.Timestamp)
        or isinstance(val, datetime)
    )
