from unittest.mock import patch

import numpy as np
import pandas as pd

from sdv.utils import convert_to_timedelta
from tests.utils import SeriesMatcher


@patch('sdv.utils.pd.to_timedelta')
def test_convert_to_timedelta(to_timedelta_mock):
    """Test that nans and values are properly converted to timedeltas."""
    # Setup
    column = pd.Series([7200, 3600, np.nan])
    to_timedelta_mock.return_value = pd.Series([
        pd.Timedelta(hours=1),
        pd.Timedelta(hours=2),
        pd.Timedelta(hours=0)
    ], dtype='timedelta64[ns]')

    # Run
    converted_column = convert_to_timedelta(column)

    # Assert
    to_timedelta_mock.assert_called_with(SeriesMatcher(pd.Series([7200, 3600, 0.0])))
    expected_column = pd.Series([
        pd.Timedelta(hours=1),
        pd.Timedelta(hours=2),
        pd.NaT
    ], dtype='timedelta64[ns]')
    pd.testing.assert_series_equal(converted_column, expected_column)
