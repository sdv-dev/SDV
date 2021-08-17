"""Utils for testing."""
import pandas as pd


class DataFrameMatcher:
    """Match a given Pandas DataFrame in a mock function call."""

    def __init__(self, df):
        self.df = df

    def __eq__(self, other):
        pd.testing.assert_frame_equal(self.df, other)
        return True
