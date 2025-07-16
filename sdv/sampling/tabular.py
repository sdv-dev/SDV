"""SDV Condition class for sampling."""

import pandas as pd


class Condition:
    """Condition class.

    This class represents a condition that is used for sampling.

    Attributes:
        column_values (dict):
            A dictionary representing the desired conditions. A mapping of
            the column name to column value, which will be satisfied in this
            condition.
        num_rows (int):
            The number of rows to generate for this condition. Defaults to 1.
    """

    column_values = {}
    num_rows = 1

    def __init__(self, column_values, num_rows=1):
        self.column_values = column_values
        self.num_rows = num_rows

    def get_column_values(self):
        """Get the column value mappings in this condition."""
        return self.column_values.copy()

    def get_num_rows(self):
        """Get the desired number of rows for this condition."""
        return self.num_rows


class DataFrameCondition:
    """DataFrameCondition class.

        This class represents a condition that is used for sampling.

    Attributes:
        dataframe (pd.DataFrame):
            A pandas DataFrame representing the desired conditions.
            A copy of the DataFrame is saved on this condition.

        table_name (str):
            The name of the table. Optional, defaults to None.
    """

    def __init__(self, dataframe, table_name=None):
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError('`dataframe` must be a pandas DataFrame object.')

        if table_name and not isinstance(table_name, str):
            raise ValueError('`table_name` must be a string or None.')

        self.dataframe = dataframe.copy()
        self.table_name = table_name

    def get_table_name(self):
        """Get the table name for this condition."""
        return self.table_name

    def get_dataframe(self):
        """Get the dataframe for this condition."""
        return self.dataframe
