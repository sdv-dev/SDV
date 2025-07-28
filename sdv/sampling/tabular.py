"""SDV Condition class for sampling."""

from collections import defaultdict

import pandas as pd

from sdv.errors import TableNameError


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

    def __init__(self, column_values, num_rows=1, table_name=None):
        if num_rows is None and table_name is None:
            raise ValueError('Table name must be set for `num_rows` to be `None`.')
        elif num_rows is not None and (not isinstance(num_rows, int) or num_rows < 1):
            raise ValueError('`num_rows` must be an integer greater than zero.')

        if table_name and not isinstance(table_name, str):
            raise TableNameError

        self.column_values = column_values
        self.num_rows = num_rows
        self.table_name = table_name

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
            raise TableNameError

        self.dataframe = dataframe.copy()
        self.table_name = table_name

    def get_table_name(self):
        """Get the table name for this condition."""
        return self.table_name

    def get_dataframe(self):
        """Get the dataframe for this condition."""
        return self.dataframe

    def get_num_rows(self):
        """Get the desired number of rows for this condition."""
        return len(self.dataframe)


class MultiTableCondition:
    """MultiTableCondition class.

        This class represents a group of conditions that should be used for sampling
        a multi-table dataset.

    Attributes:
        conditions (list[Condition or DataFrameCondition]):
            A list of Condition or DataFrameCondition objects that should be used during
            sampling.
    """

    def _validate_conditions(self, conditions):
        num_rows_set = False
        table_conditions = defaultdict(list)
        for condition in conditions:
            if isinstance(condition, Condition) or isinstance(condition, DataFrameCondition):
                table_name = getattr(condition, 'table_name', None)
                if table_name is None:
                    raise ValueError(
                        f'Invalid {condition.__class__.__name__}. Condition does not have a '
                        'table name set.'
                    )
                if condition.get_num_rows() is not None:
                    num_rows_set = True

                table_conditions[table_name].append(condition)
            else:
                raise ValueError(
                    f'Invalid condition ({condition}). Conditions must be a `Condition`'
                    'or `DataFrameCondition` object.'
                )

        for table_name, condition_list in table_conditions.items():
            if len(condition_list) > 1:
                if not all(condition.get_num_rows() is not None for condition in condition_list):
                    raise ValueError(
                        f"Multiple conditions found for table '{table_name}'. If multiple "
                        'conditions are supplied for a table, all conditions for that table must '
                        'have `num_rows` set.'
                    )

        if not num_rows_set:
            raise ValueError('At least one condition must have `num_rows` set.')

        return table_conditions

    def __init__(self, conditions):
        self.table_conditions = self._validate_conditions(conditions)
        self.conditions = conditions

    def get_conditions(self):
        """Return the list of all conditions for this condition."""
        return self.conditions.copy()

    def get_table_conditions(self):
        """Get a dict of the list of conditions for each table."""
        return self.table_conditions.copy()
