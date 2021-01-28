"""Table constraints.

This module contains constraints that are evaluated within a single table,
and which can affect one or more columns at a time, as well as one or more
rows.

Currently implemented constraints are:

    * CustomConstraint: Simple constraint to be set up by passing the python
      functions that will be used for transformation, reverse transformation
      and validation.
    * UniqueCombinations: Ensure that the combinations of values
      across several columns are the same after sampling.
    * GreaterThan: Ensure that the value in one column is always greater than
      the value in another column.
    * ColumnFormula: Compute the value of a column based on applying a formula
      on the other columns of the table.
"""

import numpy as np
import pandas as pd

from sdv.constraints.base import Constraint, import_object


class CustomConstraint(Constraint):
    """Custom Constraint Class.

    This class simply takes the ``transform``, ``reverse_transform``
    and ``is_valid`` methods as optional arguments, so users can
    pass custom functions for each one of them.

    Args:
        transform (callable):
            Function to replace the ``transform`` method.
        reverse_transform (callable):
            Function to replace the ``reverse_transform`` method.
        is_valid (callable):
            Function to replace the ``is_valid`` method.
    """

    def __init__(self, transform=None, reverse_transform=None, is_valid=None):
        if transform is not None:
            self.transform = import_object(transform)

        if reverse_transform is not None:
            self.reverse_transform = import_object(reverse_transform)

        if is_valid is not None:
            self.is_valid = import_object(is_valid)


class UniqueCombinations(Constraint):
    """Ensure that the combinations across multiple colums stay unique.

    One simple example of this constraint can be found in a table that
    contains the columns `country` and `city`, where each country can
    have multiple cities and the same city name can even be found in
    multiple countries, but some combinations of country/city would
    produce invalid results.

    This constraint would ensure that the combinations of country/city
    found in the sampled data always stay within the combinations previously
    seen during training.

    Args:
        columns (list[str]):
            Names of the columns that need to produce unique combinations.
        handling_strategy (str):
            How this Constraint should be handled, which can be ``transform``,
            ``reject_sampling`` or ``all``. Defaults to ``transform``.
    """

    _separator = None
    _joint_column = None

    def __init__(self, columns, handling_strategy='transform'):
        self._columns = columns
        super().__init__(handling_strategy)

    def _valid_separator(self, table_data):
        """Return True if separator is valid for this data.

        If the separator is contained within any of the columns
        or the column name obtained after joining the column
        names using the separator already exists, the separator
        is not valid.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            bool:
                Whether the separator is valid for this data or not.
        """
        for column in self._columns:
            if table_data[column].str.contains(self._separator).any():
                return False

            if self._separator.join(self._columns) in table_data:
                return False

        return True

    def fit(self, table_data):
        """Fit this Constraint to the data.

        The fit process consists on:

            - Finding a separtor that works for the
              current data by iteratively adding `#` to it.
            - Generating the joint column name by concatenating
              the names of ``self._columns`` with the separator.

        Args:
            table_data (pandas.DataFrame):
                Table data.
        """
        self._separator = '#'
        while not self._valid_separator(table_data):
            self._separator += '#'

        self._joint_column = self._separator.join(self._columns)
        self._combinations = table_data[self._columns].drop_duplicates().copy()

    def is_valid(self, table_data):
        """Say whether the column values are within the original combinations.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.Series:
                Whether each row is valid.
        """
        merged = table_data.merge(
            self._combinations,
            how='left',
            on=self._columns,
            indicator=self._joint_column
        )
        return merged[self._joint_column] == 'both'

    def transform(self, table_data):
        """Transform the table data.

        The transformation consist on removing all the ``self._columns`` from
        the dataframe, concatenating them using the found separator, and
        setting them back to the data as a single name with the previously
        computed name.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        lists_series = pd.Series(table_data[self._columns].values.tolist())
        table_data = table_data.drop(self._columns, axis=1)
        table_data[self._joint_column] = lists_series.str.join(self._separator)

        return table_data

    def reverse_transform(self, table_data):
        """Reverse transform the table data.

        The transformation is reversed by popping the joint column from
        the table, splitting it by the previously found separator and
        then setting all the columns back to the table with the original
        names.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        table_data = table_data.copy()
        columns = table_data.pop(self._joint_column).str.split(self._separator)
        for index, column in enumerate(self._columns):
            table_data[column] = columns.str[index]

        return table_data


class GreaterThan(Constraint):
    """Ensure that the ``high`` column is always greater than the ``low`` one.

    The transformation strategy works by replacing the ``high`` value with the
    difference between it and the ``low`` value and then computing back the ``high``
    value by adding it the ``low`` value when reversing the transformation.

    Args:
        low (str):
            Name of the column that contains the low value.
        high (str):
            Name of the column that contains the high value.
        strict (bool):
            Whether the comparison of the values should be strict ``>=`` or
            not ``>`` when comparing them. Currently, this is only respected
            if ``reject_sampling`` or ``all`` handling strategies are used.
        handling_strategy (str):
            How this Constraint should be handled, which can be ``transform``
            or ``reject_sampling``. Defaults to ``transform``.
    """

    def __init__(self, low, high, strict=False, handling_strategy='transform'):
        self._low = low
        self._high = high
        self._strict = strict
        super().__init__(handling_strategy)

    def fit(self, table_data):
        """Learn the dtype of the high column.

        Args:
            table_data (pandas.DataFrame):
                The Table data.
        """
        self._dtype = table_data[self._high].dtype

    def is_valid(self, table_data):
        """Say whether ``high`` is greater than ``low`` in each row.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.Series:
                Whether each row is valid.
        """
        if self._strict:
            return table_data[self._high] > table_data[self._low]

        return table_data[self._high] >= table_data[self._low]

    def transform(self, table_data):
        """Transform the table data.

        The transformation consist on replacing the ``high`` value with difference
        between it and the ``low`` value.

        Afterwards, a logarithm is applied to the difference + 1 to be able to ensure
        that the value stays positive when reverted afterwards using an exponential.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        table_data = table_data.copy()
        low_column = table_data[self._low]
        diff = table_data[self._high] - low_column

        if pd.api.types.is_datetime64_ns_dtype(low_column):
            diff = pd.to_numeric(diff)

        table_data[self._high] = np.log(diff + 1)

        return table_data

    def reverse_transform(self, table_data):
        """Reverse transform the table data.

        The transformation is reversed by computing an exponential of the given
        value, converting it to the original dtype, subtracting 1 and finally
        clipping the value to 0 on the low end to ensure the value is positive.

        Finally, the obtained value is added to the ``low`` column to get the final
        ``high`` value.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        table_data = table_data.copy()
        diff = (np.exp(table_data[self._high]).round() - 1).clip(0)
        low_column = table_data[self._low]

        if pd.api.types.is_datetime64_ns_dtype(low_column):
            diff = pd.to_timedelta(diff)

        table_data[self._high] = (low_column + diff).astype(self._dtype)

        return table_data


class ColumnFormula(Constraint):
    """Compute a column based on applying a formula on the others.

    This contraint accepts as input a simple function and a column name.
    During the transformation phase the column is simply dropped.
    During the reverse transformation, the column is re-generated by
    applying the whole table to the given function.

    Args:
        column (str):
            Name of the column to compute applying the formula.
        formula (callable):
            Function to use for the computation.
        handling_strategy (str):
            How this Constraint should be handled, which can be ``transform``
            or ``reject_sampling``. Defaults to ``transform``.
    """

    def __init__(self, column, formula, handling_strategy='transform'):
        self._column = column
        self._formula = import_object(formula)
        super().__init__(handling_strategy)

    def is_valid(self, table_data):
        """Say whether the data fulfills the formula.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.Series:
                Whether each row is valid.
        """
        computed = self._formula(table_data)
        return table_data[self._column] == computed

    def transform(self, table_data):
        """Transform the table data.

        The transformation consist on simply dropping the indicated column from the
        table to prevent it from being modeled.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        table_data = table_data.copy()
        del table_data[self._column]

        return table_data

    def reverse_transform(self, table_data):
        """Reverse transform the table data.

        The transformation is reversed by applying the given formula function
        to the complete table and storing the result in the indicated column.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        table_data = table_data.copy()
        table_data[self._column] = self._formula(table_data)

        return table_data
