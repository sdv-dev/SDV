"""Constraints based on data Transformations."""

import numpy as np
import pandas as pd

from sdv.constraints.base import Constraint


class TransformationConstraint(Constraint):
    """Base Transformation class.

    This class can be subclassed in order to develop other
    generic transformations or used directly by passing the
    transform and reverse_transform functions as arguments.

    Args:
        tranform (callable):
            Function to replace the ``tranform`` method.
        reverse_tranform (callable):
            Function to replace the ``reverse_tranform`` method.
    """

    def __init__(self, transform=None, reverse_transform=None):
        if transform is not None:
            self.transform = transform

        if reverse_transform is not None:
            self.reverse_transform = reverse_transform


class UniqueCombinationsConstraint(TransformationConstraint):
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
    """

    _separator = None
    _joint_column = None

    def __init__(self, columns):
        self._columns = columns

    def _valid_separator(self, data):
        """Return True if separator is valid for this data.

        If the separator is contained within any of the columns
        or the column name obtained after joining the column
        names using the separator already exists, the separator
        is not valid.

        Args:
            data (pandas.DataFrame):
                Table data.

        Returns:
            bool:
                Whether the separator is valid for this data or not.
        """
        for column in self._columns:
            if data[column].str.contains(self._separator).any():
                return False

            if self._separator.join(self._columns) in data:
                return False

        return True

    def fit(self, data):
        """Fit this Constraint to the data.

        The fit process consists on:
            - finding a separtor that works for the
              current data by iteratively adding `#` to it.
            - Generating the joint column name by concatenating
              the names of ``self._columns`` with the separator.

        Args:
            data (pandas.DataFrame):
                Table data.
        """
        self._separator = '#'
        while not self._valid_separator(data):
            self._separator += '#'

        self._joint_column = self._separator.join(self._columns)

    def transform(self, data):
        """Transform the table data.

        The transformation consist on removing all the ``self._columns`` from
        the dataframe, concatenating them using the found separator, and
        setting them back to the data as a single name with the previously
        computed name.

        Args:
            data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        lists_series = pd.Series(data[self._columns].values.tolist())
        data = data.drop(self._columns, axis=1)
        data[self._joint_column] = lists_series.str.join(self._separator)

        return data

    def reverse_transform(self, data):
        """Reverse transform the table data.

        The transformation is reversed by popping the joint column from
        the table, splitting it by the previously found separator and
        them setting all the columns back to the table with the original
        names.

        Args:
            data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        data = data.copy()
        columns = data.pop(self._joint_column).str.split(self._separator)
        for index, column in enumerate(self._columns):
            data[column] = columns.str[index]

        return data


class GreaterThanConstraint(TransformationConstraint):

    def __init__(self, low, high):
        self._low = low
        self._high = high

    def transform(self, data):
        data = data.copy()
        data[self._high] = np.log(data.pop(self._high) - data[self._low] + 1)

        return data

    def reverse_transform(self, data):
        data = data.copy()
        diff = (np.exp(data[self._high]).round().astype(int) - 1).clip(0)
        data[self._high] = data[self._low] + diff

        return data


class ColumnFormulaConstraint(TransformationConstraint):
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
    """

    def __init__(self, column, formula):
        self._column = column
        self._formula = formula

    def transform(self, data):
        data = data.copy()
        del data[self._column]

        return data

    def reverse_transform(self, data):
        data = data.copy()
        data[self._column] = self._formula(data)

        return data
