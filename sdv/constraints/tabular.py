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

import decimal
import uuid
from datetime import datetime

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
        self.fit_columns_model = False
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
    _combinations_to_uuids = None
    _uuids_to_combinations = None

    def __init__(self, columns, handling_strategy='transform', fit_columns_model=True):
        self._columns = columns
        self.constraint_columns = tuple(columns)
        super().__init__(handling_strategy=handling_strategy,
                         fit_columns_model=fit_columns_model)

    def _fit(self, table_data):
        """Fit this Constraint to the data.

        The fit process consists on:

            - Finding a separator that works for the
              current data by iteratively adding `#` to it.
            - Generating the joint column name by concatenating
              the names of ``self._columns`` with the separator.
            - Generating a mapping of the unique combinations
              to a unique identifier.

        Args:
            table_data (pandas.DataFrame):
                Table data.
        """
        self._separator = '#'
        while self._separator.join(self._columns) in table_data:
            self._separator += '#'

        self._joint_column = self._separator.join(self._columns)
        self._combinations = table_data[self._columns].drop_duplicates().copy()
        self._combinations_to_uuids = {}
        self._uuids_to_combinations = {}
        for combination in self._combinations.itertuples(index=False, name=None):
            uuid_str = str(uuid.uuid4())
            self._combinations_to_uuids[combination] = uuid_str
            self._uuids_to_combinations[uuid_str] = combination

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

    def _transform(self, table_data):
        """Transform the table data.

        The transformation consist on removing all the ``self._columns`` from
        the dataframe, and replacing them with a unique identifier that maps to
        that unique combination of column values under the previously computed
        combined column name.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        table_data = table_data.copy()
        combinations = table_data[self._columns].itertuples(index=False, name=None)
        uuids = map(self._combinations_to_uuids.get, combinations)
        table_data[self._joint_column] = list(uuids)
        return table_data.drop(self._columns, axis=1)

    def reverse_transform(self, table_data):
        """Reverse transform the table data.

        The transformation is reversed by popping the joint column from
        the table, mapping it back to the original combination of column values,
        and then setting all the columns back to the table with the original
        names.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        table_data = table_data.copy()
        columns = table_data.pop(self._joint_column).map(self._uuids_to_combinations)

        for index, column in enumerate(self._columns):
            table_data[column] = columns.str[index]

        return table_data


class GreaterThan(Constraint):
    """Ensure that the ``high`` column is always greater than the ``low`` one.

    The transformation strategy works by creating a column with the
    difference between the ``high`` and ``low`` values and then computing back the
    necessary columns using the difference and whichever other value is available.
    For example, if the ``high`` column is dropped, then the ``low`` column/value
    will be added to the diff to reconstruct the ``high`` column.

    Args:
        low (str or int):
            Either the name of the column that contains the low value,
            or a scalar that is the low value.
        high (str or int):
            Either the name of the column that contains the high value,
            or a scalar that is the high value.
        strict (bool):
            Whether the comparison of the values should be strict ``>=`` or
            not ``>`` when comparing them. Currently, this is only respected
            if ``reject_sampling`` or ``all`` handling strategies are used.
        handling_strategy (str):
            How this Constraint should be handled, which can be ``transform``
            or ``reject_sampling``. Defaults to ``transform``.
        drop (str):
            Which column to drop during transformation. Can be ``'high'``,
            ``'low'`` or ``None``.
        high_is_scalar(bool or None):
            Whether or not the value for high is a scalar or a column name.
            If ``None``, this will be determined during the ``fit`` method
            by checking if the value provided is a column name.
        low_is_scalar(bool or None):
            Whether or not the value for low is a scalar or a column name.
            If ``None``, this will be determined during the ``fit`` method
            by checking if the value provided is a column name.
    """

    _diff_column = None
    _is_datetime = None
    _column_to_reconstruct = None

    def __init__(self, low, high, strict=False, handling_strategy='transform',
                 fit_columns_model=True, drop=None, high_is_scalar=None,
                 low_is_scalar=None):
        self._low = low
        self._high = high
        self._strict = strict
        self.constraint_columns = (low, high)
        self._drop = drop
        self._high_is_scalar = high_is_scalar
        self._low_is_scalar = low_is_scalar
        super().__init__(handling_strategy=handling_strategy,
                         fit_columns_model=fit_columns_model)

    def _get_low_value(self, table_data):
        if self._low_is_scalar:
            return self._low
        elif self._low in table_data.columns:
            return table_data[self._low]

        return None

    def _get_high_value(self, table_data):
        if self._high_is_scalar:
            return self._high
        elif self._high in table_data.columns:
            return table_data[self._high]

        return None

    def _get_column_to_reconstruct(self):
        if self._drop == 'high':
            column = self._high
        elif self._drop == 'low':
            column = self._low
        elif self._high_is_scalar:
            column = self._low
        else:
            column = self._high

        return column

    def _get_diff_column_name(self, table_data):
        token = '#'
        if len(self.constraint_columns) == 1:
            name = self.constraint_columns[0] + token
            while name in table_data.columns:
                name += '#'

            return name

        while token.join(self.constraint_columns) in table_data.columns:
            token += '#'

        return token.join(self.constraint_columns)

    def _fit(self, table_data):
        """Learn the dtype of the high column.

        Args:
            table_data (pandas.DataFrame):
                The Table data.
        """
        if self._high_is_scalar is None:
            self._high_is_scalar = self._high not in table_data.columns
        if self._low_is_scalar is None:
            self._low_is_scalar = self._low not in table_data.columns

        if self._high_is_scalar and self._low_is_scalar:
            raise TypeError('`low` and `high` cannot be both scalars at the same time')
        elif self._low_is_scalar:
            self.constraint_columns = (self._high,)
            self._dtype = table_data[self._high].dtype
        elif self._high_is_scalar:
            self.constraint_columns = (self._low,)
            self._dtype = table_data[self._low].dtype
        else:
            self._dtype = table_data[self._high].dtype

        self._column_to_reconstruct = self._get_column_to_reconstruct()
        self._diff_column = self._get_diff_column_name(table_data)
        low = self._get_low_value(table_data)
        self._is_datetime = (pd.api.types.is_datetime64_ns_dtype(low)
                             or isinstance(low, pd.Timestamp)
                             or isinstance(low, datetime))

    def is_valid(self, table_data):
        """Say whether ``high`` is greater than ``low`` in each row.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.Series:
                Whether each row is valid.
        """
        low = self._get_low_value(table_data)
        high = self._get_high_value(table_data)
        if self._strict:
            return high > low

        return high >= low

    def _transform(self, table_data):
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
        diff = self._get_high_value(table_data) - self._get_low_value(table_data)

        if self._is_datetime:
            diff = pd.to_numeric(diff)

        table_data[self._diff_column] = np.log(diff + 1)
        if self._drop == 'high':
            table_data = table_data.drop(self._high, axis=1)
        elif self._drop == 'low':
            table_data = table_data.drop(self._low, axis=1)

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
        diff = (np.exp(table_data[self._diff_column]).round() - 1).clip(0)
        if self._is_datetime:
            diff = pd.to_timedelta(diff)

        high = self._get_high_value(table_data)
        low = self._get_low_value(table_data)

        if self._drop == 'high':
            table_data[self._high] = (low + diff).astype(self._dtype)
        elif self._drop == 'low':
            table_data[self._low] = (high - diff).astype(self._dtype)
        else:
            invalid = ~self.is_valid(table_data)
            if not self._high_is_scalar and not self._low_is_scalar:
                new_values = low.loc[invalid] + diff.loc[invalid]
            elif self._high_is_scalar:
                new_values = high - diff.loc[invalid]
            else:
                new_values = low + diff.loc[invalid]

            table_data[self._column_to_reconstruct].loc[invalid] = new_values.astype(self._dtype)

        table_data = table_data.drop(self._diff_column, axis=1)

        return table_data


class Positive(GreaterThan):
    """Ensure that the ``high`` column is always positive.

    The transformation strategy works by creating a column with the
    difference between ``high`` and 0 value and then computing back the ``high``
    value by adding the difference to 0 when reversing the transformation.

    Args:
        high (str or int):
            Either the name of the column that contains the high value,
            or a scalar that is the high value.
        strict (bool):
            Whether the comparison of the values should be strict ``>=`` or
            not ``>`` when comparing them. Currently, this is only respected
            if ``reject_sampling`` or ``all`` handling strategies are used.
        handling_strategy (str):
            How this Constraint should be handled, which can be ``transform``
            or ``reject_sampling``. Defaults to ``transform``.
        drop (str):
            Which column to drop during transformation. Can be ``'high'``
            or ``None``.
    """

    def __init__(self, high, strict=False, handling_strategy='transform',
                 fit_columns_model=True, drop=None):
        super().__init__(handling_strategy=handling_strategy,
                         fit_columns_model=fit_columns_model,
                         high=high, low=0, high_is_scalar=False,
                         low_is_scalar=True, drop=drop, strict=strict)


class Negative(GreaterThan):
    """Ensure that the ``low`` column is always negative.

    The transformation strategy works by creating a column with the
    difference between ``low`` and 0 and then computing back the ``low``
    value by subtracting the difference from 0 when reversing the transformation.

    Args:
        high (str or int):
            Either the name of the column that contains the high value,
            or a scalar that is the high value.
        strict (bool):
            Whether the comparison of the values should be strict ``>=`` or
            not ``>`` when comparing them. Currently, this is only respected
            if ``reject_sampling`` or ``all`` handling strategies are used.
        handling_strategy (str):
            How this Constraint should be handled, which can be ``transform``
            or ``reject_sampling``. Defaults to ``transform``.
        drop (str):
            Which column to drop during transformation. Can be ``'low'``
            or ``None``.
    """

    def __init__(self, low, strict=False, handling_strategy='transform',
                 fit_columns_model=True, drop=None):
        super().__init__(handling_strategy=handling_strategy,
                         fit_columns_model=fit_columns_model,
                         high=0, low=low, high_is_scalar=True,
                         low_is_scalar=False, drop=drop, strict=strict)


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
        super().__init__(handling_strategy, fit_columns_model=False)

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


class Rounding(Constraint):
    """Round a column based on the specified number of digits.

    Args:
        column (list[str]):
            Name of the column(s) to round.
        digits (int or dict[str->int]):
            How much to round each column. If an `int` is provided, all columns
            will be rounded to that number of digits. If a `dict` that maps column
            to digits is provided, each column will be represented to the specified
            number of digits.
        handling_strategy (str):
            How this Constraint should be handled, which can be ``transform``
            or ``reject_sampling``. Defaults to ``transform``.
        tolerance (int):
            How many differences in decimal places we will tolerante when
            reject sampling.
    """

    def __init__(self, columns, digits, handling_strategy='transform', tolerance=1):
        self._columns = columns
        self._digits = digits
        self._tolerance = tolerance
        super().__init__(handling_strategy=handling_strategy, fit_columns_model=False)

    def is_valid(self, table_data):
        """Determine if the data satisfies the rounding constranit.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.Series:
                Whether each row is valid.
        """
        table_data = table_data.copy()
        d_places = table_data[self._columns].applymap(
            lambda d: -1 * decimal.Decimal(str(d)).as_tuple().exponent)

        cols = [self._columns] if isinstance(self._digits, int) else [col for col in self._columns]
        for col in cols:
            digits = self._digits if isinstance(self._digits, int) else self._digits[col]
            low = max(digits - self._tolerance, 0)
            high = digits + self._tolerance
            table_data[col] = table_data[(d_places[col] >= low) & (d_places[col] <= high)][col]

        return table_data.notna().all(1)

    def reverse_transform(self, table_data):
        """Reverse transform the table data.

        Round the columns to the desired digits.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        if isinstance(self._digits, dict):
            digit_map = self._digits
        else:
            digit_map = {col: self._digits for col in self._columns}

        return table_data.round(digit_map)
