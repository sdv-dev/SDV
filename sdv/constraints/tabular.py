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
    * Between: Ensure that the value in one column is always between the values
      of two other columns/scalars.
    * OneHotEncoding: Ensure the rows of the specified columns are one hot encoded.
"""

import operator
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


class Between(Constraint):
    """Ensure that the ``constraint_column`` is always between ``high`` and ``low``.

    The transformation strategy works by replacing the ``constraint_column`` with a
    scaled version and then applying a logit function. The reverse transform
    applies a sigmoid to the data and then scales it back to the original space.

    Args:
        constraint_column (str):
            Name of the column to which the constraint will be applied.
        low (float or str):
            If float, lower bound on the values of the ``constraint_column``.
            If string, name of the column which will be the lower bound.
        high (float or str):
            If float, upper bound on the values of the ``constraint_column``.
            If string, name of the column which will be the upper bound.
        strict (bool):
            Whether the comparison of the values should be strict ``>=`` or
            not ``>`` when comparing them. Currently, this is only respected
            if ``reject_sampling`` or ``all`` handling strategies are used.
        handling_strategy (str):
            How this Constraint should be handled, which can be ``transform``
            or ``reject_sampling``. Defaults to ``transform``.
        high_is_scalar(bool or None):
            Whether or not the value for high is a scalar or a column name.
            If ``None``, this will be determined during the ``fit`` method
            by checking if the value provided is a column name.
        low_is_scalar(bool or None):
            Whether or not the value for low is a scalar or a column name.
            If ``None``, this will be determined during the ``fit`` method
            by checking if the value provided is a column name.
    """

    _transformed_column = None

    def __init__(self, column, low, high, strict=False, handling_strategy='transform',
                 fit_columns_model=True, high_is_scalar=None, low_is_scalar=None):
        self.constraint_column = column
        self._low = low
        self._high = high
        self._strict = strict
        self._high_is_scalar = high_is_scalar
        self._low_is_scalar = low_is_scalar
        self._lt = operator.lt if strict else operator.le
        super().__init__(handling_strategy=handling_strategy,
                         fit_columns_model=fit_columns_model)

    def _get_low_value(self, table_data):
        """Return the appropriate lower bound.

        Returns the lower bound either as a column or a scalar, depending on the
        value of ``self._low_is_scalar``. If the lower bound column doesn't exist, returns
        ``None`` instead.

        Args:
            table_data (pandas.DataFrame):
                The Table data.

        Returns:
            pandas.DataFrame, float or None:
                Either the lower bound or None if the column doesn't exist.
        """
        if self._low_is_scalar:
            return self._low
        elif self._low in table_data.columns:
            return table_data[self._low]

        return None

    def _get_high_value(self, table_data):
        """Return the appropriate upper bound.

        Returns the upper bound either as a column or a scalar, depending on the
        value of ``self._high_is_scalar``. If the upper bound column doesn't exist, returns
        ``None`` instead.

        Args:
            table_data (pandas.DataFrame):
                The Table data.

        Returns:
            pandas.DataFrame, float or None:
                Either the upper bound or None if the column doesn't exist.
        """
        if self._high_is_scalar:
            return self._high
        elif self._high in table_data.columns:
            return table_data[self._high]

        return None

    def _get_diff_column_name(self, table_data):
        token = '#'
        components = list(map(str, [self.constraint_column, self._low, self._high]))
        while token.join(components) in table_data.columns:
            token += '#'

        return token.join(components)

    def _fit(self, table_data):
        if self._high_is_scalar is None:
            self._high_is_scalar = self._high not in table_data.columns
        if self._low_is_scalar is None:
            self._low_is_scalar = self._low not in table_data.columns

        self._transformed_column = self._get_diff_column_name(table_data)

    def is_valid(self, table_data):
        """Say whether the ``constraint_column`` is between the ``low`` and ``high`` values.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.Series:
                Whether each row is valid.
        """
        satisfy_low_bound = self._lt(
            self._get_low_value(table_data), table_data[self.constraint_column]
        )
        satisfy_high_bound = self._lt(
            table_data[self.constraint_column], self._get_high_value(table_data)
        )

        return satisfy_low_bound & satisfy_high_bound

    def transform(self, table_data):
        """Transform the table data.

        The transformation consists of scaling the ``constraint_column``
        (``(column-low)/(high-low) * cnt + small_cnt``) and then applying
        a logit function to the scaled version of the column.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        table_data = table_data.copy()
        low = self._get_low_value(table_data)
        high = self._get_high_value(table_data)

        data = (table_data[self.constraint_column] - low) / (high - low)
        data = data * 0.95 + 0.025
        data = np.log(data / (1.0 - data))

        table_data[self._transformed_column] = data
        table_data = table_data.drop(self.constraint_column, axis=1)

        return table_data

    def reverse_transform(self, table_data):
        """Reverse transform the table data.

        The reverse transform consists of applying a sigmoid to the transformed
        ``constraint_column`` and then scaling it back to the original space
        ( ``(column - cnt) * (high - low) / cnt + low`` ).

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        table_data = table_data.copy()
        low = self._get_low_value(table_data)
        high = self._get_high_value(table_data)
        data = table_data[self._transformed_column]

        data = 1 / (1 + np.exp(-data))
        data = (data - 0.025) / 0.95
        data = data * (high - low) + low
        data = data.clip(low, high)

        table_data[self.constraint_column] = data
        table_data = table_data.drop(self._transformed_column, axis=1)

        return table_data


class Rounding(Constraint):
    """Round a column based on the specified number of digits.

    Args:
        columns (str or list[str]):
            Name of the column(s) to round.
        digits (int):
            How much to round each column. All columns will be rounded to this
            number of digits.
        handling_strategy (str):
            How this Constraint should be handled, which can be ``transform``
            or ``reject_sampling``. Defaults to ``transform``.
        tolerance (int):
            When reject sampling, the sample data must be within this distance
            of the desired rounded values.
    """

    def __init__(self, columns, digits, handling_strategy='transform', tolerance=None):
        if digits > 15:
            raise ValueError('The value of digits cannot exceed 15')

        if tolerance is not None and tolerance >= 10**(-1 * digits):
            raise ValueError('Tolerance must be less than the rounding level')

        if isinstance(columns, str):
            self._columns = [columns]
        else:
            self._columns = columns

        self._digits = digits
        self._round_config = {column: self._digits for column in self._columns}
        self._tolerance = tolerance if tolerance else 10**(-1 * (digits + 1))
        super().__init__(handling_strategy=handling_strategy, fit_columns_model=False)

    def is_valid(self, table_data):
        """Determine if the data satisfies the rounding constraint.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.Series:
                Whether each row is valid.
        """
        columns = table_data[self._columns]
        rounded = columns.round(self._digits)
        valid = (columns - rounded).abs() <= self._tolerance

        return valid.all(1)

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
        return table_data.round(self._round_config)


class OneHotEncoding(Constraint):
    """Ensure the appropriate columns are one hot encoded.

    This constraint allows the user to specify a list of columns where each row
    is a one hot vector. During the reverse transform, the output of the model
    is transformed so that the column with the largest value is set to 1 while
    all other columns are set to 0.

    Args:
        columns (list[str]):
            Names of the columns containing one hot rows.
        handling_strategy (str):
            How this Constraint should be handled, which can be ``transform``
            or ``reject_sampling`` (not recommended). Defaults to ``transform``.
    """

    def __init__(self, columns, handling_strategy='transform'):
        self._columns = columns
        self.constraint_columns = tuple(columns)
        super().__init__(handling_strategy, fit_columns_model=True)

    def _sample_constraint_columns(self, table_data):
        """Handle constraint columns when conditioning.

        When handling a set of one-hot columns, a subset of columns may be provided
        to condition on. To handle this, this function does the following:

        1. If the user specifies that a particular column must be 1,
           then all other columns must be 0.
        2. If the user specifies that one or more columns must be 0, then
           we need to sample the other columns and select the highest value
           and enforce the one-hot constraint.
        3. If the user specifies something invalid, we need to raise an error.

        Args:
            table_data (pandas.DataFrame):
                Table data containing the conditions.

        Returns:
            pandas.DataFrame:
                Table data with the constraint columns filled in.

        Raise:
            ``ValueError`` if the conditions are invalid.
        """
        table_data = table_data.copy()

        condition_columns = [col for col in self._columns if col in table_data.columns]
        if not table_data[condition_columns].isin([0.0, 1.0]).all(axis=1).all():
            raise ValueError('Condition values must be ones or zeros.')

        if (table_data[condition_columns].sum(axis=1) > 1.0).any():
            raise ValueError('Each row of a condition can only contain one number one.')

        has_one = table_data[condition_columns].sum(axis=1) == 1.0
        if (~has_one).sum() > 0:
            sub_table_data = table_data.loc[~has_one, condition_columns]

            if len(condition_columns) == len(self._columns) - 1:
                proposed_table_data = sub_table_data.copy()
                for column in self._columns:
                    if column not in condition_columns:
                        proposed_table_data[column] = 1.0

            else:
                proposed_table_data = self._columns_model.sample(
                    num_rows=sub_table_data[condition_columns].shape[0],
                    conditions=sub_table_data[condition_columns].iloc[0].to_dict()
                )

            for column in self._columns:
                if column not in condition_columns:
                    sub_table_data[column] = proposed_table_data[column].values
                else:
                    sub_table_data[column] = float('-inf')

            table_data.loc[~has_one, self._columns] = self.reverse_transform(sub_table_data)

        if has_one.sum() > 0:
            for column in self._columns:
                if column not in condition_columns:
                    table_data.loc[has_one, column] = 0

        return table_data

    def is_valid(self, table_data):
        """Check whether the data satisfies the one-hot constraint.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.Series:
                Whether each row is valid.
        """
        one_hot_data = table_data[self._columns]

        sum_one = one_hot_data.sum(axis=1) == 1.0
        max_one = one_hot_data.max(axis=1) == 1.0
        min_zero = one_hot_data.min(axis=1) == 0.0

        return sum_one & max_one & min_zero

    def reverse_transform(self, table_data):
        """Reverse transform the table data.

        Set the column with the largest value to one, set all other columns to zero.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        table_data = table_data.copy()

        one_hot_data = table_data[self._columns]
        transformed_data = np.zeros_like(one_hot_data.values)
        transformed_data[np.arange(len(one_hot_data)), np.argmax(one_hot_data.values, axis=1)] = 1
        table_data[self._columns] = transformed_data

        return table_data
