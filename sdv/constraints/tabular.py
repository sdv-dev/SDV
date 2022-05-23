"""Table constraints.

This module contains constraints that are evaluated within a single table,
and which can affect one or more columns at a time, as well as one or more
rows.

Currently implemented constraints are:

    * CustomConstraint: Simple constraint to be set up by passing the python
      functions that will be used for transformation, reverse transformation
      and validation.
    * FixedCombinations: Ensure that the combinations of values
      across several columns are the same after sampling.
    * GreaterThan: Ensure that the value in one column is always greater than
      the value in another column.
    * Positive: Ensure that the values in given columns are always positive.
    * Negative: Ensure that the values in given columns are always negative.
    * ColumnFormula: Compute the value of a column based on applying a formula
      on the other columns of the table.
    * Between: Ensure that the value in one column is always between the values
      of two other columns/scalars.
    * OneHotEncoding: Ensure the rows of the specified columns are one hot encoded.
"""

import operator
import uuid

import numpy as np
import pandas as pd

from sdv.constraints.base import Constraint, import_object
from sdv.constraints.errors import MissingConstraintColumnError
from sdv.constraints.utils import is_datetime_type


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

    def _run(self, function, table_data, reverse=False):
        table_data = table_data.copy()
        if self._columns:
            if reverse:
                columns = reversed(self._columns)
            else:
                columns = self._columns

            for column in columns:
                try:
                    table_data = function(table_data, column)
                except TypeError:
                    table_data[column] = function(table_data[column])

        else:
            table_data = function(table_data)

        return table_data

    def _run_transform(self, table_data):
        if self._columns:
            if any(column not in table_data.columns for column in self._columns):
                raise MissingConstraintColumnError()

        return self._run(self._transform, table_data)

    def _run_reverse_transform(self, table_data):
        return self._run(self._reverse_transform, table_data, reverse=True)

    def _run_is_valid(self, table_data):
        if self._columns:
            try:
                valid = [self._is_valid(table_data, column) for column in self._columns]
            except TypeError:
                valid = [self._is_valid(table_data[column]) for column in self._columns]

            return np.logical_and.reduce(valid)

        return self._is_valid(table_data)

    def __init__(self, columns=None, transform=None, reverse_transform=None, is_valid=None):
        if isinstance(columns, str):
            self._columns = [columns]
        else:
            self._columns = columns

        self.fit_columns_model = False
        if transform is not None:
            self._transform = import_object(transform)
            self.transform = self._run_transform

        if reverse_transform is not None:
            self._reverse_transform = import_object(reverse_transform)
            self.reverse_transform = self._run_reverse_transform

        if is_valid is not None:
            self._is_valid = import_object(is_valid)
            self.is_valid = self._run_is_valid


class FixedCombinations(Constraint):
    """Ensure that the combinations across multiple colums are fixed.

    One simple example of this constraint can be found in a table that
    contains the columns `country` and `city`, where each country can
    have multiple cities and the same city name can even be found in
    multiple countries, but some combinations of country/city would
    produce invalid results.

    This constraint would ensure that the combinations of country/city
    found in the sampled data always stay within the combinations previously
    seen during training.

    Args:
        column_names (list[str]):
            Names of the columns that need to produce fixed combinations. Must
            contain at least two columns.
        handling_strategy (str):
            How this Constraint should be handled, which can be ``transform``,
            ``reject_sampling`` or ``all``. Defaults to ``transform``.
        fit_columns_model (bool):
            If False, reject sampling will be used to handle conditional sampling.
            Otherwise, a model will be trained and used to sample other columns
            based on the conditioned column. Defaults to False.
    """

    _separator = None
    _joint_column = None
    _combinations_to_uuids = None
    _uuids_to_combinations = None

    def __init__(self, column_names, handling_strategy='transform', fit_columns_model=False):
        if len(column_names) < 2:
            raise ValueError('FixedCombinations requires at least two constraint columns.')

        self._columns = column_names
        self.constraint_columns = tuple(column_names)
        self.rebuild_columns = tuple(column_names)
        super().__init__(handling_strategy=handling_strategy,
                         fit_columns_model=fit_columns_model)

    def _fit(self, table_data):
        """Fit this Constraint to the data.

        The fit process consists on:

            - Finding a separator that works for the
              current data by iteratively adding `#` to it.
            - Generating the joint column name by concatenating
              the names of ``self._columns`` with the separator.
            - Generating a mapping of the fixed combinations
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
        low (str or list[str]):
            Either the name of the column(s) that contains the low value,
            or a scalar that is the low value.
        high (str or list[str]):
            Either the name of the column(s) that contains the high value,
            or a scalar that is the high value.
        strict (bool):
            Whether the comparison of the values should be strict ``>=`` or
            not ``>`` when comparing them. Currently, this is only respected
            if ``reject_sampling`` or ``all`` handling strategies are used.
        handling_strategy (str):
            How this Constraint should be handled, which can be ``transform``
            or ``reject_sampling``. Defaults to ``transform``.
        fit_columns_model (bool):
            If False, reject sampling will be used to handle conditional sampling.
            Otherwise, a model will be trained and used to sample other columns
            based on the conditioned column. Defaults to False.
        drop (str):
            Which column to drop during transformation. Can be ``'high'``,
            ``'low'`` or ``None``.
        scalar (str):
            Which value is a scalar. Can be ``'high'``, ``'low'`` or ``None``.
            If ``None`` then both ``high`` and ``low`` are column names.
    """

    _diff_columns = None
    _is_datetime = None
    _columns_to_reconstruct = None

    @staticmethod
    def _as_list(value):
        if not isinstance(value, list):
            return [value]

        return value

    @staticmethod
    def _validate_scalar(scalar_column, column_names, scalar):
        """Validate scalar comparison inputs.

        - Make sure that the scalar column is not a list and raise the proper error if it is.
        - If the `column_names` is not a list it would make it a list.
        - Return both the scalar column and column names with the right format
        """
        if isinstance(scalar_column, list):
            raise TypeError(f'`{scalar}` cannot be a list when scalar="{scalar}".')

        column_names = GreaterThan._as_list(column_names)

        return column_names

    @staticmethod
    def _validate_drop(scalar, drop):
        if drop == scalar:
            raise ValueError(f"Invalid `drop` value: f`{drop}`. Cannot drop a scalar.")

    @classmethod
    def _validate_inputs(cls, low, high, scalar, drop):
        if scalar is None:
            low = cls._as_list(low)
            high = cls._as_list(high)
            if len(low) > 1 and len(high) > 1:
                raise ValueError('either `high` or `low` must contain only one column.')

            constraint_columns = tuple(low + high)

        elif scalar == 'low':
            cls._validate_drop(scalar, drop)
            high = cls._validate_scalar(scalar_column=low, column_names=high, scalar=scalar)
            constraint_columns = tuple(high)
            if isinstance(low, pd.Timestamp):
                low = low.to_datetime64()

        elif scalar == 'high':
            cls._validate_drop(scalar, drop)
            low = cls._validate_scalar(scalar_column=high, column_names=low, scalar=scalar)
            constraint_columns = tuple(low)
            if isinstance(high, pd.Timestamp):
                high = high.to_datetime64()

        else:
            raise ValueError(f"Invalad `scalar` value: `{scalar}`. "
                             "Use either: 'high', 'low', or None.")

        return low, high, constraint_columns

    def _get_columns_to_reconstruct(self):
        if self._drop == 'high':
            column = self._high
        elif self._drop == 'low':
            column = self._low
        elif self._scalar == 'high':
            column = self._low
        else:
            column = self._high

        return column

    def __init__(self, low, high, strict=False, handling_strategy='transform',
                 fit_columns_model=False, drop=None, scalar=None):
        self._strict = strict
        self._drop = drop
        self._scalar = scalar
        self._low, self._high, self.constraint_columns = self._validate_inputs(
            low=low, high=high, scalar=scalar, drop=drop)
        self._columns_to_reconstruct = self._get_columns_to_reconstruct()
        self.rebuild_columns = self._columns_to_reconstruct.copy()

        if strict:
            self.operator = np.greater
        else:
            self.operator = np.greater_equal

        super().__init__(handling_strategy=handling_strategy,
                         fit_columns_model=fit_columns_model)

    def _get_value(self, table_data, field):
        variable = getattr(self, f'_{field}')
        if self._scalar == field:
            return variable

        return table_data[variable].values

    def _get_diff_columns_name(self, table_data):
        names = []
        base = ''
        column_names = list(self.constraint_columns)
        if self._scalar is None:
            base = self._low if len(self._low) == 1 else self._high
            column_names.remove(base[0])
            base = str(base[0])

        for column in list(map(str, column_names)):
            token = '#'
            name = token.join((column, base))
            while name in table_data.columns:
                token += '#'

            names.append(name)

        return names

    def _get_is_datetime(self, table_data):
        low = self._get_value(table_data, 'low')
        high = self._get_value(table_data, 'high')

        is_low_datetime = is_datetime_type(low)
        is_high_datetime = is_datetime_type(high)
        is_datetime = is_low_datetime and is_high_datetime

        if not is_datetime and any([is_low_datetime, is_high_datetime]):
            raise ValueError('Both high and low must be datetime.')

        return is_datetime

    def _check_columns_exist(self, table_data, field):
        values = getattr(self, f'_{field}')
        missing = set(values) - set(table_data.columns)
        if missing:
            raise KeyError(f'The `{field}` columns {missing} '
                           f'were not found in table_data. If `{field}` is a scalar, '
                           f'set `scalar="{field}"`.')

    def _fit(self, table_data):
        """Learn the dtype of the high column.

        Args:
            table_data (pandas.DataFrame):
                The Table data.
        """
        if self._scalar != 'high':
            self._check_columns_exist(table_data, 'high')
        if self._scalar != 'low':
            self._check_columns_exist(table_data, 'low')

        self._dtype = table_data[self._columns_to_reconstruct].dtypes
        self._diff_columns = self._get_diff_columns_name(table_data)
        self._is_datetime = self._get_is_datetime(table_data)

    def is_valid(self, table_data):
        """Say whether ``high`` is greater than ``low`` in each row.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.Series:
                Whether each row is valid.
        """
        low = self._get_value(table_data, 'low')
        high = self._get_value(table_data, 'high')
        isnull = np.logical_or(np.isnan(low), np.isnan(high))

        valid = np.logical_or(self.operator(high, low), isnull)

        return valid.all(axis=1)

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
        diff = self._get_value(table_data, 'high') - self._get_value(table_data, 'low')

        if self._is_datetime:
            diff = diff.astype(np.float64)

        table_data[self._diff_columns] = np.log(diff + 1)
        if self._drop == 'high':
            table_data = table_data.drop(self._high, axis=1)
        elif self._drop == 'low':
            table_data = table_data.drop(self._low, axis=1)

        return table_data

    def _construct_columns(self, diff, column_values, columns):
        new_values = pd.DataFrame(diff + column_values, columns=columns)
        return new_values.astype(dict(zip(columns, self._dtype)))

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
        diff = (np.exp(table_data[self._diff_columns].values).round() - 1).clip(0)
        if self._is_datetime:
            diff = diff.astype('timedelta64[ns]')

        if self._drop == 'high':
            low = self._get_value(table_data, 'low')
            table_data[self._high] = self._construct_columns(diff, low, self._high)
        elif self._drop == 'low':
            high = self._get_value(table_data, 'high')
            table_data[self._low] = self._construct_columns(-diff, high, self._low)
        else:
            low = self._get_value(table_data, 'low')
            high = self._get_value(table_data, 'high')
            invalid = ~self.is_valid(table_data)
            if self._scalar == 'high':
                new_values = high - diff[invalid]
            elif self._scalar == 'low':
                new_values = low + diff[invalid]
            else:
                new_values = low[invalid] + diff[invalid]

            for i, column in enumerate(self._columns_to_reconstruct):
                table_data.loc[invalid, column] = new_values[:, i].astype(self._dtype[i])

        table_data = table_data.drop(self._diff_columns, axis=1)

        return table_data


class Positive(GreaterThan):
    """Ensure that the given column(s) are always positive.

    The transformation strategy works by creating columns with the
    difference between given columns and zero then computing back the
    necessary columns using the difference.

    Args:
        columns (str or list[str]):
            The name of the column(s) that are constrained to be positive.
        strict (bool):
            Whether the comparison of the values should be strict; disclude
            zero ``>`` or include it ``>=``. Currently, this is only respected
            if ``reject_sampling`` or ``all`` handling strategies are used.
        handling_strategy (str):
            How this Constraint should be handled, which can be ``transform``
            or ``reject_sampling``. Defaults to ``transform``.
        fit_columns_model (bool):
            If False, reject sampling will be used to handle conditional sampling.
            Otherwise, a model will be trained and used to sample other columns
            based on the conditioned column. Defaults to False.
        drop (bool):
            Whether to drop columns during transformation.
    """

    def __init__(self, columns, strict=False, handling_strategy='transform',
                 fit_columns_model=False, drop=False):
        drop = 'high' if drop else None
        super().__init__(handling_strategy=handling_strategy,
                         fit_columns_model=fit_columns_model,
                         high=columns, low=0, scalar='low',
                         drop=drop, strict=strict)


class Negative(GreaterThan):
    """Ensure that the given columns are always negative.

    The transformation strategy works by creating columns with the
    difference between zero and given columns then computing back the
    necessary columns using the difference.

    Args:
        columns (str or list[str]):
            The name of the column(s) that are constrained to be negative.
        strict (bool):
            Whether the comparison of the values should be strict, disclude
            zero ``<`` or include it ``<=``. Currently, this is only respected
            if ``reject_sampling`` or ``all`` handling strategies are used.
        handling_strategy (str):
            How this Constraint should be handled, which can be ``transform``
            or ``reject_sampling``. Defaults to ``transform``.
        fit_columns_model (bool):
            If False, reject sampling will be used to handle conditional sampling.
            Otherwise, a model will be trained and used to sample other columns
            based on the conditioned column. Defaults to False.
        drop (bool):
            Whether to drop columns during transformation.
    """

    def __init__(self, columns, strict=False, handling_strategy='transform',
                 fit_columns_model=False, drop=False):
        drop = 'low' if drop else None
        super().__init__(handling_strategy=handling_strategy,
                         fit_columns_model=fit_columns_model,
                         high=0, low=columns, scalar='high',
                         drop=drop, strict=strict)


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
        drop_column (bool):
            Whether or not to drop the constraint column.
    """

    def __init__(self, column, formula, handling_strategy='transform', drop_column=True):
        self._column = column
        self.constraint_columns = (column,)
        self._formula = import_object(formula)
        self._drop_column = drop_column
        self.rebuild_columns = (column,)
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
        isnan = table_data[self._column].isna() & computed.isna()

        return table_data[self._column].eq(computed) | isnan

    def _transform(self, table_data):
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

        if self._drop_column and self._column in table_data:
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
        fit_columns_model (bool):
            If False, reject sampling will be used to handle conditional sampling.
            Otherwise, a model will be trained and used to sample other columns
            based on the conditioned column. Defaults to False.
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
                 fit_columns_model=False, high_is_scalar=None, low_is_scalar=None):
        self.constraint_column = column
        self.constraint_columns = (column,)
        self.rebuild_columns = (column,)
        self._strict = strict
        self._high_is_scalar = high_is_scalar
        self._low_is_scalar = low_is_scalar
        self._lt = operator.lt if strict else operator.le

        self._low = low
        if self._low_is_scalar and isinstance(low, pd.Timestamp):
            self._low = low.to_datetime64()

        self._high = high
        if self._high_is_scalar and isinstance(high, pd.Timestamp):
            self._high = high.to_datetime64()

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

    def _get_is_datetime(self, table_data):
        low = self._get_low_value(table_data)
        high = self._get_high_value(table_data)
        column = table_data[self.constraint_column]

        is_low_datetime = is_datetime_type(low)
        is_high_datetime = is_datetime_type(high)
        is_column_datetime = is_datetime_type(column)
        is_datetime = is_low_datetime and is_high_datetime and is_column_datetime

        if not is_datetime and any([is_low_datetime, is_high_datetime, is_column_datetime]):
            raise ValueError('The constraint column and bounds must all be datetime.')

        return is_datetime

    def _fit(self, table_data):
        if self._high_is_scalar is None:
            self._high_is_scalar = self._high not in table_data.columns
        if self._low_is_scalar is None:
            self._low_is_scalar = self._low not in table_data.columns

        self._transformed_column = self._get_diff_column_name(table_data)
        self._is_datetime = self._get_is_datetime(table_data)

    def is_valid(self, table_data):
        """Say whether the ``constraint_column`` is between the ``low`` and ``high`` values.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.Series:
                Whether each row is valid.
        """
        low = self._get_low_value(table_data)
        high = self._get_high_value(table_data)

        satisfy_low_bound = np.logical_or(
            self._lt(low, table_data[self.constraint_column]),
            np.isnan(low),
        )
        satisfy_high_bound = np.logical_or(
            self._lt(table_data[self.constraint_column], high),
            np.isnan(high),
        )

        return np.logical_or(
            np.logical_and(satisfy_low_bound, satisfy_high_bound),
            np.isnan(table_data[self.constraint_column]),
        )

    def _transform(self, table_data):
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

        if self._is_datetime:
            table_data[self.constraint_column] = pd.to_datetime(data)
        else:
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
        conditions_data = table_data[condition_columns]
        conditions_data_sum = conditions_data.sum(axis=1)
        if not conditions_data.isin([0.0, 1.0]).all(axis=1).all():
            raise ValueError('Condition values must be ones or zeros.')

        if (conditions_data_sum > 1.0).any():
            raise ValueError('Each row of a condition can only contain one number one.')

        has_one = conditions_data_sum == 1.0
        if (~has_one).sum() > 0:
            sub_table_data = table_data.loc[~has_one, condition_columns]
            should_transform = False

            if len(condition_columns) == len(self._columns) - 1:
                proposed_table_data = sub_table_data.copy()
                for column in self._columns:
                    if column not in condition_columns:
                        proposed_table_data[column] = 1.0

            else:
                should_transform = True
                conditions = sub_table_data[condition_columns]
                transformed_conditions = self._hyper_transformer.transform(conditions)
                proposed_table_data = self._columns_model.sample(
                    num_rows=len(sub_table_data),
                    conditions=transformed_conditions.iloc[0].to_dict()
                )

            if should_transform:
                proposed_table_data = self._hyper_transformer.reverse_transform(
                    proposed_table_data)

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
        no_nans = ~one_hot_data.isna().any(axis=1)

        return sum_one & max_one & min_zero & no_nans

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


class Unique(Constraint):
    """Ensure that each value for a specified column/group of columns is unique.

    This constraint is provided a list of columns, and guarantees that every
    unique combination of those columns appears at most once in the sampled
    data.

    Args:
        columns (str or list[str]):
            Name of the column(s) to keep unique.
    """

    def __init__(self, columns):
        self.columns = columns if isinstance(columns, list) else [columns]
        super().__init__(handling_strategy='reject_sampling', fit_columns_model=False)

    def is_valid(self, table_data):
        """Get indices of first instance of unique rows.

        If a row is the first instance of that combination of column
        values, it is valid. Otherwise it is false.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.Series:
                Whether each row is valid.
        """
        return table_data.groupby(self.columns, dropna=False).cumcount() == 0
