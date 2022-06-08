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
    * Inequality: Ensure that the value in one column is always greater than
      the value in another column.
    * ScalarInequality: Ensure that the value in one column is always greater/smaller
      than some scalar.
    * Positive: Ensure that the values in given columns are always positive.
    * Negative: Ensure that the values in given columns are always negative.
    * ColumnFormula: Compute the value of a column based on applying a formula
      on the other columns of the table.
    * Range: Ensure that the value in one column is always between the values
      of two other columns.
    * ScalarRange: Ensure that the value in one column is always between the values
      of two other scalars.
    * OneHotEncoding: Ensure the rows of the specified columns are one hot encoded.
    * Unique: Ensure that each value for a specified column/group of columns is unique.
    * FixedIncrements: Ensure that every value is a multiple of a specified increment.
"""

import operator
import uuid

import numpy as np
import pandas as pd

from sdv.constraints.base import Constraint, import_object
from sdv.constraints.errors import MissingConstraintColumnError
from sdv.constraints.utils import is_datetime_type, logit, sigmoid

INEQUALITY_TO_OPERATION = {
    '>': np.greater,
    '>=': np.greater_equal,
    '<': np.less,
    '<=': np.less_equal}


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


class Inequality(Constraint):
    """Ensure that the ``high_column_name`` column is greater than the ``low_column_name`` one.

    The transformation works by creating a column with the difference between the
    ``high_column_name`` and ``low_column_name`` columns and storing it in the
    ``high_column_name``'s place. The reverse transform adds the difference column
    and the ``low_column_name`` to reconstruct the ``high_column_name``.

    Args:
        low_column_name (str):
            Name of the column that contains the low values.
        high_column_name (str):
            Name of the column that contains the high values.
        strict_boundaries (bool):
            Whether the comparison of the values should be strict ``>=`` or
            not ``>``. Currently, this is only respected if ``reject_sampling``
            or ``all`` handling strategies are used.
            Defaults to False.
    """

    @staticmethod
    def _validate_inputs(low_column_name, high_column_name, strict_boundaries):
        if not (isinstance(low_column_name, str) and isinstance(high_column_name, str)):
            raise ValueError('`low_column_name` and `high_column_name` must be strings.')

        if not isinstance(strict_boundaries, bool):
            raise ValueError('`strict_boundaries` must be a boolean.')

    def __init__(self, low_column_name, high_column_name, strict_boundaries=False,
                 handling_strategy='transform', fit_columns_model=False):
        self._validate_inputs(low_column_name, high_column_name, strict_boundaries)
        self._low_column_name = low_column_name
        self._high_column_name = high_column_name
        self._diff_column_name = f'{self._low_column_name}#{self._high_column_name}'
        self._operator = np.greater if strict_boundaries else np.greater_equal
        self.rebuild_columns = tuple(high_column_name)
        self.constraint_columns = tuple([low_column_name, high_column_name])
        self._dtype = None
        self._is_datetime = None
        super().__init__(handling_strategy=handling_strategy, fit_columns_model=fit_columns_model)

    def _get_data(self, table_data):
        low = table_data[self._low_column_name].to_numpy()
        high = table_data[self._high_column_name].to_numpy()
        return low, high

    def _get_is_datetime(self, table_data):
        low, high = self._get_data(table_data)
        is_low_datetime = is_datetime_type(low)
        is_high_datetime = is_datetime_type(high)
        is_datetime = is_low_datetime and is_high_datetime

        if not is_datetime and any([is_low_datetime, is_high_datetime]):
            raise ValueError('Both high and low must be datetime.')

        return is_datetime

    def _validate_columns_exist(self, table_data):
        missing = set([self._low_column_name, self._high_column_name]) - set(table_data.columns)
        if missing:
            raise KeyError(f'The columns {missing} were not found in table_data.')

    def _fit(self, table_data):
        """Learn the ``dtype`` of ``_high_column_name`` and whether the data is datetime.

        Args:
            table_data (pandas.DataFrame):
                The Table data.
        """
        self._validate_columns_exist(table_data)
        self._is_datetime = self._get_is_datetime(table_data)
        self._dtype = table_data[self._high_column_name].dtypes

    def is_valid(self, table_data):
        """Check whether ``high`` is greater than ``low`` in each row.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.Series:
                Whether each row is valid.
        """
        low, high = self._get_data(table_data)
        valid = np.isnan(low) | np.isnan(high) | self._operator(high, low)
        return valid

    def _transform(self, table_data):
        """Transform the table data.

        The transformation consists on replacing the ``high_column_name`` values with the
        difference between it and the ``low_column_name`` values.

        Afterwards, a logarithm is applied to the difference + 1 to ensure that the
        value stays positive when reverted afterwards using an exponential.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        table_data = table_data.copy()
        low, high = self._get_data(table_data)
        diff_column = high - low
        if self._is_datetime:
            diff_column = diff_column.astype(np.float64)

        table_data[self._diff_column_name] = np.log(diff_column + 1)
        return table_data.drop(self._high_column_name, axis=1)

    def reverse_transform(self, table_data):
        """Reverse transform the table data.

        The transformation is reversed by computing an exponential of the difference value,
        subtracting 1 and converting it to the original dtype. Finally, the obtained column
        is added to the ``low_column_name`` column to get back the original
        ``high_column_name`` value.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        table_data = table_data.copy()
        diff_column = np.exp(table_data[self._diff_column_name]) - 1
        if self._dtype != np.dtype('float'):
            diff_column = diff_column.round()

        if self._is_datetime:
            diff_column = diff_column.astype('timedelta64[ns]')

        low = table_data[self._low_column_name].to_numpy()
        table_data[self._high_column_name] = pd.Series(diff_column + low).astype(self._dtype)
        return table_data.drop(self._diff_column_name, axis=1)


class ScalarInequality(Constraint):
    """Ensure an inequality between the ``column_name`` column and a scalar ``value``.

    The transformation works by creating a column with the difference between the ``column_name``
    and ``value`` and storing it in the ``column_name``'s place. The reverse transform adds the
    difference column and the ``value`` to reconstruct the ``column_name``.

    Args:
        column_name (str):
            Name of the column to compare.
        value (float or datetime):
            Scalar value to compare.
        relation (str):
            Describes the relation between ``column_name`` and ``value``.
            Choose one among ``'>'``, ``'>='``, ``'<'``, ``'<='``.
    """

    @staticmethod
    def _validate_inputs(column_name, value, relation):
        if not isinstance(column_name, str):
            raise ValueError('`column_name` must be a string.')

        if not (isinstance(value, (int, float)) or is_datetime_type(value)):
            raise ValueError('`value` must be a number or datetime.')

        if relation not in ['>', '>=', '<', '<=']:
            raise ValueError('`relation` must be one of the following: `>`, `>=`, `<`, `<=`')

    def __init__(self, column_name, value, relation):
        self._validate_inputs(column_name, value, relation)
        self._column_name = column_name
        self._value = value.to_datetime64() if isinstance(value, pd.Timestamp) else value
        self._diff_column_name = f'{self._column_name}#diff'
        self.constraint_columns = tuple([column_name])
        self._is_datetime = None
        self._dtype = None
        self._operator = INEQUALITY_TO_OPERATION[relation]
        super().__init__(handling_strategy='transform', fit_columns_model=False)

    def _get_is_datetime(self, table_data):
        column = table_data[self._column_name].to_numpy()
        is_column_datetime = is_datetime_type(column)
        is_value_datetime = is_datetime_type(self._value)
        is_datetime = is_column_datetime and is_value_datetime

        if not is_datetime and any([is_value_datetime, is_column_datetime]):
            raise ValueError('Both column and value must be datetime.')

        return is_datetime

    def _validate_columns_exist(self, table_data):
        if self._column_name not in table_data.columns:
            raise KeyError(f'The column {self._column_name} was not found in table_data.')

    def _fit(self, table_data):
        """Learn the ``dtype`` of ``_column_name`` and whether the data is datetime.

        Args:
            table_data (pandas.DataFrame):
                The Table data.
        """
        self._validate_columns_exist(table_data)
        self._is_datetime = self._get_is_datetime(table_data)
        self._dtype = table_data[self._column_name].dtypes

    def is_valid(self, table_data):
        """Say whether ``high`` is greater than ``low`` in each row.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.Series:
                Whether each row is valid.
        """
        column = table_data[self._column_name].to_numpy()
        valid = np.isnan(column) | self._operator(column, self._value)
        return valid

    def _transform(self, table_data):
        """Transform the table data.

        The transformation consists on replacing the ``column_name`` values with the
        difference between it and the ``value`` values.

        Afterwards, a logarithm is applied to the difference + 1 to ensure that the
        value stays positive when reverted afterwards using an exponential.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        table_data = table_data.copy()
        column = table_data[self._column_name].to_numpy()
        diff_column = abs(column - self._value)
        if self._is_datetime:
            diff_column = diff_column.astype(np.float64)

        table_data[self._diff_column_name] = np.log(diff_column + 1)
        return table_data.drop(self._column_name, axis=1)

    def reverse_transform(self, table_data):
        """Reverse transform the table data.

        The transformation is reversed by computing an exponential of the difference value,
        subtracting 1 and converting it to the original dtype. Finally, the obtained column
        is added/subtrated from the ``value`` to get back the original ``column_name``.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        table_data = table_data.copy()
        diff_column = np.exp(table_data[self._diff_column_name]) - 1
        if self._dtype != np.dtype('float'):
            diff_column = diff_column.round()

        if self._is_datetime:
            diff_column = diff_column.astype('timedelta64[ns]')

        if self._operator in [np.greater, np.greater_equal]:
            original_column = self._value + diff_column
        else:
            original_column = self._value - diff_column

        table_data[self._column_name] = pd.Series(original_column).astype(self._dtype)
        return table_data.drop(self._diff_column_name, axis=1)


class Positive(ScalarInequality):
    """Ensure the ``column_name`` column is greater than zero.

    The transformation works by applying a logarithm to the ``column_name`` + 1
    to ensure that the value stays positive when reverted afterwards using an exponential.

    Args:
        column_name (str):
            The name of the column that is constrained to be positive.
        strict (bool):
            Whether the comparison of the values should be strict; disclude
            zero ``>`` or include it ``>=``. Currently, this is only respected
            if ``reject_sampling`` or ``all`` handling strategies are used.
    """

    def __init__(self, column_name, strict=False):
        super().__init__(column_name=column_name, value=0, relation='>' if strict else '>=')


class Negative(ScalarInequality):
    """Ensure that the given columns are always negative.

    The transformation works by applying a logarithm to the negative of ``column_name`` + 1
    to ensure that the value stays positive when reverted afterwards using an exponential.

    Args:
        column_name (str):
            The name of the column that is constrained to be negative.
        strict (bool):
            Whether the comparison of the values should be strict, disclude
            zero ``<`` or include it ``<=``. Currently, this is only respected
            if ``reject_sampling`` or ``all`` handling strategies are used.
    """

    def __init__(self, column_name, strict=False):
        super().__init__(column_name=column_name, value=0, relation='<' if strict else '<=')


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


class Range(Constraint):
    """Ensure that the ``middle_column_name`` is between ``low`` and ``high`` columns.

    The transformation strategy works by replacing the ``middle_column_name`` with a
    scaled version and then applying a logit function. The reverse transform
    applies a sigmoid to the data and then scales it back to the original space.

    Args:
        low_column_name (str):
            Name of the column which will be the lower bound.
        middle_column_name (str):
            Name of the column that has to be between the lower bound and upper bound.
        high_column_name (str):
            Name of the column which will be the higher bound.
        strict_boundaries (bool):
            Whether the comparison of the values should be strict ``>=`` or
            not ``>`` when comparing them.
    """

    def __init__(self, low_column_name, middle_column_name, high_column_name,
                 handling_strategy='transform', strict_boundaries=True):

        self.constraint_columns = (low_column_name, middle_column_name, high_column_name)
        self.low_column_name = low_column_name
        self.middle_column_name = middle_column_name
        self.high_column_name = high_column_name
        self.strict_boundaries = strict_boundaries
        self._operator = operator.lt if strict_boundaries else operator.le
        super().__init__(handling_strategy=handling_strategy, fit_columns_model=True)

    def _get_diff_column_name(self, table_data):
        token = '#'
        columns = [self.middle_column_name, self.low_column_name, self.high_column_name]
        components = list(map(str, columns))
        while token.join(components) in table_data.columns:
            token += '#'

        return token.join(components)

    def _get_is_datetime(self, table_data):
        low = table_data[self.low_column_name]
        middle = table_data[self.middle_column_name]
        high = table_data[self.high_column_name]

        is_low_datetime = is_datetime_type(low)
        is_middle_datetime = is_datetime_type(middle)
        is_high_datetime = is_datetime_type(high)
        is_datetime = is_low_datetime and is_high_datetime and is_middle_datetime

        if not is_datetime and any([is_low_datetime, is_middle_datetime, is_high_datetime]):
            raise ValueError('The constraint column and bounds must all be datetime.')

        return is_datetime

    def _fit(self, table_data):
        """Fit the constraint.

        The fit process consists in generating the ``transformed_column`` name and determine
        whether or not the data is ``datetime``.

        Args:
            table_data (pandas.DataFrame):
                The Table data.
        """
        self._dtype = table_data[self.middle_column_name].dtypes
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
        low = table_data[self.low_column_name]
        middle = table_data[self.middle_column_name]
        high = table_data[self.high_column_name]

        satisfy_low_bound = np.logical_or(
            self._operator(low, middle),
            np.isnan(low),
        )
        satisfy_high_bound = np.logical_or(
            self._operator(middle, high),
            np.isnan(high),
        )

        return np.logical_or(
            np.logical_and(satisfy_low_bound, satisfy_high_bound),
            np.isnan(middle),
        )

    def _transform(self, table_data):
        """Transform the table data.

        The transformation consists of scaling the ``middle_column_name``
        (``(middle_column-low)/(high-low)``) and then applying
        a ``logit`` function to the scaled version of the column.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        table_data = table_data.copy()
        low = table_data[self.low_column_name]
        high = table_data[self.high_column_name]

        data = logit(table_data[self.middle_column_name], low, high)
        table_data[self._transformed_column] = data
        table_data = table_data.drop(self.middle_column_name, axis=1)

        return table_data

    def reverse_transform(self, table_data):
        """Reverse transform the table data.

        The reverse transform consists of applying a sigmoid to the transformed
        ``middle_column_name`` and then scaling it back to the original space
        ( ``middle_column * (high - low) / low`` ).

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        table_data = table_data.copy()
        low = table_data[self.low_column_name]
        high = table_data[self.high_column_name]
        data = table_data[self._transformed_column]

        data = sigmoid(data, low, high)
        data = data.clip(low, high)

        if self._is_datetime:
            table_data[self.middle_column_name] = pd.to_datetime(data)
        else:
            table_data[self.middle_column_name] = data.astype(self._dtype)

        table_data = table_data.drop(self._transformed_column, axis=1)

        return table_data


class ScalarRange(Constraint):
    """Ensure that the ``column_name`` is between the range of ``low`` and ``high``.

    The transformation strategy works by replacing the ``column_name`` with a
    scaled version and then applying a logit function. The reverse transform
    applies a sigmoid to the data and then scales it back to the original space.

    Args:
        column_name (str):
            Name of the column that has to be between the lower bound and upper bound.
        low_value (int or float):
            Lower bound on the values of the ``column_name``.
        high_value (int or float):
            Higher bound on the values of the ``column_name``.
        strict_boundaries (bool):
            Whether the comparison of the values should be strict ``>=`` or
            not ``>`` when comparing them.
    """

    def __init__(self, column_name, low_value, high_value, handling_strategy='transform',
                 strict_boundaries=True):

        self.constraint_columns = (column_name,)
        self.column_name = column_name
        self.low_value = low_value
        self.high_value = high_value
        self._operator = operator.lt if strict_boundaries else operator.le

        super().__init__(handling_strategy=handling_strategy, fit_columns_model=False)

    def _get_diff_column_name(self, table_data):
        token = '#'
        columns = [self.column_name, self.low_value, self.high_value]
        components = list(map(str, columns))
        while token.join(components) in table_data.columns:
            token += '#'

        return token.join(components)

    def _get_is_datetime(self, table_data):
        data = table_data[self.column_name]

        is_column_datetime = is_datetime_type(data)
        is_low_datetime = is_datetime_type(self.low_value)
        is_high_datetime = is_datetime_type(self.high_value)
        is_datetime = is_low_datetime and is_high_datetime and is_column_datetime

        if not is_datetime and any([is_low_datetime, is_column_datetime, is_high_datetime]):
            raise ValueError('The constraint column and bounds must all be datetime.')

        return is_datetime

    def _fit(self, table_data):
        """Learn whether or not the ``column_name`` is ``datetime``.

        Args:
            table_data (pandas.DataFrame):
                Table data.
        """
        self._dtype = table_data[self.column_name].dtypes
        self._is_datetime = self._get_is_datetime(table_data)
        self._transformed_column = self._get_diff_column_name(table_data)

    def is_valid(self, table_data):
        """Say whether the ``column_name`` is between the ``low`` and ``high`` values.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.Series:
                Whether each row is valid.
        """
        data = table_data[self.column_name]

        satisfy_low_bound = np.logical_or(
            self._operator(self.low_value, data),
            np.isnan(self.low_value),
        )
        satisfy_high_bound = np.logical_or(
            self._operator(data, self.high_value),
            np.isnan(self.high_value),
        )

        return np.logical_or(
            np.logical_and(satisfy_low_bound, satisfy_high_bound),
            np.isnan(data),
        )

    def _transform(self, table_data):
        """Transform the table data.

        The transformation consists of scaling the ``column_name``
        (``(column-low)/(high-low)``) and then applying
        a logit function to the scaled version of the column.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        table_data = table_data.copy()
        data = logit(table_data[self.column_name], self.low_value, self.high_value)
        table_data[self._transformed_column] = data
        table_data = table_data.drop(self.column_name, axis=1)

        return table_data

    def reverse_transform(self, table_data):
        """Reverse transform the table data.

        The reverse transform consists of applying a sigmoid to the transformed
        ``column_name`` and then scaling it back to the original space
        (``column * (high - low) / low``).

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        table_data = table_data.copy()
        data = table_data[self._transformed_column]

        data = sigmoid(data, self.low_value, self.high_value)
        data = data.clip(self.low_value, self.high_value)

        if self._is_datetime:
            table_data[self.column_name] = pd.to_datetime(data)
        else:
            table_data[self.column_name] = data.astype(self._dtype)

        table_data = table_data.drop(self._transformed_column, axis=1)

        return table_data


class FixedIncrements(Constraint):
    """Ensure every value in a column is a multiple of the specified increment.

    Args:
        column_name (str or list[str]):
            Name of the column.
        increment_value (int):
            The increment that each value in the column must be a multiple of. Must be greater
            than 0.
        handling_strategy (str):
            How this Constraint should be handled, which can be ``transform``
            or ``reject_sampling``. Defaults to ``transform``.
    """

    _dtype = None

    def __init__(self, column_name, increment_value, handling_strategy='transform'):
        if increment_value <= 0:
            raise ValueError('The increment_value must be greater than 0.')

        if increment_value % 1 != 0:
            raise ValueError('The increment_value must be a whole number.')

        self.increment_value = increment_value
        self.column_name = column_name
        super().__init__(handling_strategy=handling_strategy, fit_columns_model=False)

    def is_valid(self, table_data):
        """Determine if the data is evenly divisible by the increment.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.Series:
                Whether each row is valid.
        """
        isnan = pd.isnull(table_data[self.column_name])
        is_divisible = table_data[self.column_name] % self.increment_value == 0
        return is_divisible | isnan

    def _fit(self, table_data):
        """Learn the dtype of the column.

        Args:
            table_data (pandas.DataFrame):
                The Table data.
        """
        self._dtype = table_data[self.column_name].dtype

    def _transform(self, table_data):
        """Transform the table_data.

        The transformation works by dividing each value by the increment.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Data divided by increment.
        """
        table_data = table_data.copy()
        table_data[self.column_name] = table_data[self.column_name] / self.increment_value
        return table_data

    def reverse_transform(self, table_data):
        """Convert column to multiples of the increment.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Data as multiples of the increment.
        """
        column = table_data[self.column_name].round()
        table_data[self.column_name] = (column * self.increment_value).astype(self._dtype)
        return table_data


class OneHotEncoding(Constraint):
    """Ensure the appropriate columns are one hot encoded.

    This constraint allows the user to specify a list of columns where each row
    is a one hot vector. During the reverse transform, the output of the model
    is transformed so that the column with the largest value is set to 1 while
    all other columns are set to 0.

    Args:
        column_names (list[str]):
            Names of the columns containing one hot rows.
        handling_strategy (str):
            How this Constraint should be handled, which can be ``transform``
            or ``reject_sampling`` (not recommended). Defaults to ``transform``.
    """

    def __init__(self, column_names, handling_strategy='transform'):
        self._column_names = column_names
        self.constraint_columns = tuple(column_names)
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

        condition_columns = [col for col in self._column_names if col in table_data.columns]
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

            if len(condition_columns) == len(self._column_names) - 1:
                proposed_table_data = sub_table_data.copy()
                for column in self._column_names:
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

            for column in self._column_names:
                if column not in condition_columns:
                    sub_table_data[column] = proposed_table_data[column].values
                else:
                    sub_table_data[column] = float('-inf')

            table_data.loc[~has_one, self._column_names] = self.reverse_transform(sub_table_data)

        if has_one.sum() > 0:
            for column in self._column_names:
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
        one_hot_data = table_data[self._column_names]

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

        one_hot_data = table_data[self._column_names]
        transformed_data = np.zeros_like(one_hot_data.values)
        transformed_data[np.arange(len(one_hot_data)), np.argmax(one_hot_data.values, axis=1)] = 1
        table_data[self._column_names] = transformed_data

        return table_data


class Unique(Constraint):
    """Ensure that each value for a specified column/group of columns is unique.

    This constraint is provided a list of columns, and guarantees that every
    unique combination of those columns appears at most once in the sampled
    data.

    Args:
        column_names (list[str]):
            List of name(s) of the column(s) to keep unique.
    """

    def __init__(self, column_names):
        self.column_names = column_names
        self.constraint_columns = tuple(self.column_names)
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
        return table_data.groupby(self.column_names, dropna=False).cumcount() == 0
