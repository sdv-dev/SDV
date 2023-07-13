"""Table constraints.

This module contains constraints that are evaluated within a single table,
and which can affect one or more columns at a time, as well as one or more
rows.

Note: the data produced by the reverse transform of a constraint does not
necessarily satisfy that constraint. Any invalid rows produced will have to
be reject sampled at some later stage.

Currently implemented constraints are:

    * CustomConstraint: Simple constraint to be set up by passing the python
      functions that will be used for transformation, reverse transformation
      and validation. It can be created through the ``create_custom_constraint_class`` method.
    * FixedCombinations: Ensure that the combinations of values
      across several columns are the same after sampling.
    * Inequality: Ensure that the value in one column is always greater than
      the value in another column.
    * ScalarInequality: Ensure that the value in one column is always greater/smaller
      than some scalar.
    * Positive: Ensure that the values in given columns are always positive.
    * Negative: Ensure that the values in given columns are always negative.
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

from sdv.constraints.base import Constraint
from sdv.constraints.errors import (
    AggregateConstraintsError, ConstraintMetadataError, FunctionError, InvalidFunctionError)
from sdv.constraints.utils import (
    cast_to_datetime64, compute_nans_column, get_datetime_diff, logit, matches_datetime_format,
    revert_nans_columns, sigmoid)
from sdv.utils import convert_to_timedelta, create_unique_name, is_datetime_type

INEQUALITY_TO_OPERATION = {
    '>': np.greater,
    '>=': np.greater_equal,
    '<': np.less,
    '<=': np.less_equal
}


def _validate_inputs_custom_constraint(is_valid_fn, transform_fn=None, reverse_transform_fn=None):
    if not callable(is_valid_fn):
        raise ValueError('`is_valid` must be a function.')

    # Transform & reverse are optional but should be provided together or not at all
    if transform_fn is None and reverse_transform_fn is not None:
        raise ValueError('Missing parameter `transform_fn`.')
    if transform_fn is not None and reverse_transform_fn is None:
        raise ValueError('Missing parameter `reverse_transform_fn`.')

    if transform_fn is not None and not callable(transform_fn):
        raise ValueError('`transform_fn` must be a function.')
    if reverse_transform_fn is not None and not callable(reverse_transform_fn):
        raise ValueError('`reverse_transform_fn` must be a function.')


class _RecreateCustomConstraint():
    def __call__(self, is_valid_fn, transform_fn, reverse_transform_fn):
        constraint_class = _RecreateCustomConstraint()
        constraint_class.__class__ = create_custom_constraint_class(
            is_valid_fn=is_valid_fn,
            transform_fn=transform_fn,
            reverse_transform_fn=reverse_transform_fn
        )

        return constraint_class


def create_custom_constraint_class(is_valid_fn, transform_fn=None, reverse_transform_fn=None):
    """Create a CustomConstraint class.

    Creates a constraint class which uses the ``transform``, ``reverse_transform`` and
    ``is_valid`` methods given in the arguments.

    Args:
        is_valid (callable):
            Function to replace the ``is_valid`` method.
        transform (callable):
            Function to replace the ``transform`` method.
        reverse_transform (callable):
            Function to replace the ``reverse_transform`` method.

    Returns:
        CustomConstraint class:
            A constraint with custom ``is_valid``/``transform``/``reverse_transform`` methods.
    """
    _validate_inputs_custom_constraint(is_valid_fn, transform_fn, reverse_transform_fn)

    class CustomConstraint(Constraint):
        """CustomConstraint class.

        Args:
            column_names (list):
                List of strings representing column names involved in this constraint.
            **kwargs (dict):
                Any kwargs necessary for constraint.
        """

        @classmethod
        def _validate_inputs(cls, **kwargs):
            if 'column_names' not in set(kwargs):
                errors = [
                    ConstraintMetadataError(
                        "Missing required values {'column_names'} in a"
                        ' CustomConstraint constraint.'
                    )
                ]
                raise AggregateConstraintsError(errors)

        def __reduce__(self):
            return (
                _RecreateCustomConstraint(),
                (is_valid_fn, transform_fn, reverse_transform_fn),
                self.__dict__
            )

        def __init__(self, column_names, **kwargs):
            self.column_names = column_names
            self.kwargs = kwargs
            self.constraint_columns = tuple(column_names)

        def is_valid(self, data):
            """Check whether the column values are valid.

            Args:
                table_data (pandas.DataFrame):
                    Table data.

            Returns:
                pandas.Series:
                    Whether each row is valid.
            """
            valid = is_valid_fn(self.column_names, data, **self.kwargs)
            if len(valid) != data.shape[0]:
                raise InvalidFunctionError(
                    '`is_valid_fn` did not produce exactly 1 True/False value for each row.')

            if not isinstance(valid, pd.Series):
                raise ValueError(
                    "The custom 'is_valid' function returned an unsupported type. "
                    'The returned object must be a pandas.Series'
                )

            return valid

        def transform(self, data):
            """Transform the table data.

            Args:
                table_data (pandas.DataFrame):
                    Table data.

            Returns:
                pandas.DataFrame:
                    Transformed data.
            """
            data = data.copy()
            if transform_fn is None:
                return data

            try:
                transformed_data = transform_fn(self.column_names, data, **self.kwargs)
                if data.shape[0] != transformed_data.shape[0]:
                    raise InvalidFunctionError(
                        'Transformation did not produce the same number of rows as the original')

                self.reverse_transform(transformed_data.copy())
                return transformed_data

            except InvalidFunctionError as e:
                raise e

            except Exception:
                raise FunctionError

        def reverse_transform(self, data):
            """Reverse transform the table data.

            Args:
                table_data (pandas.DataFrame):
                    Table data.

            Returns:
                pandas.DataFrame:
                    Transformed data.
            """
            data = data.copy()
            if reverse_transform_fn is None:
                return data

            transformed_data = reverse_transform_fn(self.column_names, data, **self.kwargs)
            if data.shape[0] != transformed_data.shape[0]:
                raise InvalidFunctionError(
                    'Reverse transform did not produce the same number of rows as the original.'
                )

            return transformed_data

    return CustomConstraint


class FixedCombinations(Constraint):
    """Ensure that the combinations across multiple columns are fixed.

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
    """

    _separator = None
    _joint_column = None
    _combinations_to_uuids = None
    _uuids_to_combinations = None

    @staticmethod
    def _validate_metadata_specific_to_constraint(metadata, **kwargs):
        invalid_columns = []
        column_names = kwargs.get('column_names')
        for column in column_names:
            if metadata.columns[column]['sdtype'] not in ['boolean', 'categorical']:
                invalid_columns.append(column)

        if invalid_columns:
            columns = '", "'.join(invalid_columns)
            raise ConstraintMetadataError(f'Invalid columns ("{columns}") supplied to a '
                                          'FixedCombinations constraint. This constraint only '
                                          'supports boolean and categorical columns.')

    def __init__(self, column_names):
        if len(column_names) < 2:
            raise ValueError('FixedCombinations requires at least two constraint columns.')

        self._columns = column_names
        self.constraint_columns = tuple(column_names)

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
        combinations = table_data[self._columns].itertuples(index=False, name=None)
        uuids = map(self._combinations_to_uuids.get, combinations)
        table_data[self._joint_column] = list(uuids)
        return table_data.drop(self._columns, axis=1)

    def _reverse_transform(self, table_data):
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
            not ``>``. Defaults to False.
    """

    @staticmethod
    def _validate_init_inputs(low_column_name, high_column_name, strict_boundaries):
        if not (isinstance(low_column_name, str) and isinstance(high_column_name, str)):
            raise ValueError('`low_column_name` and `high_column_name` must be strings.')

        if not isinstance(strict_boundaries, bool):
            raise ValueError('`strict_boundaries` must be a boolean.')

    @classmethod
    def _validate_metadata_columns(cls, metadata, **kwargs):
        kwargs['column_names'] = [kwargs.get('high_column_name'), kwargs.get('low_column_name')]
        super()._validate_metadata_columns(metadata, **kwargs)

    @staticmethod
    def _validate_metadata_specific_to_constraint(metadata, **kwargs):
        high = kwargs.get('high_column_name')
        low = kwargs.get('low_column_name')
        high_sdtype = metadata.columns.get(high, {}).get('sdtype')
        low_sdtype = metadata.columns.get(low, {}).get('sdtype')
        both_datetime = high_sdtype == low_sdtype == 'datetime'
        both_numerical = high_sdtype == low_sdtype == 'numerical'
        if not (both_datetime or both_numerical) and not (high is None or low is None):
            raise ConstraintMetadataError(
                'An Inequality constraint is being applied to columns with mismatched sdtypes'
                f' {[high, low]}. Both columns must be either numerical or datetime.'
            )

    def __init__(self, low_column_name, high_column_name, strict_boundaries=False):
        self._validate_init_inputs(low_column_name, high_column_name, strict_boundaries)
        self._low_column_name = low_column_name
        self._high_column_name = high_column_name
        self._diff_column_name = f'{self._low_column_name}#{self._high_column_name}'
        self._operator = np.greater if strict_boundaries else np.greater_equal
        self.constraint_columns = (low_column_name, high_column_name)
        self._dtype = None
        self._is_datetime = None
        self._nan_column_name = None

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
        missing = {self._low_column_name, self._high_column_name} - set(table_data.columns)
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
        if self._is_datetime and self._dtype == 'O':
            low = cast_to_datetime64(low)
            high = cast_to_datetime64(high)

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
        low, high = self._get_data(table_data)
        if self._is_datetime:
            diff_column = get_datetime_diff(high, low)
        else:
            diff_column = high - low

        self._diff_column_name = create_unique_name(self._diff_column_name, table_data.columns)
        table_data[self._diff_column_name] = np.log(diff_column + 1)

        nan_col = compute_nans_column(table_data, [self._low_column_name, self._high_column_name])
        if nan_col is not None:
            self._nan_column_name = create_unique_name(nan_col.name, table_data.columns)
            table_data[self._nan_column_name] = nan_col
            if self._is_datetime:
                mean_value_low = table_data[self._low_column_name].mode()[0]
            else:
                mean_value_low = table_data[self._low_column_name].mean()
            table_data = table_data.fillna({
                self._low_column_name: mean_value_low,
                self._diff_column_name: table_data[self._diff_column_name].mean()
            })

        return table_data.drop(self._high_column_name, axis=1)

    def _reverse_transform(self, table_data):
        """Reverse transform the table data.

        The transformation is reversed by computing an exponential of the difference value,
        subtracting 1 and converting it to the original dtype. Finally, the obtained column
        is added to the ``low_column_name`` column to get back the original ``high_column_name``
        value.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        diff_column = np.exp(table_data[self._diff_column_name]) - 1
        if self._dtype != np.dtype('float'):
            diff_column = diff_column.round()

        if self._is_datetime:
            diff_column = convert_to_timedelta(diff_column)

        low = table_data[self._low_column_name].to_numpy()
        if self._is_datetime and self._dtype == 'O':
            low = cast_to_datetime64(low)

        table_data[self._high_column_name] = pd.Series(diff_column + low).astype(self._dtype)

        if self._nan_column_name and self._nan_column_name in table_data.columns:
            table_data = revert_nans_columns(table_data, self._nan_column_name)

        return table_data.drop(self._diff_column_name, axis=1)


class ScalarInequality(Constraint):
    """Ensure an inequality between the ``column_name`` column and a scalar ``value``.

    The transformation works by creating a column with the difference between the ``column_name``
    and ``value`` and storing it in the ``column_name``'s place. The reverse transform adds the
    difference column and the ``value`` to reconstruct the ``column_name``.

    Args:
        column_name (str):
            Name of the column to compare.
        relation (str):
            Describes the relation between ``column_name`` and ``value``.
            Choose one among ``'>'``, ``'>='``, ``'<'``, ``'<='``.
        value (float or datetime):
            Scalar value to compare.
    """

    @classmethod
    def _validate_inputs(cls, **kwargs):
        errors = []
        try:
            super()._validate_inputs(**kwargs)
        except Exception as e:
            errors.append(e)

        if 'relation' in kwargs and kwargs['relation'] not in {'>', '>=', '<', '<='}:
            wrong_relation = {kwargs['relation']}
            errors.append(ConstraintMetadataError(
                f'Invalid relation value {wrong_relation} in a ScalarInequality constraint.'
                " The relation must be one of: '>', '>=', '<' or '<='."
            ))

        if errors:
            raise AggregateConstraintsError(errors)

    @staticmethod
    def _validate_metadata_specific_to_constraint(metadata, **kwargs):
        column_name = kwargs.get('column_name')
        sdtype = metadata.columns.get(column_name, {}).get('sdtype')
        value = kwargs.get('value')
        if sdtype == 'numerical':
            if not isinstance(value, (int, float)):
                raise ConstraintMetadataError("'value' must be an int or float.")

        elif sdtype == 'datetime':
            datetime_format = metadata.columns.get(column_name).get('datetime_format')
            matches_format = matches_datetime_format(value, datetime_format)
            if not matches_format:
                raise ConstraintMetadataError(
                    "'value' must be a datetime string of the right format."
                )

        else:
            raise ConstraintMetadataError(
                'A ScalarInequality constraint is being applied '
                'to columns with mismatched sdtypes. '
                'Numerical columns must be compared to integer or float values. '
                'Datetimes column must be compared to datetime strings.'
            )

    @staticmethod
    def _validate_init_inputs(column_name, value, relation):
        value_is_datetime = is_datetime_type(value)
        if not isinstance(column_name, str):
            raise ValueError('`column_name` must be a string.')

        if relation not in ['>', '>=', '<', '<=']:
            raise ValueError('`relation` must be one of the following: `>`, `>=`, `<`, `<=`')

        if not (isinstance(value, (int, float)) or value_is_datetime):
            raise ValueError('`value` must be a number or a string that represents a datetime.')

        if value_is_datetime and not isinstance(value, str):
            raise ValueError('Datetime must be represented as a string.')

    def __init__(self, column_name, relation, value):
        self._validate_init_inputs(column_name, value, relation)
        self._value = cast_to_datetime64(value) if is_datetime_type(value) else value
        self._column_name = column_name
        self._diff_column_name = f'{self._column_name}#diff'
        self.constraint_columns = (column_name,)
        self._is_datetime = None
        self._dtype = None
        self._operator = INEQUALITY_TO_OPERATION[relation]

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
        if self._is_datetime and self._dtype == 'O':
            column = cast_to_datetime64(column)

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
        column = table_data[self._column_name].to_numpy()
        if self._is_datetime:
            column = cast_to_datetime64(column)
            diff_column = abs(column - self._value)
            diff_column = diff_column.astype(np.float64)
        else:
            diff_column = abs(column - self._value)

        self._diff_column_name = create_unique_name(self._diff_column_name, table_data.columns)
        table_data[self._diff_column_name] = np.log(diff_column + 1)
        return table_data.drop(self._column_name, axis=1)

    def _reverse_transform(self, table_data):
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
        diff_column = np.exp(table_data[self._diff_column_name]) - 1
        if self._dtype != np.dtype('float'):
            diff_column = diff_column.round()

        if self._is_datetime:
            diff_column = convert_to_timedelta(diff_column)

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
        strict_boundaries (bool):
            Whether the comparison of the values should be strict; disclude
            zero ``>`` or include it ``>=``.
    """

    @staticmethod
    def _validate_metadata_specific_to_constraint(metadata, **kwargs):
        column_name = kwargs.get('column_name')
        sdtype = metadata.columns.get(column_name, {}).get('sdtype')
        if sdtype != 'numerical':
            raise ConstraintMetadataError(
                f'A Positive constraint is being applied to an invalid column '
                f"'{column_name}'. This constraint is only defined for numerical columns."
            )

    def __init__(self, column_name, strict_boundaries=False):
        relation = '>' if strict_boundaries else '>='
        super().__init__(column_name=column_name, relation=relation, value=0)


class Negative(ScalarInequality):
    """Ensure that the given columns are always negative.

    The transformation works by applying a logarithm to the negative of ``column_name`` + 1
    to ensure that the value stays positive when reverted afterwards using an exponential.

    Args:
        column_name (str):
            The name of the column that is constrained to be negative.
        strict_boundaries (bool):
            Whether the comparison of the values should be strict, disclude
            zero ``<`` or include it ``<=``.
    """

    @staticmethod
    def _validate_metadata_specific_to_constraint(metadata, **kwargs):
        column_name = kwargs.get('column_name')
        sdtype = metadata.columns.get(column_name, {}).get('sdtype')
        if sdtype != 'numerical':
            raise ConstraintMetadataError(
                f'A Negative constraint is being applied to an invalid column '
                f"'{column_name}'. This constraint is only defined for numerical columns."
            )

    def __init__(self, column_name, strict_boundaries=False):
        relation = '<' if strict_boundaries else '<='
        super().__init__(column_name=column_name, relation=relation, value=0)


class Range(Constraint):
    """Ensure that the ``middle_column_name`` is between ``low`` and ``high`` columns.

    The transformation strategy works the same as the Inequality constraint but with two
    columns instead of one. We compute the difference between the ``middle_column_name``
    and the ``low`` column and then apply a logarithm to the difference + 1 to ensure
    that the value stays positive when reverted afterwards using an exponential.
    We do the same for the difference between the ``high`` and ``middle_column_name``.


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
            Defaults to True.
    """

    @classmethod
    def _validate_metadata_columns(cls, metadata, **kwargs):
        high = kwargs.get('high_column_name')
        low = kwargs.get('low_column_name')
        middle = kwargs.get('middle_column_name')
        kwargs['column_names'] = [high, low, middle]
        super()._validate_metadata_columns(metadata, **kwargs)

    @staticmethod
    def _validate_metadata_specific_to_constraint(metadata, **kwargs):
        high = kwargs.get('high_column_name')
        low = kwargs.get('low_column_name')
        middle = kwargs.get('middle_column_name')
        high_sdtype = metadata.columns.get(high, {}).get('sdtype')
        low_sdtype = metadata.columns.get(low, {}).get('sdtype')
        middle_sdtype = metadata.columns.get(middle, {}).get('sdtype')
        all_datetime = high_sdtype == low_sdtype == middle_sdtype == 'datetime'
        all_numerical = high_sdtype == low_sdtype == middle_sdtype == 'numerical'
        if not (all_datetime or all_numerical) and \
           not (high is None or low is None or middle is None):
            raise ConstraintMetadataError(
                'A Range constraint is being applied to columns with mismatched sdtypes '
                f'{[high, middle, low]}. All columns must be either numerical or datetime.'
            )

    def __init__(self, low_column_name, middle_column_name, high_column_name,
                 strict_boundaries=True):

        self.constraint_columns = (low_column_name, middle_column_name, high_column_name)
        self.low_column_name = low_column_name
        self.middle_column_name = middle_column_name
        self.high_column_name = high_column_name
        self.nan_column_name = None
        self._is_datetime = None
        self._dtype = None
        self.strict_boundaries = strict_boundaries
        self._operator = operator.lt if strict_boundaries else operator.le

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

        Args:
            table_data (pandas.DataFrame):
                The Table data.
        """
        self._dtype = table_data[self.middle_column_name].dtypes
        self._is_datetime = self._get_is_datetime(table_data)

        self.low_diff_column_name = f'{self.low_column_name}#{self.middle_column_name}'
        self.high_diff_column_name = f'{self.middle_column_name}#{self.high_column_name}'
        self.low_diff_column_name = create_unique_name(
            self.low_diff_column_name, table_data.columns
        )
        self.high_diff_column_name = create_unique_name(
            self.high_diff_column_name, table_data.columns
        )

    def is_valid(self, table_data):
        """Say whether the ``middle`` column is between the ``low`` and ``high`` columns.

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

        low_is_nan = low.isna()
        middle_is_nan = middle.isna()
        high_is_nan = high.isna()

        low_lt_middle = self._operator(low, middle) | low_is_nan | middle_is_nan
        middle_lt_high = self._operator(middle, high) | middle_is_nan | high_is_nan
        low_lt_high = self._operator(low, high) | low_is_nan | high_is_nan

        return low_lt_middle & middle_lt_high & low_lt_high

    def _transform(self, table_data):
        """Transform the table data.

        The transformation consists in replacing ``middle`` and ``high`` by the difference
        between them and ``low`` and ``middle`` respectively. To avoid negative values, the
        logarithm of the difference + 1 is taken.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        # Using ``to_numpy`` since ``get_datetime_diff`` requires ``numpy.ndarray``
        low = table_data[self.low_column_name].to_numpy()
        middle = table_data[self.middle_column_name].to_numpy()
        high = table_data[self.high_column_name].to_numpy()

        if self._is_datetime:
            low_diff_column = get_datetime_diff(middle, low, self._dtype)
            high_diff_column = get_datetime_diff(high, middle, self._dtype)
        else:
            low_diff_column = middle - low
            high_diff_column = high - middle

        table_data[self.low_diff_column_name] = np.log(low_diff_column + 1)
        table_data[self.high_diff_column_name] = np.log(high_diff_column + 1)

        list_columns_nans = [self.low_column_name, self.middle_column_name, self.high_column_name]
        nan_column = compute_nans_column(table_data, list_columns_nans)
        if nan_column is not None:
            self.nan_column_name = create_unique_name(nan_column.name, table_data.columns)
            table_data[self.nan_column_name] = nan_column
            if self._is_datetime:
                mean_value_low = table_data[self.low_column_name].mode()[0]
            else:
                mean_value_low = table_data[self.low_column_name].mean()

            table_data = table_data.fillna({
                self.low_column_name: mean_value_low,
                self.low_diff_column_name: table_data[self.low_diff_column_name].mean(),
                self.high_diff_column_name: table_data[self.high_diff_column_name].mean()
            })

        return table_data.drop([self.middle_column_name, self.high_column_name], axis=1)

    def _reverse_transform(self, table_data):
        """Reverse transform the table data.

        The reverse transformation consists in replacing ``low_diff_column`` and
        ``high_diff_column`` by the sum of ``low`` and ``low_diff_column`` and ``middle``
        and ``high_diff_column`` respectively.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        low_diff_column = np.exp(table_data[self.low_diff_column_name]) - 1
        high_diff_column = np.exp(table_data[self.high_diff_column_name]) - 1
        if self._dtype != np.dtype('float'):
            low_diff_column = low_diff_column.round()
            high_diff_column = high_diff_column.round()

        if self._is_datetime:
            low_diff_column = convert_to_timedelta(low_diff_column)
            high_diff_column = convert_to_timedelta(high_diff_column)

        low = table_data[self.low_column_name].to_numpy()
        if self._is_datetime and self._dtype == 'O':
            low = cast_to_datetime64(low)

        middle = pd.Series(low_diff_column + low).astype(self._dtype)
        table_data[self.middle_column_name] = middle
        table_data[self.high_column_name] = pd.Series(
            high_diff_column + middle.to_numpy()
        ).astype(self._dtype)

        if self.nan_column_name in table_data.columns:
            table_data = revert_nans_columns(table_data, self.nan_column_name)

        return table_data.drop([self.low_diff_column_name, self.high_diff_column_name], axis=1)


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

    @staticmethod
    def _validate_init_inputs(low_value, high_value):
        values_are_datetimes = is_datetime_type(low_value) and is_datetime_type(high_value)
        values_are_strings = isinstance(low_value, str) and isinstance(high_value, str)
        if values_are_datetimes and not values_are_strings:
            raise ValueError('Datetime must be represented as a string.')

        values_are_numerical = bool(
            isinstance(low_value, (int, float)) and isinstance(high_value, (int, float))
        )
        if not (values_are_numerical or values_are_datetimes):
            raise ValueError(
                '``low_value`` and ``high_value`` must be a number or a string that '
                'represents a datetime.'
            )

    @staticmethod
    def _validate_metadata_specific_to_constraint(metadata, **kwargs):
        column_name = kwargs.get('column_name')
        if column_name not in metadata.columns:
            raise ConstraintMetadataError(
                f'A ScalarRange constraint is being applied to invalid column names '
                f'({column_name}). The columns must exist in the table.'
            )
        sdtype = metadata.columns.get(column_name).get('sdtype')
        high_value = kwargs.get('high_value')
        low_value = kwargs.get('low_value')
        if sdtype == 'numerical':
            if not isinstance(high_value, (int, float)) or not isinstance(low_value, (int, float)):
                raise ConstraintMetadataError(
                    "Both 'high_value' and 'low_value' must be ints or floats"
                )

        elif sdtype == 'datetime':
            datetime_format = metadata.columns.get(column_name, {}).get('datetime_format')
            high_matches_format = matches_datetime_format(high_value, datetime_format)
            low_matches_format = matches_datetime_format(low_value, datetime_format)
            if not (low_matches_format and high_matches_format):
                raise ConstraintMetadataError(
                    "Both 'high_value' and 'low_value' must be a datetime string of the right "
                    'format'
                )

        else:
            raise ConstraintMetadataError(
                'A ScalarRange constraint is being applied to columns with mismatched sdtypes. '
                'Numerical columns must be compared to integer or float values. '
                'Datetimes column must be compared to datetime strings.'
            )

    def __init__(self, column_name, low_value, high_value, strict_boundaries=True):
        self.constraint_columns = (column_name,)
        self._column_name = column_name
        self._validate_init_inputs(low_value, high_value)
        self._is_datetime = None
        self._datetime_format = None
        self._low_value = low_value
        self._high_value = high_value
        self._operator = operator.lt if strict_boundaries else operator.le

    def _get_diff_column_name(self, table_data):
        token = '#'
        columns = [self._column_name, self._low_value, self._high_value]
        components = list(map(str, columns))
        while token.join(components) in table_data.columns:
            token += '#'

        return token.join(components)

    def _get_is_datetime(self, table_data):
        data = table_data[self._column_name]

        is_column_datetime = is_datetime_type(data)
        is_low_datetime = is_datetime_type(self._low_value)
        is_high_datetime = is_datetime_type(self._high_value)
        is_datetime = is_low_datetime and is_high_datetime and is_column_datetime

        if not is_datetime and any([is_low_datetime, is_column_datetime, is_high_datetime]):
            raise ValueError('The constraint column and bounds must all be datetime.')

        return is_datetime

    def _fit(self, table_data):
        """Learn whether or not the ``column_name`` is ``UnixTimestampEncoder``.

        Args:
            table_data (pandas.DataFrame):
                Table data.
        """
        self._dtype = table_data[self._column_name].dtypes
        self._is_datetime = self._get_is_datetime(table_data)
        self._transformed_column = self._get_diff_column_name(table_data)
        if self._is_datetime:
            self._low_value = cast_to_datetime64(self._low_value)
            self._high_value = cast_to_datetime64(self._high_value)

    def is_valid(self, table_data):
        """Say whether the ``column_name`` is between the ``low`` and ``high`` values.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.Series:
                Whether each row is valid.
        """
        data = table_data[self._column_name]

        if self._is_datetime:
            data = cast_to_datetime64(data)

        satisfy_low_bound = np.logical_or(
            self._operator(self._low_value, data),
            np.isnan(self._low_value),
        )
        satisfy_high_bound = np.logical_or(
            self._operator(data, self._high_value),
            np.isnan(self._high_value),
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
        data = table_data[self._column_name]
        if self._is_datetime:
            data = cast_to_datetime64(table_data[self._column_name])

        data = logit(data, self._low_value, self._high_value)
        table_data[self._transformed_column] = data
        table_data = table_data.drop(self._column_name, axis=1)

        return table_data

    def _reverse_transform(self, table_data):
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
        data = table_data[self._transformed_column]

        data = sigmoid(data, self._low_value, self._high_value)
        data = data.clip(self._low_value, self._high_value)

        if self._is_datetime:
            table_data[self._column_name] = pd.to_datetime(data)
        else:
            table_data[self._column_name] = data.astype(self._dtype)

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
    """

    _dtype = None

    @classmethod
    def _validate_inputs(cls, **kwargs):
        errors = []
        try:
            super()._validate_inputs(**kwargs)
        except AggregateConstraintsError as agg_error:
            errors.extend(agg_error.errors)
        except Exception as e:
            errors.append(e)

        if 'increment_value' in kwargs and kwargs['increment_value'] <= 0:
            wrong_increment = {kwargs['increment_value']}
            errors.append(ConstraintMetadataError(
                f'Invalid increment value {wrong_increment} in a FixedIncrements constraint.'
                ' Increments must be positive integers.'
            ))

        if errors:
            raise AggregateConstraintsError(errors)

    def __init__(self, column_name, increment_value):
        if increment_value <= 0:
            raise ValueError('The increment_value must be greater than 0.')

        if increment_value % 1 != 0:
            raise ValueError('The increment_value must be a whole number.')

        self.increment_value = increment_value
        self.column_name = column_name
        self.constraint_columns = (column_name,)

    def is_valid(self, table_data):
        """Determine if the data is evenly divisible by the increment.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.Series:
                Whether each row is valid.
        """
        isnan = pd.isna(table_data[self.column_name])
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
        table_data[self.column_name] = table_data[self.column_name] / self.increment_value
        return table_data

    def _reverse_transform(self, table_data):
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
    """

    def __init__(self, column_names):
        self._column_names = column_names
        self.constraint_columns = tuple(column_names)

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

    def _reverse_transform(self, table_data):
        """Reverse transform the table data.

        Set the column with the largest value to one, set all other columns to zero.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        one_hot_data = table_data[self._column_names]
        transformed_data = np.zeros_like(one_hot_data.to_numpy())
        max_category_indices = np.argmax(one_hot_data.to_numpy(), axis=1)
        transformed_data[np.arange(len(one_hot_data)), max_category_indices] = 1
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

    @staticmethod
    def _validate_metadata_specific_to_constraint(metadata, **kwargs):
        column_names = kwargs.get('column_names')
        keys = set()
        if isinstance(metadata.primary_key, tuple):
            keys.update(metadata.primary_key)
        else:
            keys.add(metadata.primary_key)

        for key in metadata.alternate_keys:
            if isinstance(key, tuple):
                keys.update(key)
            else:
                keys.add(key)

        if len(set(column_names) - keys) == 0:
            raise ConstraintMetadataError(
                f"A Unique constraint is being applied to columns '{column_names}'. "
                'These columns are already a key for that table.'
            )

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
