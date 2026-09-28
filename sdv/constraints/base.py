"""Base Constraint class."""

import copy
import importlib
import inspect
import logging

import pandas as pd
from copulas.multivariate.gaussian import GaussianMultivariate
from copulas.univariate import GaussianUnivariate
from rdt import HyperTransformer
from rdt.transformers import BinaryEncoder, FloatFormatter, OneHotEncoder, UnixTimestampEncoder

from sdv._utils import _format_invalid_values_string, _groupby_list
from sdv.constraints.errors import (
    AggregateConstraintsError,
    ConstraintMetadataError,
    MissingConstraintColumnError,
)
from sdv.errors import ConstraintsNotMetError

LOGGER = logging.getLogger(__name__)


def _get_qualified_name(obj):
    """Return the Fully Qualified Name from an instance or class."""
    module = obj.__module__
    if hasattr(obj, '__name__'):
        obj_name = obj.__name__
    else:
        obj_name = obj.__class__.__name__

    return module + '.' + obj_name


def _module_contains_callable_name(obj):
    """Return if module contains the name of the callable object."""
    if hasattr(obj, '__name__'):
        obj_name = obj.__name__
    else:
        obj_name = obj.__class__.__name__
    return obj_name in importlib.import_module(obj.__module__).__dict__


def get_subclasses(cls):
    """Recursively find subclasses for the current class object."""
    subclasses = {}
    for subclass in cls.__subclasses__():
        subclasses[subclass.__name__] = subclass
        subclasses.update(get_subclasses(subclass))

    return subclasses


def import_object(obj):
    """Import an object from its qualified name."""
    if isinstance(obj, str):
        package, name = obj.rsplit('.', 1)
        return getattr(importlib.import_module(package), name)

    return obj


class ConstraintMeta(type):
    """Metaclass for Constraints.

    This metaclass replaces the ``__init__`` method with a new function
    that stores the arguments passed to the __init__ method in a dict
    as the attribute ``__kwargs__``.

    This allows us to later on dump the class definition as a dict.
    """

    def __init__(self, name, bases, attr):  # noqa: N804
        super().__init__(name, bases, attr)

        old__init__ = self.__init__
        signature = inspect.signature(old__init__)
        arg_names = list(signature.parameters.keys())[1:]

        def __init__(self, *args, **kwargs):  # noqa: N807
            class_name = self.__class__.__name__
            if name == class_name:
                self.__kwargs__ = copy.deepcopy(kwargs)
                self.__kwargs__.update(dict(zip(arg_names, args)))
                self.metadata = self.__kwargs__.get('metadata')
                if 'metadata' in kwargs:
                    del kwargs['metadata']

            old__init__(self, *args, **kwargs)

        __init__.__doc__ = old__init__.__doc__
        __init__.__signature__ = signature
        self.__init__ = __init__


class Constraint(metaclass=ConstraintMeta):
    """Constraint base class.

    This class is not intended to be used directly and should rather be
    subclassed to create different types of constraints.

    Attributes:
        constraint_columns (tuple[str]):
            The names of the columns used by this constraint.
        rebuild_columns (tuple[str]):
            The names of the columns that this constraint will rebuild during
            ``reverse_transform``.
    """

    constraint_columns = ()
    _hyper_transformer = None

    @classmethod
    def _validate_inputs(cls, **kwargs):
        errors = []
        required_args = []
        args = []
        params = inspect.signature(cls).parameters
        for arg_name, value in params.items():
            args.append(arg_name)
            if value.default is inspect._empty:
                required_args.append(arg_name)

        missing_values = set(required_args) - set(kwargs)
        constraint = cls.__name__
        article = 'an' if constraint == 'Inequality' else 'a'
        if missing_values:
            errors.append(
                ValueError(
                    f'Missing required values {missing_values} '
                    f'in {article} {constraint} constraint.'
                )
            )

        invalid_vals = set(kwargs) - set(args)
        if invalid_vals:
            errors.append(
                ValueError(
                    f'Invalid values {invalid_vals} are present '
                    f'in {article} {constraint} constraint.'
                )
            )

        if errors:
            raise AggregateConstraintsError(errors)

    @classmethod
    def _validate_metadata_columns(cls, metadata, **kwargs):
        if 'column_name' in kwargs:
            column_names = [kwargs.get('column_name')]
        else:
            column_names = kwargs.get('column_names')

        missing_columns = set(column_names) - set(metadata.columns) - {None}
        if missing_columns:
            article = 'An' if cls.__name__ == 'Inequality' else 'A'
            raise ConstraintMetadataError(
                f'{article} {cls.__name__} constraint is being applied to invalid column names '
                f'{missing_columns}. The columns must exist in the table.'
            )

    @staticmethod
    def _validate_metadata_specific_to_constraint(metadata, **kwargs):
        pass

    @classmethod
    def _validate_metadata(cls, metadata, **kwargs):
        """Validate the metadata against the constraint.

        Args:
            metadata (sdv.metadata.SingleTableMetadata):
                Single table metadata instance.
            **kwargs (dict):
                Any required kwargs for the constraint.

        Raises:
            AggregateConstraintsError:
                All the errors from validating the metadata.
        """
        errors = []
        try:
            cls._validate_inputs(**kwargs)
        except AggregateConstraintsError as agg_error:
            errors.extend(agg_error.errors)

        try:
            cls._validate_metadata_columns(metadata, **kwargs)
        except Exception as e:
            errors.append(e)

        try:
            cls._validate_metadata_specific_to_constraint(metadata, **kwargs)
        except Exception as e:
            errors.append(e)

        if errors:
            raise AggregateConstraintsError(errors)

    def _validate_data_meets_constraint(self, table_data):
        """Make sure the given data is valid for the constraint.

        Args:
            data (pandas.DataFrame):
                Table data.

        Raises:
            ConstraintsNotMetError:
                If the table data is not valid for the provided constraints.
        """
        if set(self.constraint_columns).issubset(table_data.columns.to_numpy()):
            is_valid_data = self.is_valid(table_data)
            if not is_valid_data.all():
                constraint_data = table_data[list(self.constraint_columns)]
                invalid_rows = constraint_data[~is_valid_data]
                invalid_rows_str = _format_invalid_values_string(invalid_rows, 5)
                err_msg = (
                    f"Data is not valid for the '{self.__class__.__name__}' constraint:\n"
                    f'{invalid_rows_str}'
                )

                raise ConstraintsNotMetError(err_msg)

    def _fit(self, table_data):
        del table_data

    def fit(self, table_data):
        """Fit ``Constraint`` class to data.

        Args:
            table_data (pandas.DataFrame):
                Table data.
        """
        self._fit(table_data)
        self._validate_data_meets_constraint(table_data)

    def _transform(self, table_data):
        return table_data

    def transform(self, table_data):
        """Perform necessary transformations needed by constraint.

        Subclasses can optionally overwrite this method. If the transformation
        requires certain columns to be present in ``table_data``, then the subclass
        should overwrite the ``_transform`` method instead. This method raises a
        ``MissingConstraintColumnError`` if the ``table_data`` is missing any columns
        needed to do the transformation. If columns are present, this method will call
        the ``_transform`` method. If ``_transform`` fails, the data will be returned
        unchanged.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Input data unmodified.
        """
        table_data = table_data.copy()
        missing_columns = [col for col in self.constraint_columns if col not in table_data.columns]
        if missing_columns:
            raise MissingConstraintColumnError(missing_columns=missing_columns)

        return self._transform(table_data)

    def fit_transform(self, table_data):
        """Fit this Constraint to the data and then transform it.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        self.fit(table_data)
        return self.transform(table_data)

    def _reverse_transform(self, table_data):
        return table_data

    def reverse_transform(self, table_data):
        """Handle logic around reverse transforming constraints.

        If the ``transform`` method was skipped, then this method should be too.
        Otherwise attempt to reverse transform and if that fails, return the data
        unchanged to fall back on reject sampling.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Input data unmodified.
        """
        table_data = table_data.copy()
        return self._reverse_transform(table_data)

    def is_valid(self, table_data):
        """Say whether the given table rows are valid.

        This is a dummy version of the method that returns a series of ``True``
        values to avoid dropping any rows. This should be overwritten by all
        the subclasses that have a way to decide which rows are valid and which
        are not.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.Series:
                Series of ``True`` values
        """
        return pd.Series(True, index=table_data.index)

    def filter_valid(self, table_data):
        """Get only the rows that are valid.

        The filtering is done by calling the method ``is_valid``, which should
        be overwritten by subclasses, while this method should stay untouched.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Input data unmodified.
        """
        valid = self.is_valid(table_data)
        invalid = sum(~valid)
        if invalid:
            LOGGER.debug(
                '%s: %s invalid rows out of %s.', self.__class__.__name__, sum(~valid), len(valid)
            )

        if isinstance(valid, pd.Series):
            return table_data[valid.to_numpy()]

        return table_data[valid]

    @classmethod
    def _get_class_from_dict(cls, constraint_class):
        subclasses = get_subclasses(cls)
        if isinstance(constraint_class, str):
            if '.' in constraint_class:
                constraint_class = import_object(constraint_class)
            else:
                constraint_class = subclasses[constraint_class]

        return constraint_class

    @classmethod
    def from_dict(cls, constraint_dict):
        """Build a Constraint object from a dict.

        Args:
            constraint_dict (dict):
                Dict containing the keyword ``constraint_name`` alongside
                any additional arguments needed to create the instance.

        Returns:
            Constraint:
                New constraint instance.
        """
        constraint_class = constraint_dict.get('constraint_class')
        constraint_class = cls._get_class_from_dict(constraint_class)

        return constraint_class(**constraint_dict.get('constraint_parameters', {}))

    def to_dict(self):
        """Return a dict representation of this Constraint.

        The dictionary will contain the Qualified Name of the constraint
        class in the key ``constraint_name``, as well as any other arguments
        that were passed to the constructor when the instance was created.

        Returns:
            dict:
                Dict representation of this Constraint.
        """
        constraint_dict = {'constraint_class': _get_qualified_name(self.__class__)}

        constraint_parameters = {}
        for key, obj in copy.deepcopy(self.__kwargs__).items():
            if callable(obj) and _module_contains_callable_name(obj):
                constraint_parameters[key] = _get_qualified_name(obj)
            else:
                constraint_parameters[key] = obj

        constraint_dict['constraint_parameters'] = constraint_parameters
        return constraint_dict


class ColumnsModel:
    """ColumnsModel class.

    The ``ColumnsModel`` class enables the usage of conditional sampling when a column is a
    ``constraint_name``.
    """

    _columns_model = None

    def __init__(self, constraint, constraint_columns):
        if isinstance(constraint_columns, list):
            self.constraint_columns = constraint_columns
        else:
            self.constraint_columns = [constraint_columns]

        self.constraint = constraint

    @staticmethod
    def _get_hyper_transformer_config(data_to_model):
        sdtypes = {}
        transformers = {}
        for column_name, data in data_to_model.items():
            dtype = data.dropna().infer_objects().dtype.kind
            if dtype in ('i', 'f'):
                sdtypes[column_name] = 'numerical'
                transformers = FloatFormatter(
                    missing_value_replacement='mean', missing_value_generation='from_column'
                )
            elif dtype == 'O':
                sdtypes[column_name] = 'categorical'
                transformers[column_name] = OneHotEncoder
            elif dtype == 'M':
                sdtypes[column_name] = 'datetime'
                transformers[column_name] = UnixTimestampEncoder(
                    missing_value_replacement='mean', missing_value_generation='from_column'
                )
            elif dtype == 'b':
                sdtypes[column_name] = 'boolean'
                transformers[column_name] = BinaryEncoder(
                    missing_value_replacement=-1, missing_value_generation='from_column'
                )

        return {'sdtypes': sdtypes, 'transformers': transformers}

    def fit(self, table_data):
        """Fit the ``ColumnsModel``.

        Fit a ``GaussianUnivariate`` model to the ``self.constraint_column`` columns in the
        ``table_data`` in order to sample those columns when missing.

        Args:
            table_data (pandas.DataFrame):
                Table data.
        """
        data_to_model = table_data[self.constraint_columns]
        ht_config = self._get_hyper_transformer_config(data_to_model)

        self._hyper_transformer = HyperTransformer()
        self._hyper_transformer.set_config(ht_config)
        transformed_data = self._hyper_transformer.fit_transform(data_to_model)

        self._model = GaussianMultivariate(distribution=GaussianUnivariate)
        self._model.fit(transformed_data)

    def _reject_sample(self, num_rows, conditions):
        sampled = self._model.sample(num_rows=num_rows, conditions=conditions)
        sampled = self._hyper_transformer.reverse_transform(sampled)
        valid_rows = sampled[self.constraint.is_valid(sampled)]
        counter = 0
        total_sampled = num_rows

        while len(valid_rows) < num_rows:
            num_valid = len(valid_rows)
            if counter >= 100:
                if len(valid_rows) == 0:
                    raise ValueError('Could not get enough valid rows within 100 trials.')
                else:
                    multiplier = num_rows // num_valid
                    num_rows_missing = num_rows % num_valid
                    remainder_rows = valid_rows.iloc[0:num_rows_missing, :]
                    valid_rows = pd.concat(
                        [valid_rows] * multiplier + [remainder_rows], ignore_index=True
                    )
                    break

            remaining = num_rows - num_valid
            valid_probability = (num_valid + 1) / (total_sampled + 1)
            max_rows = num_rows * 10
            num_to_sample = min(int(remaining / valid_probability), max_rows)
            total_sampled += num_to_sample
            new_sampled = self._model.sample(num_rows=num_to_sample, conditions=conditions)
            new_sampled = self._hyper_transformer.reverse_transform(new_sampled)
            new_valid_rows = new_sampled[self.constraint.is_valid(new_sampled)]
            valid_rows = pd.concat([valid_rows, new_valid_rows], ignore_index=True)
            counter += 1

        return valid_rows.iloc[0:num_rows, :]

    def sample(self, table_data):
        """Sample any missing columns.

        Sample any missing columns, ``self.constraint_columns``, that ``table_data``
        does not contain.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Table data with additional ``constraint_columns``.
        """
        condition_columns = [c for c in self.constraint_columns if c in table_data.columns]
        grouped_conditions = table_data[condition_columns].groupby(_groupby_list(condition_columns))
        all_sampled_rows = []
        for group, dataframe in grouped_conditions:
            if not isinstance(group, tuple):
                group = [group]

            transformed_condition = self._hyper_transformer.transform(dataframe).iloc[0].to_dict()
            sampled_rows = self._reject_sample(
                num_rows=dataframe.shape[0], conditions=transformed_condition
            )
            all_sampled_rows.append(sampled_rows)

        sampled_data = pd.concat(all_sampled_rows, ignore_index=True)
        return sampled_data
