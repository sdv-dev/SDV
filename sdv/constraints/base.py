"""Base Constraint class."""

import copy
import importlib
import inspect
import logging

import pandas as pd
from copulas.multivariate.gaussian import GaussianMultivariate
from rdt import HyperTransformer

from sdv.constraints.errors import MissingConstraintColumnError

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
    subclasses = dict()
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

    def __init__(self, name, bases, attr):
        super().__init__(name, bases, attr)

        old__init__ = self.__init__
        signature = inspect.signature(old__init__)
        arg_names = list(signature.parameters.keys())[1:]

        def __init__(self, *args, **kwargs):
            class_name = self.__class__.__name__
            if name == class_name:
                self.__kwargs__ = copy.deepcopy(kwargs)
                self.__kwargs__.update(dict(zip(arg_names, args)))

            old__init__(self, *args, **kwargs)

        __init__.__doc__ = old__init__.__doc__
        __init__.__signature__ = signature
        self.__init__ = __init__


class Constraint(metaclass=ConstraintMeta):
    """Constraint base class.

    This class is not intended to be used directly and should rather be
    subclassed to create different types of constraints.

    If ``handling_strategy`` is passed with the value ``transform``
    or ``reject_sampling``, the ``filter_valid`` or ``transform`` and
    ``reverse_transform`` methods will be replaced respectively by a simple
    identity function.

    Args:
        handling_strategy (str):
            How this Constraint should be handled, which can be ``transform``,
            ``reject_sampling`` or ``all``.
        fit_columns_model (bool):
            If False, reject sampling will be used to handle conditional sampling.
            Otherwise, a model will be trained and used to sample other columns
            based on the conditioned column.
    """

    constraint_columns = ()
    _hyper_transformer = None
    _columns_model = None

    def _identity(self, table_data):
        return table_data

    def __init__(self, handling_strategy, fit_columns_model=True):
        self.fit_columns_model = fit_columns_model
        if handling_strategy == 'transform':
            self.filter_valid = self._identity
        elif handling_strategy == 'reject_sampling':
            self.transform = self._identity
            self.reverse_transform = self._identity
        elif handling_strategy != 'all':
            raise ValueError('Unknown handling strategy: {}'.format(handling_strategy))

    def _fit(self, table_data):
        del table_data

    def fit(self, table_data):
        """Fit ``Constraint`` class to data.

        If ``fit_columns_model`` is True, then this method will fit
        a ``GaussianCopula`` model to the relevant columns in ``table_data``.
        Subclasses can overwrite this method, or overwrite the ``_fit`` method
        if they will not be needing the model to handle conditional sampling.

        Args:
            table_data (pandas.DataFrame):
                Table data.
        """
        self._fit(table_data)

        if self.fit_columns_model and len(self.constraint_columns) > 1:
            data_to_model = table_data[list(self.constraint_columns)]
            self._hyper_transformer = HyperTransformer(dtype_transformers={
                'O': 'one_hot_encoding',
            })
            transformed_data = self._hyper_transformer.fit_transform(data_to_model)
            self._columns_model = GaussianMultivariate()
            self._columns_model.fit(transformed_data)

    def _transform(self, table_data):
        return table_data

    def _reject_sample(self, num_rows, conditions):
        sampled = self._columns_model.sample(
            num_rows=num_rows,
            conditions=conditions
        )
        sampled = self._hyper_transformer.reverse_transform(sampled)
        valid_rows = sampled[self.is_valid(sampled)]
        counter = 0
        total_sampled = num_rows

        while len(valid_rows) < num_rows:
            num_valid = len(valid_rows)
            if counter >= 100:
                if len(valid_rows) == 0:
                    error = 'Could not get enough valid rows within 100 trials.'
                    raise ValueError(error)
                else:
                    multiplier = num_rows // num_valid
                    num_rows_missing = num_rows % num_valid
                    remainder_rows = valid_rows.iloc[0:num_rows_missing, :]
                    valid_rows = pd.concat([valid_rows] * multiplier + [remainder_rows],
                                           ignore_index=True)
                    break

            remaining = num_rows - num_valid
            valid_probability = (num_valid + 1) / (total_sampled + 1)
            max_rows = num_rows * 10
            num_to_sample = min(int(remaining / valid_probability), max_rows)
            total_sampled += num_to_sample
            new_sampled = self._columns_model.sample(
                num_rows=num_to_sample,
                conditions=conditions
            )
            new_sampled = self._hyper_transformer.reverse_transform(new_sampled)
            new_valid_rows = new_sampled[self.is_valid(new_sampled)]
            valid_rows = pd.concat([valid_rows, new_valid_rows], ignore_index=True)
            counter += 1

        return valid_rows.iloc[0:num_rows, :]

    def _sample_constraint_columns(self, table_data):
        condition_columns = [c for c in self.constraint_columns if c in table_data.columns]
        grouped_conditions = table_data[condition_columns].groupby(condition_columns)
        all_sampled_rows = list()
        for group, df in grouped_conditions:
            if not isinstance(group, tuple):
                group = [group]

            transformed_condition = self._hyper_transformer.transform(df).iloc[0].to_dict()
            sampled_rows = self._reject_sample(
                num_rows=df.shape[0],
                conditions=transformed_condition
            )
            all_sampled_rows.append(sampled_rows)

        sampled_data = pd.concat(all_sampled_rows, ignore_index=True)
        return sampled_data

    def _validate_constraint_columns(self, table_data):
        """Validate the columns in ``table_data``.

        If ``fit_columns_model`` is False and any columns in ``constraint_columns``
        are not present in ``table_data``, this method will raise a
        ``MissingConstraintColumnError``. Otherwise it will return the ``table_data``
        unchanged. If ``fit_columns_model`` is True, then this method will sample
        any missing ``constraint_columns`` from its model conditioned on the
        ``constraint_columns`` that ``table_data`` does contain. If ``table_data``
        doesn't contain any of the ``constraint_columns`` then a
        ``MissingConstraintColumnError`` will be raised.

        Args:
            table_data (pandas.DataFrame):
                Table data.
        """
        missing_columns = [col for col in self.constraint_columns if col not in table_data.columns]
        if missing_columns:
            all_columns_missing = len(missing_columns) == len(self.constraint_columns)
            if self._columns_model is None or all_columns_missing:
                raise MissingConstraintColumnError()

            else:
                sampled_data = self._sample_constraint_columns(table_data)
                other_columns = [c for c in table_data.columns if c not in self.constraint_columns]
                sampled_data[other_columns] = table_data[other_columns]
                return sampled_data

        return table_data

    def transform(self, table_data):
        """Perform necessary transformations needed by constraint.

        Subclasses can optionally overwrite this method. If the transformation
        requires certain columns to be present in ``table_data``, then the subclass
        should overwrite the ``_transform`` method instead. This method raises a
        ``MissingConstraintColumnError`` if the ``table_data`` is missing any columns
        needed to do the transformation. If columns are present, this method will call
        the ``_transform`` method.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Input data unmodified.
        """
        table_data = self._validate_constraint_columns(table_data)
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

    def reverse_transform(self, table_data):
        """Identity method for completion. To be optionally overwritten by subclasses.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Input data unmodified.
        """
        return table_data

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
            LOGGER.debug('%s: %s invalid rows out of %s.',
                         self.__class__.__name__, sum(~valid), len(valid))

        return table_data[valid]

    @classmethod
    def from_dict(cls, constraint_dict):
        """Build a Constraint object from a dict.

        Args:
            constraint_dict (dict):
                Dict containing the keyword ``constraint`` alongside
                any additional arguments needed to create the instance.

        Returns:
            Constraint:
                New constraint instance.
        """
        constraint_dict = constraint_dict.copy()
        constraint_class = constraint_dict.pop('constraint')
        subclasses = get_subclasses(cls)
        if isinstance(constraint_class, str):
            if '.' in constraint_class:
                constraint_class = import_object(constraint_class)
            else:
                constraint_class = subclasses[constraint_class]

        return constraint_class(**constraint_dict)

    def to_dict(self):
        """Return a dict representation of this Constraint.

        The dictionary will contain the Qualified Name of the constraint
        class in the key ``constraint``, as well as any other arguments
        that were passed to the constructor when the instance was created.

        Returns:
            dict:
                Dict representation of this Constraint.
        """
        constraint_dict = {
            'constraint': _get_qualified_name(self.__class__),
        }

        for key, obj in copy.deepcopy(self.__kwargs__).items():
            if callable(obj) and _module_contains_callable_name(obj):
                constraint_dict[key] = _get_qualified_name(obj)
            else:
                constraint_dict[key] = obj

        return constraint_dict
