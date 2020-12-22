"""Base Constraint class."""

import copy
import importlib
import inspect
import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)


def _get_qualified_name(obj):
    """Return the Fully Qualified Name from an instance or class."""
    module = obj.__module__
    if hasattr(obj, '__name__'):
        obj_name = obj.__name__
    else:
        obj_name = obj.__class__.__name__

    return module + '.' + obj_name


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
    """

    def _identity(self, table_data):
        return table_data

    def __init__(self, handling_strategy):
        if handling_strategy == 'transform':
            self.filter_valid = self._identity
        elif handling_strategy == 'reject_sampling':
            self.transform = self._identity
            self.reverse_transform = self._identity
        elif handling_strategy != 'all':
            raise ValueError('Unknown handling strategy: {}'.format(handling_strategy))

    def fit(self, table_data):
        """No-op method written for completion. To be optionally overwritten by subclasses.

        Args:
            table_data (pandas.DataFrame):
                Table data.
        """
        del table_data

    def transform(self, table_data):
        """Identity method for completion. To be optionally overwritten by subclasses.

        Args:
            table_data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Input data unmodified.
        """
        return table_data

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
            if not callable(obj):
                constraint_dict[key] = obj
            else:
                constraint_dict[key] = _get_qualified_name(obj)

        return constraint_dict
