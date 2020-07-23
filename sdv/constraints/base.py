"""Base Constraint class."""

import copy
import importlib
import inspect
import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)


def get_qualified_name(_object):
    """Return the Fully Qualified Name from an instance or class."""
    module = _object.__module__
    if hasattr(_object, '__name__'):
        _class = _object.__name__

    else:
        _class = _object.__class__.__name__

    return module + '.' + _class


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
    """Constraint base class."""

    _handling_strategy = 'all'

    def _identity(self, table_data):
        return table_data

    def __init__(self):
        if self._handling_strategy == 'transform':
            self.filter_valid = self._identity
        elif self._handling_strategy == 'reject_sampling':
            self.transform = self._identity
            self.reverse_transform = self._identity

    def fit(self, table_data):
        """No-op method."""
        pass

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
    def _get_subclasses(cls):
        """Recursively find subclasses for the current class object."""
        subclasses = dict()
        for subclass in cls.__subclasses__():
            subclasses[subclass.__name__] = subclass
            subclasses.update(subclass._get_subclasses())

        return subclasses

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
        subclasses = cls._get_subclasses()
        if isinstance(constraint_class, str):
            if '.' in constraint_class:
                constraint_class = import_object(constraint_class)
            else:
                constraint_class = subclasses[constraint_class]

        return constraint_class(**constraint_dict)

    def to_dict(self):
        """Return a dict representation of this Constraint."""
        constraint_dict = {
            'constraint': get_qualified_name(self.__class__),
        }

        for key, obj in copy.deepcopy(self.__kwargs__).items():
            if not callable(obj):
                constraint_dict[key] = obj
            else:
                constraint_dict[key] = get_qualified_name(obj)

        return constraint_dict
