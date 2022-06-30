.. _developer_constraints:

Constraints
===========

The Constraints are implemented in the sub-package :ref:`sdv.constraints`.

Base Constraint
---------------

All the Constraints in SDV inherit from the ``sdv.constraint.base.Constraint`` class.

The ``Constraint`` defines the public API for all its subclasses. The public API is implemented
in some cases using no-op (methods that do nothing) or identity (methods that return what they
are given) methods, so subclasses can overwrite only what they need but still have a complete
API which is compatible with the rest of the project.

The following public methods are implemented in this class:

* ``fit``: No-op method.
* ``transform``: Identity method.
* ``fit_transform``: Call ``self.fit`` and then call ``self.transform`` and return its outputs.
* ``reverse_transform``: Identity method.
* ``is_valid``: Return a ``pandas.Series`` full or ``True`` values with the same length as the
  given data.
* ``filter_valid``: Return only the rows for which ``self.is_valid`` returns ``True``.
* ``from_dict``: Build a ``Constraint`` from its dict representation.
* ``to_dict``: Return a dict representing the ``Constraint``.

Implementing a Custom Constraint
--------------------------------

In order to implement a custom constraint, all you need to do is create a subclass of
``Constraint`` and implement your own ``fit``, ``transform``, ``reverse_transform`` and
``is_valid`` methods.

Let us think, for example, of the following scenario: Suppose we have a dataset about invertebrate
and that there is a column that indicates their number of legs. Insects, which are one of the most
common invertebrate, always have 6 legs, but there are other families which have none, or just 2,
and some even extend to hundreds of legs. But this value has the following properties:

* It is always positive.
* It is always an even number.

Expecting our tabular models to learn such properties on their own is very hard, especially
regarding the even numbers, so we will try to help the models by defining a custom Constraint
called ``PositiveEvenInteger``, which inherits from ``Constraint``.

.. code-block:: python

    class PositiveEvenInteger(Constraint):
        """Ensure that values are positive and even."""

        pass

The simplest way to ensure that the values have the desired properties is to validate them,
so let's define the ``is_valid`` method accordingly:

.. code-block:: python

    class PositiveEven(Constraint):
        """Ensure that values are positive and even."""

        def __init__(self, column_name):
            self._column_name = column_name

        def is_valid(self, table_data):
            """Say if values are positive and even."""
            column_data = table_data[self._column_name]
            positive = column_data >= 0
            even = column_data.mod(2) == 0

            return positive & even

.. note:: Notice how we also had to add a ``column_name`` argument to our ``__init__`` method,
          so we know which column we need to validate.

With the current implementation modeling would happen as usual. However, during sampling,
all the rows would be validated using the ``is_valid`` method that we implemented, and invalid
rows would be rejected and re-sampled until the number of desired rows has been generated.

In this case this might be acceptable because each row only has a 50% chance of being invalid,
which means that, on average, we would need the model to sample only 2 times the number of rows
that we need in order to get enough valid rows. However, in some other cases this can take a long
time, especially if the condition imposed has a very low chance of being true. In such cases, we
might want to use a transformation strategy where the data is transformed before modeling into
something that the model can learn more easily, and then reverted after sampling back into the
original format.

For our dataset, a possibility would be to divide the number of legs by two, so we end up
modeling and sampling the number of `pairs of legs` instead of the number of `legs`:

.. code-block:: python

    class PositiveEven(Constraint):
        """Ensure that values are positive and even."""

        def __init__(self, column_name):
            self._column_name = column_name

        def is_valid(self, table_data):
            """Say if values are positive and even."""
            column_data = table_data[self._column_name]
            positive = column_data >= 0
            even = column_data.mod(2) == 0

            return positive & even

        def transform(self, table_data):
            """Divide the data by two before modeling."""
            table_data[self._column_name] = table_data[self._column_name] / 2
            return table_data

        def reverse_transform(self, table_data):
            """Multiply the data by two after sampling."""
            table_data[self._column_name] = table_data[self._column_name] * 2
            return table_data

.. code-block:: python

    class PositiveEven(Constraint):
        """Ensure that values are positive and even."""

        def __init__(self, column_name):
            self._column_name = column_name

        def is_valid(self, table_data):
            """Say if values are positive and even."""
            column_data = table_data[self._column_name]
            positive = column_data >= 0
            even = column_data.mod(2) == 0

            return positive & even

        def transform(self, table_data):
            """Divide the data by two before modeling."""
            table_data[self._column_name] = table_data[self._column_name] / 2
            return table_data

        def reverse_transform(self, table_data):
            """Multiply the data by two after sampling."""
            table_data[self._column_name] = table_data[self._column_name] * 2
            return table_data
