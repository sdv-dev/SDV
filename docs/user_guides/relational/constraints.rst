.. _relational_constraints:

Constraints
===========

SDV supports adding constraints within a single table. See :ref:`single_table_constraints`
for more information about the available single table constraints.

In order to use single-table constraints within a relational model, you can pass
in a list of applicable constraints when adding a table to your relational ``Metadata``.
(See :ref:`relational_metadata` for more information on constructing a ``Metadata`` object.)

In this example, we wish to add a ``FixedCombinations`` constraint to our ``sessions`` table,
which is a child table of ``users``. First, we will create a ``Metadata`` object and add the
``users`` table.
