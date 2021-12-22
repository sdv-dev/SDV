.. _relational_constraints:

Constraints
===========

SDV supports adding constraints within a single table. See :ref:`single_table_constraints`
for more information about the available single table constraints.

In order to use single-table constraints within a relational model, you can pass
in a list of applicable constraints when adding a table to your relational ``Metadata``.
(See :ref:`relational_metadata` for more information on constructing a ``Metadata`` object.)

In this example, we wish to add a ``UniqueCombinations`` constraint to our ``sessions`` table,
which is a child table of ``users``. First, we will create a ``Metadata`` object and add the
``users`` table.

.. ipython:: python
    :okwarning:
   
    from sdv import load_demo, Metadata

    tables = load_demo()

    metadata = Metadata()

    metadata.add_table(
        name='users',
        data=tables['users'],
        primary_key='user_id'
    )

The metadata now contains the ``users`` table.

.. ipython:: python
    :okwarning:

    metadata

Now, we want to add a child table ``sessions`` which contains a single table constraint.
In the ``sessions`` table, we wish to only have combinations of ``(device, os)`` that 
appear in the original data.

.. ipython:: python
    :okwarning:

    from sdv.constraints import UniqueCombinations

    constraint = UniqueCombinations(columns=['device', 'os'])

    metadata.add_table(
        name='sessions',
        data=tables['sessions'],
        primary_key='session_id',
        parent='users',
        foreign_key='user_id',
        constraints=[constraint],
    )

If we get the table metadata for ``sessions``, we can see that the constraint has been added.

.. ipython:: python
    :okwarning:

    metadata.get_table_meta('sessions')

We can now use this metadata to fit a relational model and synthesize data.

.. ipython:: python
    :okwarning:

    from sdv.relational import HMA1

    model = HMA1(metadata)
    model.fit(tables)
    new_data = model.sample()

In the sampled data, we should see that our constraint is being satisfied.

.. ipython:: python
    :okwarning:

    new_data
