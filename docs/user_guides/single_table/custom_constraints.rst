.. _custom_constraints:

Customize your Constraints
==========================

In some cases, you would like to construct your own constraint. You can
use ``CustomConstraint`` to define your own logic on how you would like to
apply it to your data. There are three main functions that you can create:

- ``transform`` which is responsible for the forward pass when using ``transform`` strategy.
- ``reverse_transform`` which defines how to reverse the transformation of the ``transform``.
- ``is_valid`` which computes the invalid rows.

Let's look at a demo dataset:

.. ipython:: python
    :okwarning:

    from sdv.demo import load_tabular_demo

    employees = load_tabular_demo()
    employees

The dataset defined in :ref:`_single_table_constraints`contains basic details about employees.
We will use this dataset to demonstrate how you can create your own constraint. 


Custom Constraints
------------------

Let's consider the following example where we wish to generate synthetic data and we want
a particular column, age for example, to be even. We will define ``transform``, 
``reverse_transform``, and ``is_valid`` methods to make our data satisfy our constraints.


In this example, we would like to create a constraint that makes ``age`` even. We can
create a ``transform_even`` function that makes all the ``age`` values even.

.. ipython:: python
    :okwarning:

    def transform_even(table_data):
        table_data['age'] = table_data['age'] * 2
        return table_data

We transform the column ``age`` to be even by multiplicating with 2. When now define
the ``reverse_transform_even`` as doing the opposite effect.

.. ipython:: python
    :okwarning:

    def reverse_transform_even(table_data):
        table_data['age'] = table_data['age'] / 2
        return table_data

Lastly, we write the ``is_valid`` function to assess whether the column is even or not.

.. ipython:: python
    :okwarning:

    def is_even(table_data):
        return table_data['age'] % 2 == 0

We put every thing together in ``CustomConstraint`` and then we can create our model and 
generate synthetic data by passing the constraint we just created.

.. ipython:: python
    :okwarning:

    constraint = CustomConstraint(
        transform=transform_even, 
        reverse_transform=reverse_transform_even, 
        is_valid=is_even
    )

    gc = GaussianCopula(constraints=[constraint])

    gc.fit(employees)

    sampled = gc.sample(10)

When we view the ``sampled`` data, we should find that all the rows in the sampled data have an
even age value.

.. ipython:: python
    :okwarning:

    sampled


.. note::
    If you are using ``reject_sampling`` strategy, it is sufficient to define
    ``is_valid`` alone. For example, ``CustomConstraint(is_valid=is_even)``.


Applying a Constraint to Multiple Columns
-----------------------------------------

Column Based
~~~~~~~~~~~~

What if I want to apply the even constraint to multiple columns? Say we want ``age``
and ``age_when_joined`` to be both even. Rather than defining two constraints,
we provide another style of writing functions such that the function should accept 
a column data as input.

The ``transform_even`` function takes ``column_data`` and input and returns the transformed
column.

.. code-block:: python

    def transform_even(column_data):
        return column_data * 2

Similarly we defined ``reverse_transform_even`` and ``is_even`` in a way that it operates
on the data of a single column.

.. code-block:: python

    def reverse_transform_even(column_data):
        return column_data / 2


    def is_even(column_data):
        return column_data % 2 == 0

Now that we have our functions, we initialize ``CustomConstraint`` and we 
specify which column(s) are the desired ones.

.. code-block:: python

    constraint = CustomConstraint(
        columns=['age', 'age_when_joined'],
        transform=transform_even, 
        reverse_transform=reverse_transform_even, 
        is_valid=is_even
    )


Table Based
~~~~~~~~~~~

In some cases, we need to design our methods to be applied to a slice of the data but 
with access to the entire table. For example, we wish to make sure that ``age`` and 
``age_when_joined`` are always larger than a "fixed" column ``years_in_the_company``.

To support this requirement, we write functions takes as input ``table_data`` and ``column``. 
In this case, we define our function as

.. code-block:: python

    def is_larger(table_data, column):
    	return table_data[column] > table_data['years_in_the_company']

    constraint = CustomConstraint(columns=['age', 'age_when_joined'], is_valid=is_larger)

This style gives flexibility to access any column in the table while still operating on 
a column basis.

.. note::
	The ``transform`` and ``reverse_transform`` methods return a table. Except
	when operating on ``column_data`` then it returns the transformed or 
	reverse transformed column.

