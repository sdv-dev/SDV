.. _custom_constraints:

Defining Custom Constraints
===========================

In some cases, the predefined constraints do not cover all your needs. 
In such scenarios, you can use ``CustomConstraint`` to define your own 
logic on how you would like to apply it to your data. There are three 
main functions that you can create:

- ``transform`` which is responsible for the forward pass when using ``transform`` strategy.
- ``reverse_transform`` which defines how to reverse the transformation of the ``transform``.
- ``is_valid`` which indicates which rows satisfy the constraint and which ones do not.

Let's look at a demo dataset:

.. ipython:: python
    :okwarning:

    from sdv.demo import load_tabular_demo

    employees = load_tabular_demo()
    employees

The dataset defined in :ref:`_single_table_constraints` contains basic details about employees.
We will use this dataset to demonstrate how you can create your own constraint. 


Using the ``CustomConstraint``
------------------------------

Let's consider the following example where we wish to generate synthetic data and 
we want a particular column, ``annual_bonus`` for example, to be a multiple of a 
*base* value, e.g. 150. In other words, the ``annual_bonus`` increments by 150. 
We will define ``transform``, ``reverse_transform``, and ``is_valid`` methods to 
make our data satisfy our constraints.

We can achieve our goal by performing transformations in a 2 step process:

- divide ``annual_bonus`` by the base unit (150).
- reversing the effect by multiplying it back with the base unit (150).


.. ipython:: python
    :okwarning:

    def transform(table_data):
        base = 150
        table_data['annual_bonus'] = table_data['annual_bonus'] / base
        return table_data


After defining ``transform`` we create ``reverse_transform`` that reverses
the operations made.

.. ipython:: python
    :okwarning:

    def reverse_transform(table_data):
        base = 150
        table_data['annual_bonus'] *= base
        return table_data


Lastly, we write the ``is_valid`` function to assess whether the column is a 
multiple of the base or not, we know this through two factors: if the quotient 
is larger than zero, as well as not containing any decimal values.

.. ipython:: python
    :okwarning:

    def is_valid(table_data):
        base = 150
        quotient = table_data['annual_bonus'] / base
        is_dividable = quotient > 0
        is_int = quotient.apply(float.is_integer)
        return is_dividable & is_int


We put every thing together in ``CustomConstraint`` and then we can create our
model and generate synthetic data by passing the constraint we just created.

.. ipython:: python
    :okwarning:

    from sdv.constraints import CustomConstraint
    from sdv.tabular import GaussianCopula

    constraint = CustomConstraint(
        transform=transform, 
        reverse_transform=reverse_transform, 
        is_valid=is_valid
    )

    gc = GaussianCopula(constraints=[constraint])

    gc.fit(employees)

    sampled = gc.sample(10)

When we view the ``sampled`` data, we should find that all the rows in the sampled 
data have an annual bonus that is a multiple of the base value.

.. ipython:: python
    :okwarning:

    sampled


.. note::
    It is sufficient to define ``is_valid`` function alone. In this case, the constraint will
    use the ``reject_sampling`` strategy. For example, ``CustomConstraint(is_valid=is_valid)``.


Can I apply the same function to multiple columns?
--------------------------------------------------

Say we want ``annual_bonus`` and ``salary`` to be both composed of the base unit. 
Rather than defining two constraints, or editing the code of our functions for each 
new column that we want to constraint, we provide another style of writing functions 
such that the function should accept a column data as input.

The ``transform`` function takes ``column_data`` as input and returns the transformed column.


.. ipython:: python
    :okwarning:

    def transform(column_data):
        base = 150
        return column_data / base

Similarly we defined ``reverse_transform`` and ``is_valid`` in a way that it operates
on the data of a single column.

.. ipython:: python
    :okwarning:

    def reverse_transform(column_data):
        base = 150
        return column_data * base


    def is_valid(column_data):
        base = 150
        quotient = column_data / base
        is_dividable = quotient > 0
        is_int = quotient.apply(float.is_integer)
        return is_dividable & is_int

Now that we have our functions, we initialize ``CustomConstraint`` and we 
specify which column(s) are the desired ones.

.. ipython:: python
    :okwarning:

    constraint = CustomConstraint(
        columns=['annual_bonus', 'salary'],
        transform=transform, 
        reverse_transform=reverse_transform, 
        is_valid=is_valid
    )

Now we create our model and pass our constraints.

.. ipython:: python
    :okwarning:

    gc = GaussianCopula(constraints=[constraint])

    gc.fit(employees)

    sampled = gc.sample(10)

Viewing ``sampled`` we now see two columns that are always a multiple of 150.

.. ipython:: python
    :okwarning:

    sampled


Can I access the rest of the table from my column functions?
------------------------------------------------------------

In addition to wanting to construct values that are a multiple of a base unit,
we would like ``annual_bonus`` and ``salary`` to be based of a "fixed" column 
``years_in_the_company`` such that every record should be receiving an annual 
bonus or salary that is at least a thousand in value, which we call a minimum 
value. This minimum value doubles each year.

To support this requirement, we write functions that take as input:

-  ``table_data`` which contains all the information.
-  ``column`` which is a an argument to represent the columns of interest.

Now we can construct our functions freely, we write our methods
with said arguments and be able to access ``'years_in_the_company'``.

We first write our ``transform`` function:

.. ipython:: python
    :okwarning:

    def transform(table_data, column):
        base = 150
        table_data[column] = table_data[column] / base
        return table_data

Now we define our ``reverse_transform`` to reverse the operations performed
in the ``transform``.

.. ipython:: python
    :okwarning:

    def reverse_transform(table_data, column):
        base = 150
        table_data[column] *= base 
        return table_data

Lastly, we write our ``is_valid`` function to identify invalid rows.

.. ipython:: python
    :okwarning:

    def is_valid(table_data, column):
        base = 150
        minimum = 1000
        quotient = table_data[column] / base
        is_dividable = quotient > 0
        is_int = quotient.apply(float.is_integer)
        is_larger = table_data[column] > (table_data['years_in_the_company'] * minimum)
        return is_dividable & is_int & is_larger

We now stich everything together and pass it to the model.

.. ipython:: python
    :okwarning:

    constraint = CustomConstraint(
        columns=['age', 'age_when_joined'],
        transform=transform, 
        reverse_transform=reverse_transform, 
        is_valid=is_valid
    )

    gc = GaussianCopula(constraints=[constraint])

    gc.fit(employees)

    sampled = gc.sample(10)

    sampled

This style gives flexibility to access any column in the table while still operating on 
a column basis.

