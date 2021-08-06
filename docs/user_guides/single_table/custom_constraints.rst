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

Let's consider the following example where we wish to generate synthetic data and we want
a particular column, age for example, to be even. We will define ``transform``, 
``reverse_transform``, and ``is_valid`` methods to make our data satisfy our constraints.
We can achieve our goal by doing a 2 step process: 

- dividing the input data by 2
- multiplying the sampled data by 2

We can create a ``transform`` function that makes all the ``age`` values even.

.. ipython:: python
    :okwarning:

    def transform(table_data):
        table_data['age'] = table_data['age'] / 2
        return table_data

We transform the column ``age`` to be even by dividing it by 2. We now define
the ``reverse_transform`` as doing the opposite effect.

.. ipython:: python
    :okwarning:

    def reverse_transform(table_data):
        table_data['age'] = table_data['age'] * 2
        return table_data

Lastly, we write the ``is_valid`` function to assess whether the column is even or not.

.. ipython:: python
    :okwarning:

    def is_valid(table_data):
        return table_data['age'] % 2 == 0

We put every thing together in ``CustomConstraint`` and then we can create our model and 
generate synthetic data by passing the constraint we just created.

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

When we view the ``sampled`` data, we should find that all the rows in the sampled data have an
even age value.

.. ipython:: python
    :okwarning:

    sampled


.. note::
    It is sufficient to define ``is_valid`` function alone. In this case, the constraint will
    use ``reject_sampling`` strategy. For example, ``CustomConstraint(is_valid=is_valid)``.


Can I apply the same function to multiple columns?
--------------------------------------------------

Say we want ``age`` and ``age_when_joined`` to be both even. Rather than defining 
two constraints, or editing the code of our functions for each new column that we 
want to constraint, we provide another style of writing functions such that the 
function should accept a column data as input.

The ``transform`` function takes ``column_data`` as input and returns the transformed
column.

.. ipython:: python
    :okwarning:

    def transform(column_data):
        return column_data / 2

Similarly we defined ``reverse_transform`` and ``is_valid`` in a way that it operates
on the data of a single column.

.. ipython:: python
    :okwarning:

    def reverse_transform(column_data):
        return column_data * 2


    def is_valid(column_data):
        return column_data % 2 == 0

Now that we have our functions, we initialize ``CustomConstraint`` and we 
specify which column(s) are the desired ones.

.. ipython:: python
    :okwarning:

    constraint = CustomConstraint(
        columns=['age', 'age_when_joined'],
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

Viewing ``sampled`` we now see two columns that are always even.

.. ipython:: python
    :okwarning:

    sampled


Can I access the rest of the table from my column functions?
------------------------------------------------------------

In addition to wanting to construct even columns, we would like ``age`` and
``age_when_joined`` to be always larger than a "fixed" column ``years_in_the_company``.

To support this requirement, we write functions that take as input:

-  ``table_data`` which contains all the information.
-  ``column`` which is a an argument to represent the columns of interest.

Now we can construct our functions freely, we write our methods
with said arguments and be able to access ``'years_in_the_company'``.

We first write our ``transform`` function to:

1. add a value of ``'years_in_the_company'``.
2. multiply the result with 2 to make sure it is even.

.. ipython:: python
    :okwarning:

    def transform(table_data, column):
        added_value = table_data[column] + table_data['years_in_the_company']
        table_data[column] = added_value / 2
        return table_data

Now we define our ``reverse_transform`` to reverse the operations performed
in the ``transform``.

.. ipython:: python
    :okwarning:

    def reverse_transform(table_data, column):
        value = table_data[column] - table_data['years_in_the_company']
        table_data[column] *= 2 
        return table_data

Lastly, we write our ``is_valid`` function to identify invalid rows.

.. ipython:: python
    :okwarning:

    def is_valid(table_data, column):
        is_larger = table_data[column] > table_data['years_in_the_company']
        is_even = table_data[column] % 2 == 0
        return is_larger & is_even

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

