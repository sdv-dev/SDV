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
a particular column, ``salary`` for example, to be a multiple of a *base* value, e.g. 500. 
In other words, the ``salary`` increments by 500. We will define ``transform`` and 
``reverse_transform`` methods to make sure our data satisfy our constraints.

We can achieve our goal by performing transformations in a 2 step process:

- Divide ``salary`` by the base unit (500). This transformation makes it easier to model the data
  since it would now learn regular integer values without any explicit constraint on the data.
- Reversing the effect by multiplying ``salary`` back with the base unit. Now that the model has 
  learned regular integer values, we multiply it with the base (500) such that it now conforms to our original data range.


.. ipython:: python
    :okwarning:

    def transform(table_data):
        base = 500
        table_data['salary'] = table_data['salary'] / base
        return table_data


After defining ``transform`` we create ``reverse_transform`` that reverses the operations made.

.. ipython:: python
    :okwarning:

    def reverse_transform(table_data):
        base = 500
        table_data['salary'] = table_data['salary'].round() * base
        return table_data


Then, we pack every thing together in ``CustomConstraint`` and we can create our model 
and generate synthetic data by passing the constraint we just created.

.. ipython:: python
    :okwarning:

    constraint = CustomConstraint(
        transform=transform, 
        reverse_transform=reverse_transform
    )

    gc = GaussianCopula(constraints=[constraint])

    gc.fit(employees)

    sampled = gc.sample(10)

When we view the ``sampled`` data, we should find that all the rows in the sampled 
data have a salary that is a multiple of the base value.

.. ipython:: python
    :okwarning:

    sampled


Can I apply the same function to multiple columns?
--------------------------------------------------

Say we want ``salary`` and ``annual_bonus`` to be both composed of the base unit. Rather than 
defining two constraints, or editing the code of our functions for each new column that we want 
to constraint, we provide another style of writing functions such that the function should accept 
a column data as input.

The ``transform`` function takes ``column_data`` as input and returns the transformed column.


.. ipython:: python
    :okwarning:

    def transform(column_data):
        base = 500
        return column_data / base

Similarly we defined ``reverse_transform`` in a way that it operates on the data of a 
single column.

.. ipython:: python
    :okwarning:

    def reverse_transform(column_data):
        base = 500
        return column_data.round() * base

Now that we have our functions, we initialize ``CustomConstraint`` and we 
specify which column(s) are the desired ones.

.. ipython:: python
    :okwarning:

    constraint = CustomConstraint(
        columns=['salary', 'annual_bonus'],
        transform=transform, 
        reverse_transform=reverse_transform
    )

Now we create our model and pass our constraints.

.. ipython:: python
    :okwarning:

    gc = GaussianCopula(constraints=[constraint])

    gc.fit(employees)

    sampled = gc.sample(10)

Viewing ``sampled`` we now see two columns that are always a multiple of 500.

.. ipython:: python
    :okwarning:

    sampled


Can I access the rest of the table from my column functions?
------------------------------------------------------------

If we look closely at the data, we notice that ``salary`` and ``annual_bonus`` are only a 
multiple of 500 when the employee is not a "contractor". To take this requirement into 
consideration, we refer to a "fixed" column ``contractor`` in order to know whether we
should apply this constraint or not. The access to ``contractor`` column will allow us
to properly transform and reverse transform the data.

We write our functions to take as input:

-  ``table_data`` which contains all the information.
-  ``column`` which is a an argument to represent the columns of interest.

Now we can construct our functions freely, we write our methods
with said arguments and be able to access ``'contractor'``.

We first write our ``transform`` function as we have done previously:

.. ipython:: python
    :okwarning:

    def transform(table_data, column):
        base = 500
        display(table_data)
        table_data[column] = table_data[column] / base
        return table_data

When it comes to defining ``reverse_transform``, we need to distinguish between
contractors and non contractors, the operations are as follows:

1. round values to four decimal points for contractors such that the end result will 
   be two decimal points after multiplying the result with 500.
2. round values to zero for employees that are not contractors such that the end
   result will be a multiple of 500.

.. ipython:: python
    :okwarning:

    def reverse_transform(table_data, column):
        base = 500.
        is_not_contractor = table_data.contractor == 0.
        table_data[column] = table_data[column].round(4)
        table_data.at[is_not_contractor, column] = table_data[column].loc[is_not_contractor].round()
        table_data[column] *= base
        return table_data

We now stich everything together and pass it to the model.

.. ipython:: python
    :okwarning:

    constraint = CustomConstraint(
        columns=['salary', 'annual_bonus'],
        transform=transform, 
        reverse_transform=reverse_transform
    )

    gc = GaussianCopula(constraints=[constraint])

    gc.fit(employees)

    sampled = gc.sample(10)

    sampled

This style gives flexibility to access any column in the table while still operating on 
a column basis.


Can I write a ``CustomConstraint`` based on reject sampling?
------------------------------------------------------------

In the previous section, we defined our ``CustomConstraint`` using ``transform`` and 
``reverse_transform`` functions. Sometimes, our constraints are not possible to implement 
using these methods, that is when we rely on ``reject_sampling`` strategy. In ``reject_sampling`` 
we need to implement an ``is_valid`` function that identifies which rows do not follow the 
said constraint, in our case, which rows are not a multiple of the *base* unit.

We can define ``is_valid`` according to the three styles mentioned in the previous section:

1. function with ``table_data`` argument.
2. function with ``column_data`` argument.
3. function with ``table_data`` and ``column`` argument.

Here is an example of how you would define ``is_valid`` for each one of the mentioned styles:

.. code-block:: python

    def is_valid(table_data):
        base = 500
        return table_data['salary'] % base == 0

    def is_valid(column_data):
        base = 500
        return column_data % base == 0

    def is_valid(table_data, column):
        base = 500
        is_contractor = table_data.contractor == 1
        valid = table_data[column] % base == 0
        contractor_salary = employees['salary'].loc[is_contractor]
        valid.loc[is_contractor] = contractor_salary == contractor_salary.round(2)
        return valid

Then we construct ``CustomConstraint`` to take ``is_valid`` on its own.

.. code-block:: python

    constraint = CustomConstraint(
        columns=['salary', 'annual_bonus'],
        is_valid=is_valid
    )

