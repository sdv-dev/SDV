.. _custom_constraints:

Customize your Constraints
==========================

In some cases, you would like to construct your own constraint. You can
use ``CustomConstraint`` to define your own logic on how you would like to
apply it to your data. There are three main functions that you can create:

- ``transform`` which is responsible for the forward pass when using ``transform`` strategy.
- ``reverse_transform`` which defines how to reverse the transformation of the ``transform``.
- ``is_valid`` which computes the invalid rows.

Let's look at a demo dataset defined in :ref:`_single_table_constraints`:

.. ipython:: python
    :okwarning:

    from sdv.demo import load_tabular_demo

    employees = load_tabular_demo()
    employees

Let's consider the following example where we wish to generate synthetic data and we want
a particular column (age) to be positive. Then we can create `is_positive` as a validation
function to use in ``reject_sampling`` strategy and only sample records with even dates.

.. ipython:: python
    :okwarning:

    from sdv.constraints import CustomConstraint

    def is_positive(table_data):
    	return table_data['age'] > 0

    constraint = CustomConstraint(is_valid=is_positive)

Then we can create our model and generate synthetic data by passing the constraint
we just created.

.. ipython:: python
    :okwarning:

    from sdv.tabular import GaussianCopula

    gc = GaussianCopula(constraints=[constraint])

    gc.fit(employees)

    sampled = gc.sample(10)

When we view the ``sampled`` data, we should find that all the rows in the sampled data have a 
positive age value.

.. ipython:: python
    :okwarning:

    sampled


Writing Functions
-----------------

What if I want to apply the constraint on multiple columns? In general,
there are three main designs to follow when writing your functions.


1. Functions to be applied to the entire table.

This is similar to the previous example, where ``is_positive`` accepts
``table_data`` as input.

.. code-block:: python

    def is_positive(table_data):
    	return table_data['age'] > 0

    constraint = CustomConstraint(is_valid=is_positive)


2. Functions to be applied to a slice of the data.

In this case, the function should accept a column data as input,
and we specify which column is that one when initializing the 
constraint.

.. code-block:: python

    def is_positive(column_data):
    	return column_data > 0

    constraint = CustomConstraint(columns='age', is_valid=is_positive)

We can also apply the constraint to multiple columns, for example,
if we wish to make the ``prior_years_experience`` positive as well, 
we can add it to the list of columns.

.. code-block:: python

    constraint = CustomConstraint(columns=['age', 'prior_years_experience'], is_valid=is_positive)


3. Functions to be applied to a slice of the data but with access to the entire table.

The last style of supported functions takes as input ``table_data`` and ``column``. The
need for this approach stems from requiring access to another "fixed" column in 
``table_data``. Consider the case where rather than ensuring ``'age'`` and
``'prior_years_experience'`` are positive, we would like to ensure that they are larger than 
``'years_in_the_company'``. In this case, we define our function as

.. code-block:: python

    def is_larger(table_data, column):
    	return table_data[column] > table_data['years_in_the_company']

    constraint = CustomConstraint(columns=['age', 'prior_years_experience'], is_valid=is_larger)

This style gives flexibility to access any column in the table while still operating on 
a column basis.

We covered three supported styles for writing functions for ``is_valid`` method. ``tranform``
and ``reverse_transform`` follow a similar style in their definition as well.

.. note::
	The ``transform`` and ``reverse_transform`` methods return a table. Except
	when operating on ``column_data`` then it returns the transformed or 
	reverse transformed column.


Example
-------

Next we go through a complete example to cover a wholesome implementation.

In this example, we would like to create a constraint that makes ``age`` even. We can
write our functions in either style mentioned above. First, let's look at the ``transform``
function.

.. ipython:: python
    :okwarning:

    def transform_even(table_data):
    	table_data['age'] = table_data['age'] * 2
    	return table_data

We transform the column ``age`` to be even by multiplicating with 2. When now define
the ``reverse_transform`` as doing the opposite effect.

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

We put every thing together in ``CustomConstraint`` and then pass it to our model to 
generate synthetic data.

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


Style 2
~~~~~~~

.. ipython:: python
    :okwarning:

    def transform_even(column_data):
    	return column_data * 2


    def reverse_transform_even(column_data):
    	return column_data / 2


    def is_even(column_data):
    	return column_data % 2 == 0

    constraint = CustomConstraint(
    	columns='age',
    	transform=transform_even, 
    	reverse_transform=reverse_transform_even, 
    	is_valid=is_even
    )

    gc = GaussianCopula(constraints=[constraint])

    gc.fit(employees)

    sampled = gc.sample(10)

Style 3
~~~~~~~

.. ipython:: python
    :okwarning:

    def transform_even(table_data, column):
    	table_data[column] = table_data[column] * 2
    	return table_data


    def reverse_transform_even(table_data, column):
    	table_data[column] = table_data[column] / 2
    	return table_data


    def is_even(table_data, column):
    	return table_data[column] % 2 == 0

    constraint = CustomConstraint(
    	columns=['age', 'prior_years_experience'],
    	transform=transform_even, 
    	reverse_transform=reverse_transform_even, 
    	is_valid=is_even
    )

    gc = GaussianCopula(constraints=[constraint])

    gc.fit(employees)

    sampled = gc.sample(10)
