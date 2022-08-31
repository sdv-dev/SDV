.. _handling_constraints:

Inputting Business Logic Using Constraints
==========================================

Do you have rules in your dataset that every row in the data must follow? You can use constraints
to describe business logic to any of the SDV single table models.

The SDV has predefined constraints that are commonly found in datasets. For example:


- Fixing combinations. Your table might have two different columns for city and country. The
  values in those columns should not be shuffled because that would result in incorrect locations
  (eg. Paris USA or London Italy).
- Comparing inequalities. Your table might have two different columns for an employee's start_date
  and end_date that are related to each other: The start_date must always come before the end_date.

In this guide, we'll walk through the usage of each predefined constraint.

Load a Tabular Demo
-------------------

To illustrate some of the constraints, let's load a small table that contains some details
about employees from several companies.

.. ipython:: python
    :okwarning:

    from sdv.demo import load_tabular_demo

    employees = load_tabular_demo()
    employees

This table contains a few rules that can be written as predefined constraints. We'll use it as an
example when describing the constraints.

Predefined Constraints
----------------------

Unique
~~~~~~

The Unique constraint enforces that the values in column (or set of columns) are unique within the
entire table.

In our demo table, there is a Unique constraint: Within a company, all the employee ids must be
unique.

Enforce this by creating a Unique constraint. This object accepts a list of 1 or more column names.

.. ipython:: python
    :okwarning:

    from sdv.constraints import Unique

    unique_employee_id_company_constraint = Unique(
        column_names=['employee_id', 'company']
    )

.. note::
    The SDV already ensures that primary keys are unique in the dataset. You do not need to add
    a Unique constraint on these columns.

FixedCombinations
~~~~~~~~~~~~~~~~~

The FixedCombinations constraint enforces that the combinations between a set of columns are fixed.
That is, no other permutations or shuffling is allowed other than what's already observed in the
real data.

In our demo table, there is a FixedCombinations constraint: Each company has a fixed set of
departments. The company and department values should not be shuffled in the synthetic data.

Enforce this by creating a FixedCombinations constraint. This object accepts a list of 2 or
more column names.

.. ipython:: python
    :okwarning:

    from sdv.constraints import FixedCombinations

    fixed_company_department_constraint = FixedCombinations(
        column_names=['company', 'department']
    )

Inequality
~~~~~~~~~~

The Inequality constraint enforces an inequality relationship between a pair of columns.
For every row, the value in one column must be greater than a value in another.

In our demo table, there is an Inequality constraint: The current age of an employee must be
greater than or equal to the age they were when they joined.

Enforce this by creating an Inequality constraint. This object accepts column names for the high
and low columns. The columns can be either numerical or datetime.

.. ipython:: python
    :okwarning:

    from sdv.constraints import Inequality

    age_gt_age_when_joined_constraint = Inequality(
        low_column_name='age_when_joined',
        high_column_name='age'
    )

ScalarInequality
~~~~~~~~~~~~~~~~

The ScalarInequality constraint enforces that all values in a column are greater or less than a
fixed (scalar) value. That is, it enforces a lower or upper bound to the synthetic data.

In our demo table, we can define a ScalarInequality constraint: All employees must be 18 or older.

Enforce this by creating a ScalarInequality constraint. This object accepts a numerical or
datetime column name and value. It also expects an inequality relation that must be one of
">", ">=", "<" or "<=".

.. ipython:: python
    :okwarning:

    from sdv.constraints import ScalarInequality

    age_gt_18 = ScalarInequality(
        column_name='age',
        relation='>=',
        value=18
    )

.. note::
    All SDV tabular models have an enforce_min_max_values parameter that you set to enforce bounds
    on all columns. This constraint is redundant if you set this model parameter.

Positive and Negative
~~~~~~~~~~~~~~~~~~~~~

The Positive and Negative constraints are shortcuts to the ScalarInequality constraint when
the column's values must be >0 or <0.

In our demo table, we can define a Positive constraint: All employee ages must be positive.

Enforce this by creating a Positive constraint. This object accepts a numerical column name.
(The Negative constraint works the same way.)

.. ipython:: python

    from sdv.constraints import Positive

    age_positive = Positive(column_name='age')

.. note::
    All SDV tabular models have an enforce_min_max_value parameter that you set to enforce bounds
    on all columns. This constraint is redundant if you set this model parameter.

OneHotEncoding
~~~~~~~~~~~~~~

The OneHotEncoding constraint enforces that a set of columns follow a
`one hot encoding scheme <https://en.wikipedia.org/wiki/One-hot#Machine_learning_and_statistics>`__
. That is, exactly one of the columns must contain a value of 1 while all the others must be 0.

In our demo table, we have a OneHotEncoding constraint: An employee can only be one of: full time,
part time or contractor. That is, only 1 of these columns must be 1 while the others must be a 0.

Enforce this by creating a OneHotEncoding constraint. The object accepts a list of column names
that, together, are part of the one hot encoding scheme.

.. ipython:: python

    from sdv.constraints import OneHotEncoding

    job_category_constraint = OneHotEncoding(
        column_names=['full_time', 'part_time', 'contractor']
    )

FixedIncrements
~~~~~~~~~~~~~~~

The FixedIncrements constraint enforces that all the values in a column are increments of a
particular, fixed value. That is, all the data must be divisible by the value.

We do not have a FixedIncrements constraint in our demo table. But we can imagine a table where
all the salary values must be divisible by 500.

Enforce this by creating a FixedIncrements constraint. This object accepts a numerical column
name and an increment value that must be an integer greater than 1.

.. ipython:: python

    from sdv.constraints import FixedIncrements

    # this constraint does not actually exist in the demo dataset
    salary_divisble_by_500 = FixedIncrements(
        column_name='salary',
        increment_value=500
    )

Range
~~~~~

The Range constraint enforces that for all rows, the value of one of the columns is bounded by
the values in the other two columns.

We do not have a Range constraint in our demo table. But we can imagine a table where an
employee's age is bounded by the age when they first started working and an age when they will
retire.

Enforce this by creating a Range constraint. This object accepts high, middle and low column names.
The columns can be either numerical or datetime.

.. ipython:: python

    from sdv.constraints import Range

    # this constraint does not actually exist in the demo dataset
    age_btwn_joined_retirement = Range(
        low_column_name='age_started_working',
        middle_column_name='age_today',
        high_column_name='age_when_retiring'
    )

.. note::
    This constraint assumes strict bounds between the low, middle and high column names.
    That is: low < middle < high. You can express other business logic using a multiple
    Inequality and ScalarInequality constraints.

ScalarRange
~~~~~~~~~~~

The ScalarRange constraint enforces that all the values in a column are in between two known,
fixed values. That is, it enforces upper and lower bounds to the data.

In our demo table, we can define a ScalarRange constraint: All employees must be between the
ages of 18 and 100.

Enforce this by creating a ScalarRange constraint. This object accepts a numerical or datetime
column name and the low and high values. It also accepts a boolean that describes whether the
ranges are strict (exclusive) or not (inclusive).

.. ipython:: python

    from sdv.constraints import ScalarRange

    age_btwn_18_100 = ScalarRange(
        column_name='age',
        low_value=18,
        high_value=100,
        strict_boundaries=False
    )

.. note::
    All SDV tabular models have an enforce_min_max_values parameter that you set to enforce bounds
    on all columns. This constraint is redundant if you set this model parameter.

Applying the Constraints
------------------------

Once you have defined the constraints, you can use them in any SDV single table model
(TabularPreset, GaussianCopula, CopulaGAN, CTGAN and TVAE). Use the constraints parameter
to pass in the objects a list.

.. ipython:: python

    from sdv.tabular import GaussianCopula

    constraints = [
        unique_employee_id_company_constraint,
        fixed_company_department_constraint,
        age_gt_age_when_joined_constraint,
        job_category_constraint,
        age_btwn_18_100
    ]

    model = GaussianCopula(constraints=constraints, enforce_min_max_values=False)

Then you can fit the model using the real data. During this process, the SDV ensures that the
model learns the constraints.

.. ipython:: python
    :okwarning:

    model.fit(employees)

.. warning::
    The constraints must accurately describe the data. Constraints are business rules that must be
    followed by every row of your data. If the real data does not fully meet the constraint, the
    model will not be able to learn it well. The SDV will throw an error.

Finally, you can sample synthetic data. Observe that every row in the synthetic data adheres to
the constraints.

.. ipython:: python

    synthetic_data = model.sample(num_rows=10)
    synthetic_data

FAQs
----

.. warning::
    **Constraints may slow down the synthetic data model & leak privacy.** Before adding a
    constraint to your model, carefully consider whether it is necessary.  Here are a few questions
    to ask:

    - How do I plan to use the synthetic data? Without the constraint, the rule may still be valid
      a majority of the time. Only add the constraint if you require 100% adherence.
    - Who do I plan to share the synthetic data with? Consider whether they will be able to use
      the business rule to uncover sensitive information about the real data.
    - How did the rule come to be? In some cases, there may be other data sources that are present
      without extra columns and rules.

    In the ideal case, there are only a handful constraints you are applying to your model.

.. collapse:: When do constraints affect the modeling & sampling performance?

    In most cases, the time it takes to fit the model and sample synthetic data should not be
    significantly affected if you add a few constraints. However, there are certain scenarios
    where you may notice a slow-down:

    - You have a large number of constraints that overlap. That is, multiple constraints are
      referencing the same column(s) in the data.

    - Your constrained data has a high cardinality. For example, you have a categorical column
      with hundreds of possible categories that you are using in a FixedCombinations constraint.

    - You are conditional sampling on a constrained column. This requires some special processing
      and it may not always be possible to efficiently create conditional synthetic data.

    For any questions or feature requests related to performance, please create an issue describing
    your data, constraints and sampling needs.

.. collapse:: What happened to Rounding and ColumnFormula?

    Rounding and ColumnFormula constraints were available in older versions of the SDV. These
    constraints are no longer included as predefined constraints because there are other ways
    to achieve the same logic:

    - **Rounding**: All SDV single table models contain a 'rounding' parameter. By default, they
      learn the number of decimal digits in your data and enforce that the synthetic data has the
      same.

    - **ColumnFormula**: In this version of the SDV, you can implement a formula as a
      CustomConstraint. See the Defining Custom Constraints guide for more details.

.. collapse:: Why am I getting a ConstraintsNotMetError when I try to fit my data?

    A constraint should describe a rule that is true for every row in your real data. If any rows
    in the real data violate the rule, the SDV will throw a ConstraintsNotMetError. Since the
    constraint is not true in your real data, the model will not be able to learn it.

    If you see this error, you have two options:

    - (recommended) Remove the constraint. This ensures the model learns patterns that exist in the
      real data. You can use conditional sampling later to generate synthetic data with specific
      values.

    - Clean your input dataset. If you remove the violative rows in the real data, then you will be
      able to apply the constraint. This is not recommended because even if the model can learn the
      constraint, it is not truly representative of the full, original dataset.

.. collapse:: How does the SDV handle the constraints?

    Under-the-hood, the SDV uses a combination of strategies to ensure that the synthetic data
    always follows the constraints. These strategies are:

    1. **Transformation**: Most of the time, it's possible to transform the data in a way that
       guarantees the models will be able to learn the constraint. This is paired with a reverse
       transformation to ensure the synthetic data looks like the original.

    2. **Reject Sampling**: Another strategy is to model and sample synthetic data as usual, and
       then throw away any rows in the synthetic data that violate the constraints.

    Transformation is the most efficient strategy, but it is not always possible to use. For
    example, multiple constraints might be attempting to transform the same column, or the
    logic itself may not be possible to achieve through transformation. 

    In such cases, the SDV will fall back to using reject sampling. You'll get a warning when
    this happens. Reject sampling may slow down the sampling process but there will be no other
    effect on the synthetic data's quality or validity.
