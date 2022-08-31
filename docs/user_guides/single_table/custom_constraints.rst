.. _custom_constraints:

Custom Constraints
==================

If you have business logic that cannot be represented using
:ref:`Predefined Constraints <Predefined Constraints>`,
you can define custom logic. In this guide, we'll walk through the process for defining a custom
constraint and using it.

Defining your custom constraint
-------------------------------
To define your custom constraint you need to write some functionality in a separate Python file.
This includes:

* **Validity Check**: A test that determines whether a row in the data meets the rule, and
* (optional) **Transformation Functions**: Functions to modify the data before & after modeling

The SDV then uses the functionality you provided, as shown in the diagram below.

.. image:: /images/custom_constraint.png

Each function (validity, transform and reverse transform) must accept the same inputs:

- column_names: The names of the columns involved in the constraints
- data: The full dataset, represented as a ``pandas.DataFrame``
- <other parameters>: Any other parameters that are necessary for your logic

Example
~~~~~~~

Let's demonstrate this using our demo dataset.

.. ipython:: python
    :okwarning:

    from sdv.demo import load_tabular_demo

    employees = load_tabular_demo()
    employees


The dataset contains basic details about employees in some fictional companies. Many of the rules
in the dataset can be described using predefined constraints. However, there is one complex rule
that needs a custom constraint:

- If the employee is not a contractor (contractor == 0), then the salary must be divisible by 500
- Otherwise if the employee is a contractor (contractor == 1), then this rule does not apply

.. note::
    This is similar to the predefined :ref:`FixedIncrements <FixedIncrements>` constraint
    with the addition of an exclusion criteria (exclude the constraint check if the employee
    is a contractor).

Validity Check
^^^^^^^^^^^^^^

The validity check should return a ``pandas.Series`` of ``True``/``False`` values that determine
whether each row is valid.

Let's code the logic up using parameters:

- **column_names** will be a single item list containing the column that must be divisible
  (eg. salary)
- **data** will be the full dataset
- Custom parameter: **increment** describes the numerical increment (eg. 500)
- Custom parameter: **exclusion_column** describes the column with the exclusion criteria
  (eg. contractor)

.. code-block:: python

    def is_valid(column_names, data, increment, exclusion_column):
        column_name=column_names[0]

        is_divisible = (data[column_name] % increment == 0)
        is_excluded = (data[exclusion_column] > 0)

        return (is_divisible | is_excluded)


Transformations
^^^^^^^^^^^^^^^

The transformations must return the full datasets with particular columns transformed. We can
modify, delete or add columns as long as we can reverse the transformation later.

In our case, the transformation can just divide each of the values in the column by the increment.

.. code-block:: python

    def transform(column_names, data, increment, exclusion_column):
        column_name = column_names[0]
        data[column_name] = data[column_name] / increment
        return data


Reversing the transformation is trickier. If we multiply every value by the increment, the
salaries won't necessarily be divisible by 500. Instead we should:

- Round values to whole numbers whenever the employee is not a contractor first, and then
- Multiply every value by 500

.. code-block:: python

    def reverse_transform(column_names, transformed_data, increment, exclusion_column):
        column_name = column_names[0]
  
        is_included = (transformed_data[exclusion_column] == 0)
        rounded_data = transformed_data[is_included][column_name].round()
        transformed_data.at[is_included, column_name] = rounded_data

        transformed_data[column_name] *= increment
        return transformed_data


Creating your class
~~~~~~~~~~~~~~~~~~~

Finally, we can put all the functionality together to create a class that describes our
constraint. Use the **create_custom_constraint** factory method to do this. It accepts your
functions as inputs and returns a class that's ready to use.

You can name this class whatever you'd like. Since our constraint is similar to
``FixedIncrements``, let's call it ``FixedIncrementsWithExclusion``.

.. ipython:: python
    :okwarning:

    from sdv.constraints import create_custom_constraint

    FixedIncrementsWithExclusion = create_custom_constraint(
        is_valid_fn=is_valid,
        transform_fn=transform, # optional
        reverse_transform_fn=reverse_transform # optional
    )


Using your custom constraint
----------------------------

Now that you have a class, you can use it like any other predefined constraint. Create an object
by putting in the parameters you defined. Note that you do not need to input the data.

You can apply the same constraint to other columns by creating a different object. In our case
the **annual_bonus** column also follows the same logic.

.. ipython:: python
    :okwarning:

    salary_divis_500 = FixedIncrementsWithExclusion(
       column_names=['salary'],
       increment=500,
       exclusion_column='contractor'
    )

    bonus_divis_500 = FixedIncrementsWithExclusion(
       column_names=['annual_bonus'],
       increment=500,
       exclusion_column='contractor'
    )


Finally, input these constraints into your model using the constraints parameter just like you
would for predefined constraints.

.. ipython:: python
    :okwarning:

    from sdv.tabular import GaussianCopula

    constraints = [
      # you can add predefined constraints here too
      salary_divis_500,
      bonus_divis_500
    ]

    model = GaussianCopula(constraints=constraints, enforce_min_max_values=False)

    model.fit(employees)

Now, when you sample from the model, all rows of the synthetic data will follow the custom
constraint.

.. ipython:: python
    :okwarning:

    synthetic_data = model.sample(num_rows=10)
    synthetic_data
