.. _tabular_preset:

Tabular Preset
==============

The ``TabularPreset`` is a tabular model that comes with pre-configured settings. This
is meant for users who want to get started with using synthetic data and spend
less time worrying about which model to choose or how to tune its parameters.

.. note::

   We are currently in Beta testing our speed-optimized machine learning preset. Help us by
   testing the model and `filing issues <https://github.com/sdv-dev/SDV/issues/new/choose>`__
   for any bugs or feature requests you may have.

What is the FAST_ML preset?
-----------------------

The ``FAST_ML`` preset is our first preset. It uses machine learning (ML) to model your data
while optimizing for the modeling time. This is a great choice if it's your first time using
the SDV for a large custom dataset or if you're exploring the benefits of using ML to create
synthetic data.

What will you get with this preset?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- This preset optimizes for the modeling time while still applying machine learning to model
  and generate synthetic data.
- Your synthetic data will capture correlations that exist between the columns of the original
  data.
- Your synthetic data will adhere to the basic statistical properties of the original columns:
  min/max values, averages and standard deviations.

While other SDV models may create higher quality synthetic data, they will take longer.
Using the ``FAST_ML`` preset allows you to get started with ML to create synthetic data
right now.

Quick Usage
-----------

Preparation
~~~~~~~~~~~

To use this preset, you must have:

1. Your data, loaded as a pandas DataFrame, and
2. (Optional but strongly recommended) A metadata file that describes the columns of your dataset

For this guide, we'll load the demo data and metadata from the SDV. This data contains
information about students, including their grades, major and work experience.

.. ipython:: python
   :okwarning:

   from sdv.demo import load_tabular_demo

   metadata, data = load_tabular_demo('student_placements', metadata=True)
   data.head()

If you want to use your custom dataset, you can load it using pandas.
For example, if your data is available as a CSV file, you can use the ``read_csv`` method.

You can write your metadata as a dictionary. Follow the
`Metadata guide <https://sdv.dev/SDV/developer_guides/sdv/metadata.html#table>`__ to create
a dictionary for a single table. For example, the metadata for our table looks
something like this:

.. code-block:: json

   {
       'fields': {
           'start_date': {'type': 'datetime', 'format': '%Y-%m-%d'},
           'end_date': {'type': 'datetime', 'format': '%Y-%m-%d'},
           'salary': {'type': 'numerical', 'subtype': 'integer'},
           'duration': {'type': 'categorical'},
           'student_id': {'type': 'id', 'subtype': 'integer'},
           'high_perc': {'type': 'numerical', 'subtype': 'float'},
           'high_spec': {'type': 'categorical'},
           'mba_spec': {'type': 'categorical'},
           'second_perc': {'type': 'numerical', 'subtype': 'float'},
           'gender': {'type': 'categorical'},
           'degree_perc': {'type': 'numerical', 'subtype': 'float'},
           'placed': {'type': 'boolean'},
           'experience_years': {'type': 'numerical', 'subtype': 'integer'},
           'employability_perc': {'type': 'numerical', 'subtype': 'float'},
           'mba_perc': {'type': 'numerical', 'subtype': 'float'},
           'work_experience': {'type': 'boolean'},
           'degree_type': {'type': 'categorical'}
       },
       'constraints': [],
       'primary_key': 'student_id'
   }

Modeling
~~~~~~~~
Pass in your metadata to create the TabularPreset FAST_ML model.

.. ipython:: python
   :okwarning:

   from sdv.lite import TabularPreset

   # Use the FAST_ML preset to optimize for modeling time
   model = TabularPreset(name='FAST_ML', metadata=metadata)

Then, simply pass in your data to train the model.

.. ipython:: python
   :okwarning:

   model.fit(data)

The modeling step is optimized for speed. The exact time it takes depends on several factors
including the number of rows, columns and distinct categories in categorical columns. As a
rough benchmark, our analysis shows that:

- Datasets with around 100K rows and 50-100 columns will take a few minutes to model
- Larger datasets with around 1M rows and hundreds of columns may take closer to an hour

After you are finished modeling, you can save the fitted model and load it in again for future use.

.. ipython:: python
   :okwarning:

   # save the model in a new file
   model.save('fast_ml_model.pkl')

   # later, you can load it in again
   model = TabularPreset.load('fast_ml_model.pkl')

Sampling
~~~~~~~~
Once you have your model, you can begin to create synthetic data. Use the sample method and
pass in the number of rows you want to synthesize.

.. ipython:: python
   :okwarning:

   synthetic_data = model.sample(num_rows=100)
   synthetic_data.head()

For creating large amounts of synthetic data, provide a batch_size. This breaks up the sampling
into multiple batches and shows a progress bar. Use the output_file_path parameter to write
results to a file.

.. ipython:: python
   :okwarning:

   model.sample(num_rows=1_000_000, batch_size=10_000, output_file_path='synthetic_data.csv')

Conditional Sampling
~~~~~~~~~~~~~~~~~~~~
The model generates new synthetic data – synthetic rows that do not refer to the original.
But sometimes you may want to fix some values.

For example, you might only be interested in synthesizing science and commerce students with
work experience. Using **conditional sampling**, you can specify the exact, fixed values that
you need. The SDV model will then synthesize the rest of the data.

First, use the Condition object to specify the exact values you want. You specify a dictionary
of column names and the exact value you want, along with the number of rows to synthesize.

.. ipython:: python
   :okwarning:

   from sdv.sampling.tabular import Condition

   # 100 science students with work experience
   science_students = Condition(
      column_values={'high_spec': 'Science', 'work_experience': True}, num_rows=100)

   # 200 commerce students with work experience
   commerce_students = Condition(
      column_values={'high_spec': 'Commerce', 'work_experience': True}, num_rows=200)

You can now use the sample_conditions function and pass in a list of conditions.

.. ipython:: python
   :okwarning:

   all_conditions = [science_students, commerce_students]
   model.sample_conditions(conditions=all_conditions)

Advanced Usage
--------------

Adding Constraints
~~~~~~~~~~~~~~~~~~
A constraint is a logical business rule that must be met by every row in your dataset.

In most cases, the preset is able to learn a general trend and create synthetic data where *most*
of the rows follow the rule. Use a constraint if you want to enforce that *all* of the rows
must follow the rule.

In our dataset, we have a constraint: All the numerical values in the duration column must be divisible by 3.
We can describe this using a FixedIncrements constraint.


.. ipython:: python
   :okwarning:

   from sdv.constraints import FixedIncrements

   # use the formula when defining the constraint
   duration_constraint = FixedIncrements(
       column_name='duration',
       increment_value=3,
   )

You can input constraints into the presets when creating your model.

.. ipython:: python
   :okwarning:

   constrained_model = TabularPreset(
       name='FAST_ML',
       metadata=metadata,
       constraints=[duration_constraint],
   )
   constrained_model.fit(data)

When you sample from the model, the synthetic data will follow the constraints

.. ipython:: python
   :okwarning:

   constrained_synthetic_data = constrained_model.sample(num_rows=1_000)
   constrained_synthetic_data.head(10)

To read more about defining constraints, see the
`Handling Constraints User Guide <https://sdv.dev/SDV/user_guides/single_table/handling_constraints.html>`__.

Resources
---------
The SDV (Synthetic Data Vault) is an open source project built & maintained by DataCebo. It is free to use under the MIT License.

For other resources see our: `GitHub <https://github.com/sdv-dev/SDV>`__,
`Docs <https://sdv.dev/SDV/>`__, `Blog <https://sdv.dev/blog/>`__.
