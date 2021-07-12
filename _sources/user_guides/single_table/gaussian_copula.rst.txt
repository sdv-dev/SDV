.. _gaussian_copula:

GaussianCopula Model
====================

In this guide we will go through a series of steps that will let you
discover functionalities of the ``GaussianCopula`` model, including how
to:

-  Create an instance of a ``GaussianCopula``.
-  Fit the instance to your data.
-  Generate synthetic versions of your data.
-  Use ``GaussianCopula`` to anonymize PII information.
-  Customize the data transformations to improve the learning process.
-  Specify the column distributions to improve the output quality.

What is GaussianCopula?
-----------------------

The ``sdv.tabular.GaussianCopula`` model is based on
`copula funtions <https://en.wikipedia.org/wiki/Copula_%28probability_theory%29>`__.

In mathematical terms, a *copula* is a distribution over the unit
cube :math:`{\displaystyle [0,1]^{d}}` which is constructed from a
multivariate normal distribution over
:math:`{\displaystyle \mathbb {R} ^{d}}` by using the probability
integral transform. Intuitively, a *copula* is a mathematical function
that allows us to describe the joint distribution of multiple random
variables by analyzing the dependencies between their marginal
distributions.

Let's now discover how to learn a dataset and later on generate
synthetic data with the same format and statistical properties by using
the ``GaussianCopula`` model.

Quick Usage
-----------

We will start by loading one of our demo datasets, the
``student_placements``, which contains information about MBA students
that applied for placements during the year 2020.

.. ipython:: python
    :okwarning:

    from sdv.demo import load_tabular_demo

    data = load_tabular_demo('student_placements')
    data.head()

As you can see, this table contains information about students which
includes, among other things:

-  Their id and gender
-  Their grades and specializations
-  Their work experience
-  The salary that they were offered
-  The duration and dates of their placement

You will notice that there is data with the following characteristics:

-  There are float, integer, boolean, categorical and datetime values.
-  There are some variables that have missing data. In particular, all
   the data related to the placement details is missing in the rows
   where the student was not placed.

Let us use the ``GaussianCopula`` to learn this data and then sample
synthetic data about new students to see how well the model captures the
characteristics indicated above. In order to do this you will need to:

-  Import the ``sdv.tabular.GaussianCopula`` class and create an
   instance of it.
-  Call its ``fit`` method passing our table.
-  Call its ``sample`` method indicating the number of synthetic rows
   that you want to generate.

.. ipython:: python
    :okwarning:

    from sdv.tabular import GaussianCopula

    model = GaussianCopula()
    model.fit(data)

.. note::

    Notice that the model ``fitting`` process took care of transforming the
    different fields using the appropriate `Reversible Data
    Transforms <http://github.com/sdv-dev/RDT>`__ to ensure that the data
    has a format that the ``GaussianMultivariate`` model can handle.


Generate synthetic data from the model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the modeling has finished you are ready to generate new synthetic
data by calling the ``sample`` method from your model passing the number
of rows that we want to generate.

.. ipython:: python
    :okwarning:

    new_data = model.sample(200)

This will return a table identical to the one which the model was fitted
on, but filled with new data which resembles the original one.

.. ipython:: python
    :okwarning:

    new_data.head()

.. note::

    You can control the number of rows by specifying the number of
    ``samples`` in the ``model.sample(<num_rows>)``. To test, try
    ``model.sample(10000)``. Note that the original table only had ~200
    rows.

Save and Load the model
~~~~~~~~~~~~~~~~~~~~~~~

In many scenarios it will be convenient to generate synthetic versions
of your data directly in systems that do not have access to the original
data source. For example, if you may want to generate testing data on
the fly inside a testing environment that does not have access to your
production database. In these scenarios, fitting the model with real
data every time that you need to generate new data is feasible, so you
will need to fit a model in your production environment, save the fitted
model into a file, send this file to the testing environment and then
load it there to be able to ``sample`` from it.

Let's see how this process works.

Save and share the model
^^^^^^^^^^^^^^^^^^^^^^^^

Once you have fitted the model, all you need to do is call its ``save``
method passing the name of the file in which you want to save the model.
Note that the extension of the filename is not relevant, but we will be
using the ``.pkl`` extension to highlight that the serialization
protocol used is
`pickle <https://docs.python.org/3/library/pickle.html>`__.

.. ipython:: python
    :okwarning:

    model.save('my_model.pkl')

This will have created a file called ``my_model.pkl`` in the same
directory in which you are running SDV.

.. important::

    If you inspect the generated file you will notice that its size is much
    smaller than the size of the data that you used to generate it. This is
    because the serialized model contains **no information about the
    original data**, other than the parameters it needs to generate
    synthetic versions of it. This means that you can safely share this
    ``my_model.pkl`` file without the risk of disclosing any of your real
    data!

Load the model and generate new data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The file you just generated can be sent over to the system where the
synthetic data will be generated. Once it is there, you can load it
using the ``GaussianCopula.load`` method, and then you are ready to
sample new data from the loaded instance:

.. ipython:: python
    :okwarning:

    loaded = GaussianCopula.load('my_model.pkl')
    new_data = loaded.sample(200)

.. warning::

    Notice that the system where the model is loaded needs to also have
    ``sdv`` installed, otherwise it will not be able to load the model and
    use it.


Specifying the Primary Key of the table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One of the first things that you may have noticed when looking at the demo
data is that there is a ``student_id`` column which acts as the primary
key of the table, and which is supposed to have unique values. Indeed,
if we look at the number of times that each value appears, we see that
all of them appear at most once:

.. ipython:: python
    :okwarning:

    data.student_id.value_counts().max()

However, if we look at the synthetic data that we generated, we observe
that there are some values that appear more than once:

.. ipython:: python
    :okwarning:

    new_data[new_data.student_id == new_data.student_id.value_counts().index[0]]


This happens because the model was not notified at any point about the
fact that the ``student_id`` had to be unique, so when it generates new
data it will provoke collisions sooner or later. In order to solve this,
we can pass the argument ``primary_key`` to our model when we create it,
indicating the name of the column that is the index of the table.

.. ipython:: python
    :okwarning:

    model = GaussianCopula(
        primary_key='student_id'
    )
    model.fit(data)
    new_data = model.sample(200)
    new_data.head()

As a result, the model will learn that this column must be unique and
generate a unique sequence of values for the column:

.. ipython:: python
    :okwarning:

    new_data.student_id.value_counts().max()


Anonymizing Personally Identifiable Information (PII)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There will be many cases where the data will contain Personally
Identifiable Information which we cannot disclose. In these cases, we
will want our Tabular Models to replace the information within these
fields with fake, simulated data that looks similar to the real one but
does not contain any of the original values.

Let's load a new dataset that contains a PII field, the
``student_placements_pii`` demo, and try to generate synthetic versions
of it that do not contain any of the PII fields.

.. note::

    The ``student_placements_pii`` dataset is a modified version of the
    ``student_placements`` dataset with one new field, ``address``, which
    contains PII information about the students. Notice that this additional
    ``address`` field has been simulated and does not correspond to data
    from the real users.

.. ipython:: python
    :okwarning:

    data_pii = load_tabular_demo('student_placements_pii')
    data_pii.head()


If we use our tabular model on this new data we will see how the
synthetic data that it generates discloses the addresses from the real
students:

.. ipython:: python
    :okwarning:

    model = GaussianCopula(
        primary_key='student_id',
    )
    model.fit(data_pii)
    new_data_pii = model.sample(200)
    new_data_pii.head()

More specifically, we can see how all the addresses that have been generated
actually come from the original dataset:

.. ipython:: python
    :okwarning:

    new_data_pii.address.isin(data_pii.address).sum()


In order to solve this, we can pass an additional argument
``anonymize_fields`` to our model when we create the instance. This
``anonymize_fields`` argument will need to be a dictionary that
contains:

-  The name of the field that we want to anonymize.
-  The category of the field that we want to use when we generate fake
   values for it.

The list complete list of possible categories can be seen in the `Faker
Providers <https://faker.readthedocs.io/en/master/providers.html>`__
page, and it contains a huge list of concepts such as:

-  name
-  address
-  country
-  city
-  ssn
-  credit\_card\_number
-  credit\_card\_expire
-  credit\_card\_security\_code
-  email
-  telephone
-  ...

In this case, since the field is an e-mail address, we will pass a
dictionary indicating the category ``address``

.. ipython:: python
    :okwarning:

    model = GaussianCopula(
        primary_key='student_id',
        anonymize_fields={
            'address': 'address'
        }
    )
    model.fit(data_pii)


As a result, we can see how the real ``address`` values have been
replaced by other fake addresses:

.. ipython:: python
    :okwarning:

    new_data_pii = model.sample(200)
    new_data_pii.head()


Which means that none of the original addresses can be found in the sampled
data:

.. ipython:: python
    :okwarning:

    data_pii.address.isin(new_data_pii.address).sum()


Advanced Usage
--------------

Now that we have discovered the basics, let's go over a few more
advanced usage examples and see the different arguments that we can pass
to our ``GaussianCopula`` Model in order to customize it to our needs.

How to set transforms to use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One thing that you may have noticed when executing the previous steps is
that the fitting process took much longer on the
``student_placements_pii`` dataset than it took on the previous version
that did not contain the student ``address``. This happens because the
``address`` field is interpreted as a categorical variable, which the
``GaussianCopula`` `one-hot
encoded <https://en.wikipedia.org/wiki/One-hot>`__ generating 215 new
columns that it had to learn afterwards.

This transformation, which in this case was very inefficient, happens
because the Tabular Models apply `Reversible Data
Transforms <https://github.com/sdv-dev/RDT>`__ under the hood to
transform all the non-numerical variables, which the underlying models
cannot handle, into numerical representations which they can properly
work with. In the case of the ``GaussianCopula``, the default
transformation is a One-Hot encoding, which can work very well with
variables that have a small number of different values, but which is
very inefficient in cases where there is a large number of values.

For this reason, the Tabular Models have an additional argument called
``field_transformers`` that let you select which transformer to apply to
each column. This ``field_transformers`` argument must be passed as a
``dict`` which contains the name of the fields for which we want to use
a transformer different than the default, and the name of the
transformer that we want to use.

Possible transformer names are:

-  ``integer``: Uses a ``NumericalTransformer`` of dtype ``int``.
-  ``float``: Uses a ``NumericalTransformer`` of dtype ``float``.
-  ``categorical``: Uses a ``CategoricalTransformer`` without gaussian
   noise.
-  ``categorical_fuzzy``: Uses a ``CategoricalTransformer`` adding
   gaussian noise.
-  ``one_hot_encoding``: Uses a ``OneHotEncodingTransformer``.
-  ``label_encoding``: Uses a ``LabelEncodingTransformer``.
-  ``boolean``: Uses a ``BooleanTransformer``.
-  ``datetime``: Uses a ``DatetimeTransformer``.

**NOTE**: For additional details about each one of the transformers,
please visit `RDT <https://github.com/sdv-dev/RDT>`__

Let's now try to improve the previous fitting process by changing the
transformer that we use for the ``address`` field to something other
than the default. As an example, we will use the ``label_encoding``
transformer, which instead of generating one column for each possible
value, it just replaces each value with a unique integer value.

.. ipython:: python
    :okwarning:

    model = GaussianCopula(
        primary_key='student_id',
        anonymize_fields={
            'address': 'address'
        },
        field_transformers={
            'address': 'label_encoding'
        }
    )
    model.fit(data_pii)
    new_data_pii = model.sample(200)
    new_data_pii.head()

Setting Bounds and Specifying Rounding for Numerical Columns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, the model will learn the upper and lower bounds of the
input data, and use that for sampling. This means that all sampled data
will be between the maximum and minimum values found in the original
dataset for each numeric column. This option can be overwritten using the
``min_value`` and ``max_value`` model arguments. These values can either
be set to a numeric value, set to ``'auto'`` which is the default setting,
or set to ``None`` which will mean the column is boundless.

The model will also learn the number of decimal places to round to by default.
This option can be overwritten using the ``rounding`` parameter. The value can
be an int specifying how many decimal places to round to, ``'auto'`` which is
the default setting, or ``None`` which means the data will not be rounded.

Since we may want to sample values outside of the ranges in the original data,
let's pass the ``min_value`` and ``max_value`` arguments as `None` to the model.
To keep the number of decimals consistent across columns, we can set ``rounding``
to be 2.

.. ipython:: python
    :okwarning:

    model = GaussianCopula(
        primary_key='student_id',
        min_value=None,
        max_value=None,
        rounding=2
    )
    model.fit(data)

    unbounded_data = model.sample(10)
    unbounded_data

As you may notice, the sampled data may have values outside the range of
the original data.

Exploring the Probability Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

During the previous steps, every time we fitted the ``GaussianCopula``
it performed the following operations:

1. Learn the format and data types of the passed data
2. Transform the non-numerical and null data using `Reversible Data
   Transforms <https://github.com/sdv-dev/RDT>`__ to obtain a fully
   numerical representation of the data from which we can learn the
   probability distributions.
3. Learn the probability distribution of each column from the table
4. Transform the values of each numerical column by converting them
   to their marginal distribution CDF values and then applying an
   inverse CDF transformation of a standard normal on them.
5. Learn the correlations of the newly generated random variables.

After this, when we used the model to generate new data for our table
using the ``sample`` method, it did:

5. Sample from a Multivariate Standard Normal distribution with the
   learned correlations.
6. Revert the sampled values by computing their standard normal CDF
   and then applying the inverse CDF of their marginal distributions.
7. Revert the RDT transformations to go back to the original data
   format.

As you can see, during these steps the *Marginal Probability
Distributions* have a very important role, since the ``GaussianCopula``
had to learn and reproduce the individual distributions of each column
in our table. We can explore the distributions which the
``GaussianCopula`` used to model each column using its
``get_distributions`` method:

.. ipython:: python
    :okwarning:

    model = GaussianCopula(
        primary_key='student_id',
        min_value=None,
        max_value=None
    )
    model.fit(data)
    distributions = model.get_distributions()

This will return us a ``dict`` which contains the name of the
distribution class used for each column:

.. ipython:: python
    :okwarning:

    distributions

.. note::

    In this list we will see multiple distributions for each one of the
    columns that we have in our data. This is because the RDT
    transformations used to encode the data numerically often use more than
    one column to represent each one of the input variables.

Let's explore the individual distribution of one of the columns in our
data to better understand how the ``GaussianCopula`` processed them and
see if we can improve the results by manually specifying a different
distribution. For example, let's explore the ``experience_years`` column
by looking at the frequency of its values within the original data:

.. ipython:: python
    :okwarning:

    data.experience_years.value_counts()

    @savefig experience_years_1.png width=4in
    data.experience_years.hist();


By observing the data we can see that the behavior of the values in this
column is very similar to a Gamma or even some types of Beta
distribution, where the majority of the values are 0 and the frequency
decreases as the values increase.

Was the ``GaussianCopula`` able to capture this distribution on its own?

.. ipython:: python
    :okwarning:

    distributions['experience_years']


It seems that it was not, as it rather thought that the behavior was
closer to a Gaussian distribution. And, as a result, we can see how the
generated values now contain negative values which are invalid for this
column:

.. ipython:: python
    :okwarning:

    new_data.experience_years.value_counts()

    @savefig experience_years_2.png width=4in
    new_data.experience_years.hist();


Let's see how we can improve this situation by passing the
``GaussianCopula`` the exact distribution that we want it to use for
this column.

Setting distributions for individual variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``GaussianCopula`` class offers the possibility to indicate which
distribution to use for each one of the columns in the table, in order
to solve situations like the one that we just described. In order to do
this, we need to pass a ``field_distributions`` argument with ``dict``
that indicates the distribution that we want to use for each column.

Possible values for the distribution argument are:

-  ``univariate``: Let ``copulas`` select the optimal univariate
   distribution. This may result in non-parametric models being used.
-  ``parametric``: Let ``copulas`` select the optimal univariate
   distribution, but restrict the selection to parametric distributions
   only.
-  ``bounded``: Let ``copulas`` select the optimal univariate
   distribution, but restrict the selection to bounded distributions
   only. This may result in non-parametric models being used.
-  ``semi_bounded``: Let ``copulas`` select the optimal univariate
   distribution, but restrict the selection to semi-bounded
   distributions only. This may result in non-parametric models being
   used.
-  ``parametric_bounded``: Let ``copulas`` select the optimal univariate
   distribution, but restrict the selection to parametric and bounded
   distributions only.
-  ``parametric_semi_bounded``: Let ``copulas`` select the optimal
   univariate distribution, but restrict the selection to parametric and
   semi-bounded distributions only.
-  ``gaussian``: Use a Gaussian distribution.
-  ``gamma``: Use a Gamma distribution.
-  ``beta``: Use a Beta distribution.
-  ``student_t``: Use a Student T distribution.
-  ``gaussian_kde``: Use a GaussianKDE distribution. This model is
   non-parametric, so using this will make ``get_parameters`` unusable.
-  ``truncated_gaussian``: Use a Truncated Gaussian distribution.

Let's see what happens if we make the ``GaussianCopula`` use the
``gamma`` distribution for our column.

.. ipython:: python
    :okwarning:

    from sdv.tabular import GaussianCopula

    model = GaussianCopula(
        primary_key='student_id',
        field_distributions={
            'experience_years': 'gamma'
        },
        min_value=None,
        max_value=None
    )
    model.fit(data)

After this, we can see how the ``GaussianCopula`` used the indicated
distribution for the ``experience_years`` column

.. ipython:: python
    :okwarning:

    model.get_distributions()['experience_years']


And, as a result, we can see how the generated data now have a
behavior which is closer to the original data and always stays within
the valid values range.

.. ipython:: python
    :okwarning:

    new_data = model.sample(len(data))
    new_data.experience_years.value_counts()

    @savefig experience_years_3.png width=4in
    new_data.experience_years.hist();


.. note::

    Even though there are situations like the one shown above where manually
    choosing a distribution seems to give better results, in most cases the
    ``GaussianCopula`` will be able to find the optimal distribution on its
    own, making this manual search of the marginal distributions necessary
    on very little occasions.

Conditional Sampling
~~~~~~~~~~~~~~~~~~~~

As the name implies, conditional sampling allows us to sample from a conditional
distribution using the ``GaussianCopula`` model, which means we can generate only values that
satisfy certain conditions. These conditional values can be passed to the ``conditions``
parameter in the ``sample`` method either as a dataframe or a dictionary.

In case a dictionary is passed, the model will generate as many rows as requested,
all of which will satisfy the specified conditions, such as ``gender = M``.

.. ipython:: python
    :okwarning:

    conditions = {
        'gender': 'M'
    }
    model.sample(5, conditions=conditions)


It's also possible to condition on multiple columns, such as
``gender = M, 'experience_years': 0``.

.. ipython:: python
    :okwarning:

    conditions = {
        'gender': 'M',
        'experience_years': 0
    }
    model.sample(5, conditions=conditions)


The ``conditions`` can also be passed as a dataframe. In that case, the model
will generate one sample for each row of the dataframe, sorted in the same
order. Since the model already knows how many samples to generate, passing
it as a parameter is unnecessary. For example, if we want to generate three
samples where ``gender = M`` and three samples with ``gender = F``, we can do the
following:

.. ipython:: python
    :okwarning:

    import pandas as pd

    conditions = pd.DataFrame({
        'gender': ['M', 'M', 'M', 'F', 'F', 'F'],
    })
    model.sample(conditions=conditions)


``GaussianCopula`` also supports conditioning on continuous values, as long as the values
are within the range of seen numbers. For example, if all the values of the
dataset are within 0 and 1, ``GaussianCopula`` will not be able to set this value to 1000.

.. ipython:: python
    :okwarning:

    conditions = {
        'degree_perc': 70.0
    }
    model.sample(5, conditions=conditions)


.. note::

    Currently, conditional sampling works through a rejection sampling process,
    where rows are sampled repeatedly until one that satisfies the conditions is
    found. In case you are running into a ``Could not get enough valid rows within
    x trials`` or simply wish to optimize the results, there are three parameters
    that can be fine-tuned: ``max_rows_multiplier``, ``max_retries`` and ``float_rtol``.
    More information about these parameters can be found in the `API section
    <https://sdv.dev/SDV/api_reference/tabular/api/sdv.tabular.copulas.GaussianCopula.sample.
    html>`__.


How do I specify constraints?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you look closely at the data you may notice that some properties were
not completely captured by the model. For example, you may have seen
that sometimes the model produces an ``experience_years`` number greater
than ``0`` while also indicating that ``work_experience`` is ``False``.
These types of properties are what we call ``Constraints`` and can also
be handled using ``SDV``. For further details about them please visit
the :ref:`single_table_constraints` guide.


Can I evaluate the Synthetic Data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A very common question when someone starts using **SDV** to generate
synthetic data is: *"How good is the data that I just generated?"*

In order to answer this question, **SDV** has a collection of metrics
and tools that allow you to compare the *real* that you provided and the
*synthetic* data that you generated using **SDV** or any other tool.

You can read more about this in the :ref:`evaluation` guide.
