.. _ctgan:

CTGAN Model
===========

In this guide we will go through a series of steps that will let you
discover functionalities of the ``CTGAN`` model, including how to:

-  Create an instance of ``CTGAN``.
-  Fit the instance to your data.
-  Generate synthetic versions of your data.
-  Use ``CTGAN`` to anonymize PII information.
-  Customize the data transformations to improve the learning process.
-  Specify hyperparameters to improve the output quality.

What is CTGAN?
--------------

The ``sdv.tabular.CTGAN`` model is based on the GAN-based Deep Learning
data synthesizer which was presented at the NeurIPS 2020 conference by
the paper titled `Modeling Tabular data using Conditional
GAN <https://arxiv.org/abs/1907.00503>`__.

Let's now discover how to learn a dataset and later on generate
synthetic data with the same format and statistical properties by using
the ``CTGAN`` class from SDV.

Quick Usage
-----------

We will start by loading one of our demo datasets, the
``student_placements``, which contains information about MBA students
that applied for placements during the year 2020.

.. warning::

    In order to follow this guide you need to have ``ctgan`` installed on
    your system. If you have not done it yet, please install ``ctgan`` now
    by executing the command ``pip install sdv`` in a terminal.

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

Let us use ``CTGAN`` to learn this data and then sample synthetic data
about new students to see how well the model captures the characteristics
indicated above. In order to do this you will need to:

-  Import the ``sdv.tabular.CTGAN`` class and create an instance of it.
-  Call its ``fit`` method passing our table.
-  Call its ``sample`` method indicating the number of synthetic rows
   that you want to generate.

.. ipython:: python
    :okwarning:

    from sdv.tabular import CTGAN

    model = CTGAN()
    model.fit(data)

.. note::

    Notice that the model ``fitting`` process took care of transforming the
    different fields using the appropriate `Reversible Data
    Transforms <http://github.com/sdv-dev/RDT>`__ to ensure that the data
    has a format that the underlying CTGANSynthesizer class can handle.

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
    ``my_model.pkl`` file without the risc of disclosing any of your real
    data!

Load the model and generate new data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The file you just generated can be sent over to the system where the
synthetic data will be generated. Once it is there, you can load it
using the ``CTGAN.load`` method, and then you are ready to sample new
data from the loaded instance:

.. ipython:: python
    :okwarning:

    loaded = CTGAN.load('my_model.pkl')
    new_data = loaded.sample(200)

.. warning::

    Notice that the system where the model is loaded needs to also have
    ``sdv`` and ``ctgan`` installed, otherwise it will not be able to load
    the model and use it.

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

    model = CTGAN(
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

    model = CTGAN(
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

    model = CTGAN(
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
to our ``CTGAN`` Model in order to customize it to our needs.

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

    model = CTGAN(
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

How to modify the CTGAN Hyperparameters?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A part from the common Tabular Model arguments, ``CTGAN`` has a number
of additional hyperparameters that control its learning behavior and can
impact on the performance of the model, both in terms of quality of the
generated data and computational time.

-   ``epochs`` and ``batch_size``: these arguments control the number of
    iterations that the model will perform to optimize its parameters,
    as well as the number of samples used in each step. Its default
    values are ``300`` and ``500`` respectively, and ``batch_size`` needs
    to always be a value which is multiple of ``10``.

    These hyperparameters have a very direct effect in time the training
    process lasts but also on the performance of the data, so for new
    datasets, you might want to start by setting a low value on both of
    them to see how long the training process takes on your data and later
    on increase the number to acceptable values in order to improve the
    performance.

-   ``log_frequency``: Whether to use log frequency of categorical levels
    in conditional sampling. It defaults to ``True``.
    This argument affects how the model processes the frequencies of the
    categorical values that are used to condition the rest of the values.
    In some cases, changing it to ``False`` could lead to better performance.

-   ``embedding_dim`` (int): Size of the random sample passed to the
    Generator. Defaults to 128.

-   ``generator_dim`` (tuple or list of ints): Size of the output samples for
    each one of the Residuals. A Resiudal Layer will be created for each
    one of the values provided. Defaults to (256, 256).

-   ``discriminator_dim`` (tuple or list of ints): Size of the output samples for
    each one of the Discriminator Layers. A Linear Layer will be created
    for each one of the values provided. Defaults to (256, 256).

-   ``generator_lr`` (float): Learning rate for the generator. Defaults to 2e-4.

-   ``generator_decay`` (float): Generator weight decay for the Adam Optimizer.
    Defaults to 1e-6.

-   ``discriminator_lr`` (float): Learning rate for the discriminator.
    Defaults to 2e-4.

-   ``discriminator_decay`` (float): Discriminator weight decay for the Adam
    Optimizer. Defaults to 1e-6.

-   ``discriminator_steps`` (int): Number of discriminator updates to do for
    each generator update. From the WGAN paper: https://arxiv.org/abs/1701.07875.
    WGAN paper default is 5. Default used is 1 to match original CTGAN
    implementation.

-   ``verbose``: Whether to print fit progress on stdout. Defaults to ``False``.

-   ``cuda`` (bool or str): If ``True``, use CUDA. If a ``str``, use the
    indicated device. If ``False``, do not use cuda at all.

.. warning::

    Notice that the value that you set on the ``batch_size`` argument must always be a
    multiple of ``10``!

As an example, we will try to fit the ``CTGAN`` model slightly
increasing the number of epochs, reducing the ``batch_size``, adding one
additional layer to the models involved and using a smaller wright
decay.

Before we start, we will evaluate the quality of the previously
generated data using the ``sdv.evaluation.evaluate`` function

.. ipython:: python
    :okwarning:

    from sdv.evaluation import evaluate

    evaluate(new_data, data)


Afterwards, we create a new instance of the ``CTGAN`` model with the
hyperparameter values that we want to use

.. ipython:: python
    :okwarning:

    model = CTGAN(
        primary_key='student_id',
        epochs=500,
        batch_size=100,
        generator_dim=(256, 256, 256),
        discriminator_dim=(256, 256, 256)
    )

And fit to our data.

.. ipython:: python
    :okwarning:

    model.fit(data)

Finally, we are ready to generate new data and evaluate the results.

.. ipython:: python
    :okwarning:

    new_data = model.sample(len(data))
    evaluate(new_data, data)


As we can see, in this case these modifications changed the obtained
results slightly, but they did neither introduce dramatic changes in the
performance.

Conditional Sampling
~~~~~~~~~~~~~~~~~~~~

As the name implies, conditional sampling allows us to sample from a conditional
distribution using the ``CTGAN`` model, which means we can generate only values that
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


``CTGAN`` also supports conditioning on continuous values, as long as the values
are within the range of seen numbers. For example, if all the values of the
dataset are within 0 and 1, ``CTGAN`` will not be able to set this value to 1000.

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
    <https://sdv.dev/SDV/api_reference/tabular/api/sdv.tabular.ctgan.CTGAN.sample.
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
