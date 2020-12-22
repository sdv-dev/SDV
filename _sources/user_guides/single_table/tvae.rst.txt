.. _tvae:

TVAE Model
===========

In this guide we will go through a series of steps that will let you
discover functionalities of the ``TVAE`` model, including how to:

-  Create an instance of ``TVAE``.
-  Fit the instance to your data.
-  Generate synthetic versions of your data.
-  Use ``TVAE`` to anonymize PII information.
-  Customize the data transformations to improve the learning process.
-  Specify hyperparameters to improve the output quality.

What is TVAE?
--------------

The ``sdv.tabular.TVAE`` model is based on the VAE-based Deep Learning
data synthesizer which was presented at the NeurIPS 2020 conference by
the paper titled `Modeling Tabular data using Conditional
GAN <https://arxiv.org/abs/1907.00503>`__.

Let's now discover how to learn a dataset and later on generate
synthetic data with the same format and statistical properties by using
the ``TVAE`` class from SDV.

Quick Usage
-----------

We will start by loading one of our demo datasets, the
``student_placements``, which contains information about MBA students
that applied for placements during the year 2020.

.. warning::

    In order to follow this guide you need to have ``tvae`` installed on
    your system. If you have not done it yet, please install ``tvae`` now
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
-  The salary that they where offered
-  The duration and dates of their placement

You will notice that there is data with the following characteristics:

-  There are float, integer, boolean, categorical and datetime values.
-  There are some variables that have missing data. In particular, all
   the data related to the placement details is missing in the rows
   where the student was not placed.

Let us use ``TVAE`` to learn this data and then sample synthetic data
about new students to see how well de model captures the characteristics
indicated above. In order to do this you will need to:

-  Import the ``sdv.tabular.TVAE`` class and create an instance of it.
-  Call its ``fit`` method passing our table.
-  Call its ``sample`` method indicating the number of synthetic rows
   that you want to generate.

.. ipython:: python
    :okwarning:

    from sdv.tabular import TVAE

    model = TVAE()
    model.fit(data)

.. note::

    Notice that the model ``fitting`` process took care of transforming the
    different fields using the appropriate `Reversible Data
    Transforms <http://github.com/sdv-dev/RDT>`__ to ensure that the data
    has a format that the underlying TVAESynthesizer class can handle.

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

The file you just generated can be send over to the system where the
synthetic data will be generated. Once it is there, you can load it
using the ``TVAE.load`` method, and then you are ready to sample new
data from the loaded instance:

.. ipython:: python
    :okwarning:

    loaded = TVAE.load('my_model.pkl')
    new_data = loaded.sample(200)

.. warning::

    Notice that the system where the model is loaded needs to also have
    ``sdv`` and ``tvae`` installed, otherwise it will not be able to load
    the model and use it.

Specifying the Primary Key of the table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One of the first things that you may have noticed when looking that demo
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

    model = TVAE(
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

    model = TVAE(
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

    model = TVAE(
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



How do I specify constraints?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you look closely at the data you may notice that some properties were
not completely captured by the model. For example, you may have seen
that sometimes the model produces an ``experience_years`` number greater
than ``0`` while also indicating that ``work_experience`` is ``False``.
These type of properties are what we call ``Constraints`` and can also
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
