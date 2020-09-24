HMA1 Class
==========

In this guide we will go through a series of steps that will let you
discover functionalities of the ``HMA1`` class.

What is HMA1?
-------------

The ``sdv.relational.HMA1`` class implements what is called a
*Hierarchical Modeling Algorithm* which is an algorithm that allows to
recursively walk through a relational dataset and apply tabular models
across all the tables in a way that lets the models learn how all the
fields from all the tables are related.

Let’s now discover how to use the ``HMA1`` class.

Quick Usage
-----------

We will start by loading and exploring one of our demo datasets.

.. ipython:: python
    :okwarning:

    from sdv import load_demo
    
    metadata, tables = load_demo(metadata=True)

This will return two objects:

1. A ``Metadata`` object with all the information that **SDV** needs to
   know about the dataset.

.. ipython:: python
    :okwarning:

    metadata

    @suppress
    metadata.visualize('images/hma1_1.png');
    metadata.visualize();

.. image:: /images/hma1_1.png


For more details about how to build the ``Metadata`` for your own
dataset, please refer to the :ref:`relational_metadata` Guide.

2. A dictionary containing three ``pandas.DataFrames`` with the tables
   described in the metadata object.

.. ipython:: python
    :okwarning:

    tables


Let us now use the ``HMA1`` class to learn this data to be ready to
sample synthetic data about new users. In order to do this you will need
to:

-  Import the ``sdv.relational.HMA1`` class and create an instance of it
   passing the ``metadata`` that we just loaded.
-  Call its ``fit`` method passing the ``tables`` dict.

.. ipython:: python
    :okwarning:

    from sdv.relational import HMA1
    
    model = HMA1(metadata)
    model.fit(tables)

.. note::

   During the previous steps SDV walked through all the tables in the
   dataset following the relationships specified by the metadata,
   learned each table using a :ref:`gaussian_copula` and then
   augmented the parent tables using the copula parameters before
   learning them. By doing this, each copula model was able to learn how
   the child table rows were related to their parent tables.

Generate synthetic data from the model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the training process has finished you are ready to generate new
synthetic data by calling the ``sample_all`` method from your model.

.. ipython:: python
    :okwarning:

    new_data = model.sample()

This will return a dictionary of tables identical to the one which the
model was fitted on, but filled with new data which resembles the
original one.

.. ipython:: python
    :okwarning:

    new_data


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

Let’s see how this process works.

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

   If you inspect the generated file you will notice that its size is
   much smaller than the size of the data that you used to generate it.
   This is because the serialized model contains **no information about
   the original data**, other than the parameters it needs to generate
   synthetic versions of it. This means that you can safely share this
   ``my_model.pkl`` file without the risc of disclosing any of your real
   data!

Load the model and generate new data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The file you just generated can be send over to the system where the
synthetic data will be generated. Once it is there, you can load it
using the ``HMA1.load`` method, and then you are ready to sample new
data from the loaded instance:

.. ipython:: python
    :okwarning:

    loaded = HMA1.load('my_model.pkl')
    new_data = loaded.sample()
    new_data.keys()


.. warning::

   Notice that the system where the model is loaded needs to also have
   ``sdv`` installed, otherwise it will not be able to load the model
   and use it.

How to control the number of rows?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the steps above we did not tell the model at any moment how many rows
we wanted to sample, so it produced as many rows as there were in the
original dataset.

If you want to produce a different number of rows you can pass it as the
``num_rows`` argument and it will produce the indicated number of rows:

.. ipython:: python
    :okwarning:

    model.sample(num_rows=5)


.. note::

   Notice that the root table ``users`` has the indicated number of rows
   but some of the other tables do not. This is because the number of
   rows from the child tables is sampled based on the values form the
   parent table, which means that only the root table of the dataset is
   affected by the passed ``num_rows`` argument.

Can I sample a subset of the tables?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In some occasions you will not be interested in generating rows for the
entire dataset and would rather generate data for only one table and its
children.

To do this you can simply pass the name of table that you want to
sample.

For example, pass the name ``sessions`` to the ``sample`` method, the
model will only generate data for the ``sessions`` table and its child
table, ``transactions``.

.. ipython:: python
    :okwarning:

    model.sample('sessions', num_rows=5)


If you want to further restrict the sampling process to only one table
and also skip its child tables, you can add the argument
``sample_children=False``.

For example, you can sample data from the table ``users`` only without
producing any rows for the tables ``sessions`` and ``transactions``.

.. ipython:: python
    :okwarning:

    model.sample('users', num_rows=5, sample_children=False)

.. note::

   In this case, since we are only producing a single table, the output
   is given directly as a ``pandas.DataFrame`` instead of a dictionary.
