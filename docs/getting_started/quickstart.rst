.. _quickstart:

Quickstart
==========

In this short tutorial we will guide you through a series of steps that
will help you getting started using **SDV**.

Model the dataset using SDV
---------------------------

To model a multi table, relational dataset, we follow two steps. In the
first step, we will load the data and configures the meta data. In the
second step, we will use the SDV API to fit and save a hierarchical
model. We will cover these two steps in this section using an example
dataset.

Load example data
~~~~~~~~~~~~~~~~~

**SDV** comes with a toy dataset to play with, which can be loaded using
the ``sdv.load_demo`` function:

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
    metadata.visualize('images/quickstart_1.png');
    metadata.visualize();

.. image:: /images/quickstart_1.png

For more details about how to build the ``Metadata`` for your own
dataset, please refer to the :ref:`relational_metadata` guide.

2. A dictionary containing three ``pandas.DataFrames`` with the tables
   described in the metadata object.

.. ipython:: python
    :okwarning:

    tables

Fit a model using the SDV API.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, we build a hierarchical statistical model of the data using
**SDV**. For this we will create an instance of the ``sdv.SDV`` class
and use its ``fit`` method.

During this process, **SDV** will traverse across all the tables in your
dataset following the primary key-foreign key relationships and learn
the probability distributions of the values in the columns.

.. ipython:: python
    :okwarning:

    from sdv import SDV

    sdv = SDV()
    sdv.fit(metadata, tables)

Sample data from the fitted model
---------------------------------

Once the modeling has finished you are ready to generate new synthetic
data using the ``sdv`` instance that you have.

For this, all you have to do is call the ``sample_all`` method from your
instance passing the number of rows that you want to generate:

.. ipython:: python
    :okwarning:

    sampled = sdv.sample_all()

This will return a dictionary identical to the ``tables`` one that we
passed to the SDV instance for learning, filled in with new synthetic
data.

.. note::

    Only the parent tables of your dataset will have the specified number of rows,
    as the number of child rows that each row in the parent table has is also sampled
    following the original distribution of your dataset.

.. ipython:: python
    :okwarning:

    sampled

Saving and Loading your model
-----------------------------

In some cases, you might want to save the fitted SDV instance to be able
to generate synthetic data from it later or on a different system.

In order to do so, you can save your fitted ``SDV`` instance for later
usage using the ``save`` method of your instance.

.. ipython:: python
    :okwarning:

    sdv.save('sdv.pkl')

The generated ``pkl`` file will not include any of the original data in
it, so it can be safely sent to where the synthetic data will be
generated without any privacy concerns.

Later on, in order to sample data from the fitted model, we will first
need to load it from its ``pkl`` file.

.. ipython:: python
    :okwarning:

    sdv = SDV.load('sdv.pkl')

After loading the instance, we can sample synthetic data using its
``sample_all`` method like before.

.. ipython:: python
    :okwarning:

    sampled = sdv.sample_all(5)

    sampled
