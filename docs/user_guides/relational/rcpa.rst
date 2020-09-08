.. _rcpa:

RCPA Model
==========

In this tutorial we will be showing how to model a real world
multi-table dataset using SDV.

About the datset
----------------

We have a store series, each of those have a size and a category and
additional information in a given date: average temperature in the
region, cost of fuel in the region, promotional data, the customer price
index, the unemployment rate and whether the date is a special holiday.

From those stores we obtained a training of historical data between
2010-02-05 and 2012-11-01. This historical data includes the sales of
each department on a specific date. In this notebook, we will show you
step-by-step how to download the "Walmart" dataset, explain the
structure and sample the data.

In this demonstration we will show how SDV can be used to generate
synthetic data. And lately, this data can be used to train machine
learning models.

*The dataset used in this example can be found in
`Kaggle <https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data>`__,
but we will show how to download it from SDV.*

Data model summary
~~~~~~~~~~~~~~~~~~

.. raw:: html

   <p style="text-align: center">

stores

.. raw:: html

   <p>

+---------+---------------+-----------+-------------------------+
| Field   | Type          | Subtype   | Additional Properties   |
+=========+===============+===========+=========================+
| Store   | id            | integer   | Primary key             |
+---------+---------------+-----------+-------------------------+
| Size    | numerical     | integer   |                         |
+---------+---------------+-----------+-------------------------+
| Type    | categorical   |           |                         |
+---------+---------------+-----------+-------------------------+

Contains information about the 45 stores, indicating the type and size
of store.

.. raw:: html

   <p style="text-align: center">

features

.. raw:: html

   <p>

+----------------+-------------+-----------+------------------------------+
| Fields         | Type        | Subtype   | Additional Properties        |
+================+=============+===========+==============================+
| Store          | id          | integer   | foreign key (stores.Store)   |
+----------------+-------------+-----------+------------------------------+
| Date           | datetime    |           | format: "%Y-%m-%d"           |
+----------------+-------------+-----------+------------------------------+
| IsHoliday      | boolean     |           |                              |
+----------------+-------------+-----------+------------------------------+
| Fuel\_Price    | numerical   | float     |                              |
+----------------+-------------+-----------+------------------------------+
| Unemployment   | numerical   | float     |                              |
+----------------+-------------+-----------+------------------------------+
| Temperature    | numerical   | float     |                              |
+----------------+-------------+-----------+------------------------------+
| CPI            | numerical   | float     |                              |
+----------------+-------------+-----------+------------------------------+
| MarkDown1      | numerical   | float     |                              |
+----------------+-------------+-----------+------------------------------+
| MarkDown2      | numerical   | float     |                              |
+----------------+-------------+-----------+------------------------------+
| MarkDown3      | numerical   | float     |                              |
+----------------+-------------+-----------+------------------------------+
| MarkDown4      | numerical   | float     |                              |
+----------------+-------------+-----------+------------------------------+
| MarkDown5      | numerical   | float     |                              |
+----------------+-------------+-----------+------------------------------+

Contains historical training data, which covers to 2010-02-05 to
2012-11-01.

.. raw:: html

   <p style="text-align: center">

depts

.. raw:: html

   <p>

+-----------------+-------------+-----------+-------------------------------+
| Fields          | Type        | Subtype   | Additional Properties         |
+=================+=============+===========+===============================+
| Store           | id          | integer   | foreign key (stores.Stores)   |
+-----------------+-------------+-----------+-------------------------------+
| Date            | datetime    |           | format: "%Y-%m-%d"            |
+-----------------+-------------+-----------+-------------------------------+
| Weekly\_Sales   | numerical   | float     |                               |
+-----------------+-------------+-----------+-------------------------------+
| Dept            | numerical   | integer   |                               |
+-----------------+-------------+-----------+-------------------------------+
| IsHoliday       | boolean     |           |                               |
+-----------------+-------------+-----------+-------------------------------+

Contains additional data related to the store, department, and regional
activity for the given dates.

Load relational data
--------------------

Let's start downloading the data set. In this case, we will download the
data set *walmart*. We will use the SDV function ``load_demo``, we can
specify the name of the dataset we want to use and if we want its
Metadata object or not. To know more about the ``load_demo`` function
`see its
documentation <https://sdv-dev.github.io/SDV/api_reference/api/sdv.demo.load_demo.html>`__.

.. ipython:: python
    :okwarning:

    from sdv import load_demo

    metadata, tables = load_demo(dataset_name='walmart', metadata=True)

Our dataset is downloaded from an `Amazon S3
bucket <http://sdv-datasets.s3.amazonaws.com/index.html>`__ that
contains all available data sets of the ``load_demo`` method.

We can now visualize the metadata structure and the tables

.. ipython:: python
    :okwarning:

    @suppress
    metadata.visualize('images/rcpa_1.png');
    metadata.visualize();

.. image:: /images/rcpa_1.png


.. ipython:: python
    :okwarning:

    tables

And also validate that the metadata is correctly defined for our data

.. ipython:: python
    :okwarning:

    metadata.validate(tables)


Model the data with SDV
-----------------------

Once we download it, we have to create an SDV instance. With that
instance, we have to analyze the loaded tables to generate a statistical
model from the data. In this case, the process of adjusting the model is
quickly because the dataset is small. However, with larger datasets it
can be a slow process.

.. ipython:: python
    :okwarning:

    from sdv import SDV

    sdv = SDV()
    sdv.fit(metadata, tables=tables)

Note: We may not want to train the model every time we want to generate
new synthetic data. We can
`save <https://sdv-dev.github.io/SDV/api/sdv.sdv.html#sdv.sdv.SDV.save>`__
the SDV instance to
`load <https://sdv-dev.github.io/SDV/api/sdv.sdv.html#sdv.sdv.SDV.save>`__
it later.

Generate synthetic data
-----------------------

Once the instance is trained, we are ready to generate the synthetic
data.

The easiest way to generate synthetic data for the entire dataset is to
call the ``sample_all`` method. By default, this method generates only 5
rows, but we can specify the row number that will be generated with the
``num_rows`` argument.

.. ipython:: python
    :okwarning:

    samples = sdv.sample_all()

This returns a dictionary with the same format as the input ``tables``,
with a ``pandas.DataFrame`` for each table.

.. ipython:: python
    :okwarning:

    samples


We may not want to generate data for all tables in the dataset, rather
for just one table. This is possible with SDV using the ``sample``
method. To use it we only need to specify the name of the table we want
to synthesize and the row numbers to generate. In this case, the
"walmart" data set has 3 tables: stores, features and depts.

In the following example, we will generate 1000 rows of the "features"
table.

.. ipython:: python
    :okwarning:

    sdv.sample('features', 1000, sample_children=False)
