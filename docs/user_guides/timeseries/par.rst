.. _par:

PAR Model
=========

In this guide we will go through a series of steps that will let you
discover functionalities of the ``PAR`` model for timeseries data.

What is PAR?
------------

The ``PAR`` class is an implementation of a Probabilistic AutoRegressive
model that allows learning **multi-type, multivariate timeseries data**
and later on generate new synthetic data that has the same format and
properties as the learned one.

Additionally, the ``PAR`` model has the ability to generate new synthetic
timeseries conditioned on the properties of the entity to which this
timeseries data would be associated.

.. note::

   The PAR model is under active development. Please use it, try it on
   your data and give us feedback on a `github
   issue <https://github.com/sdv-dev/SDV/issues>`__ or our `Slack
   workspace <https://join.slack.com/t/sdv-space/shared_invite/zt-gdsfcb5w-0QQpFMVoyB2Yd6SRiMplcw>`__

Quick Usage
-----------

We will start by loading one of our demo datasets, the
``nasdaq100_2019``, which contains daily stock marked data from the
NASDAQ 100 companies during the year 2019.

.. ipython:: python
    :okwarning:

    from sdv.demo import load_timeseries_demo

    data = load_timeseries_demo()
    data.head()

As you can see, this table contains information about multiple Tickers,
including:

-  Symbol of the Ticker.
-  Date associated with the stock market values.
-  The opening and closing prices for the day.
-  The Volume of transactions of the day.
-  The MarketCap of the company
-  The Sector and the Industry in which the company operates.

This data format is a very common and well known format for timeseries
data which includes 4 types of columns:

Entity Columns
~~~~~~~~~~~~~~

These are columns that indicate how the rows are associated with
external, abstract, ``entities``. The group of rows associated with each
``entity_id`` form a time series sequence, where order of the rows
matters and where inter-row dependencies exist. However, the rows of
different ``entities`` are completely independent from each other.

In this case, the external ``entity`` is the company, and the identifier
of the company within our data is the ``Symbol`` column.

.. ipython:: python
    :okwarning:

    entity_columns = ['Symbol']

.. note::

   In some cases, the datsets do not contain any ``entity_columns``
   because the rows are not associated with any external entity. In
   these cases, the ``entity_columns`` specification can be omitted and
   the complete dataset will be interpreted as a single timeseries
   sequence.

Context
~~~~~~~

The timeseries datasets may have one or more ``context_columns``.
``context_columns`` are variables that provide information about the
entities associated with the timeseries in the form of attributes and
which may condition how the timeseries variables evolve.

For example, in our stock market case, the ``MarketCap``, the ``Sector``
and the ``Industry`` variables are all contextual attributes associated
with each company and which have a great impact on what each timeseries
look like.

.. ipython:: python
    :okwarning:

    context_columns = ['MarketCap', 'Sector', 'Industry']

.. note::

   The ``context_columns`` are attributes that are associated with the
   entities, and which do not change over time. For this reason, since
   each timeseries sequence has a single entity associated, the values
   of the ``context_columns`` are expected to remain constant alongside
   each combination of ``entity_columns`` values.

Sequence Index
~~~~~~~~~~~~~~

By definition, the timeseries datasets have inter-row dependencies for
which the order of the rows matter. In most cases, this order will be
indicated by a ``sequence_index`` column that will contain sortable
values such as integers, floats or datetimes. In some other cases there
may be no ``sequence_index``, which means that the rows are assumed to
be already given in the right order.

In this case, the column that indicates us the order of the rows within
each sequence is the ``Date`` column:

.. ipython:: python
    :okwarning:

    sequence_index = 'Date'

Data Columns
~~~~~~~~~~~~

Finally, the rest of the columns of the dataset are what we call the
``data_columns``, and they are the columns that our ``PAR`` model will
learn to generate synthetically conditioned on the values of the
``context_columns``.

Let's now see how to use the ``PAR`` class to learn this timeseries
dataset and generate new synthetic timeseries that replicate its
properties.

For this, you will need to:

-  Import the ``sdv.timeseries.PAR`` class and create an instance of it
   passing the variables that we just created.
-  Call its ``fit`` method passing the timeseries data.
-  Call its ``sample`` method indicating the number of sequences that we
   want to generate.

.. ipython:: python
    :okwarning:

    from sdv.timeseries import PAR

    model = PAR(
        entity_columns=entity_columns,
        context_columns=context_columns,
        sequence_index=sequence_index,
    )
    model.fit(data)

.. note::

   Notice that the model ``fitting`` process took care of transforming
   the different fields using the appropriate `Reversible Data
   Transforms <http://github.com/sdv-dev/RDT>`__ to ensure that the data
   has a format that the underlying models can handle.

Generate synthetic data from the model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the modeling has finished you are ready to generate new synthetic
data by calling the ``sample`` method from your model passing the number
of the sequences that we want to generate.

Let’s start by generating a single sequence.

.. ipython:: python
    :okwarning:

    new_data = model.sample(1)

This will return a table identical to the one which the model was fitted
on, but filled with new synthetic data which resembles the original one.

.. ipython:: python
    :okwarning:

    new_data.head()

.. note::

   **Note**

   Notice how the model generated a random string for the ``Symbol``
   identifier which does not look like the regular Ticker symbols that
   we saw in the original data. This is because the model needs you to
   tell it how these symbols need to be generated by providing a regular
   expression that it can use. We will see how to do this in a later
   section.

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

.. note::

   If you inspect the generated file you will notice that its size is
   much smaller than the size of the data that you used to generate it.
   This is because the serialized model contains **no information about
   the original data**, other than the parameters it needs to generate
   synthetic versions of it. This means that you can safely share this
   ``my_model.pkl`` file without the risk of disclosing any of your real
   data!

Load the model and generate new data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The file you just generated can be sent over to the system where the
synthetic data will be generated. Once it is there, you can load it
using the ``PAR.load`` method, and then you are ready to sample new data
from the loaded instance:

.. ipython:: python
    :okwarning:

    loaded = PAR.load('my_model.pkl')
    loaded.sample(num_sequences=1).head()

.. warning::

   Notice that the system where the model is loaded needs to also have
   ``sdv`` installed, otherwise it will not be able to load the model
   and use it.

Conditional Sampling
~~~~~~~~~~~~~~~~~~~~

In the previous examples we had the model generate random values for use
to populate the ``context_columns`` and the ``entity_columns``. In order
to do this, the model learned the context and entity values using a
``GaussianCopula``, which later on was used to sample new realistic values
for them. This is fine for cases in which we do not have any constraints
regarding the type of data that we generate, but in some cases we might
want to control the values of the contextual columns to force the model
into generating data of a certain type.

In order to achieve this, we will first have to create a
``pandas.DataFrame`` with the expected values.

As an example, let’s generate values for two companies in the Technology
and Health Care sectors.

.. ipython:: python
    :okwarning:

    import pandas as pd

    context = pd.DataFrame([
        {
            'Symbol': 'AAAA',
            'MarketCap': 1.2345e+11,
            'Sector': 'Technology',
            'Industry': 'Electronic Components'
        },
        {
            'Symbol': 'BBBB',
            'MarketCap': 4.5678e+10,
            'Sector': 'Health Care',
            'Industry': 'Medical/Nursing Services'
        },
    ])
    context

Once you have created this, you can simply pass the dataframe as the
``context`` argument to the ``sample`` method.

.. ipython:: python
    :okwarning:

    new_data = model.sample(context=context)

And we can now see the data generated for the two companies:

.. ipython:: python
    :okwarning:

    new_data[new_data.Symbol == 'AAAA'].head()

.. ipython:: python
    :okwarning:

    new_data[new_data.Symbol == 'BBBB'].head()

Advanced Usage
--------------

Now that we have discovered the basics, let’s go over a few more
advanced usage examples and see the different arguments that we can pass
to our ``PAR`` Model in order to customize it to our needs.

How to customize the generated IDs?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the previous examples we saw how the ``Symbol`` values were generated
as random strings that do not look like those typically seen for
Tickers, which usually are strings made of between 2 and 4 uppercase
letters.

In order to fix this and force the model to generate values that are
valid for the field, we can use the ``field_types`` argument to indicate
the characteristics of each field by passing a dictionary that follows
the ``Metadata`` field specification.

For this case in particular, we will indicate that the ``Symbol`` field
needs to be generated using the regular expression ``[A-Z]{2,4}``.

.. ipython:: python
    :okwarning:

    field_types = {
        'Symbol': {
            'type': 'id',
            'subtype': 'string',
            'regex': '[A-Z]{2,4}'
        }
    }
    model = PAR(
        entity_columns=entity_columns,
        context_columns=context_columns,
        sequence_index=sequence_index,
        field_types=field_types
    )
    model.fit(data)

After this, we can observe how the new ``Symbols`` are generated as
indicated.

.. ipython:: python
    :okwarning:

    model.sample(num_sequences=1).head()

.. note::

   Notice how in this case we only specified the properties of the
   ``Symbol`` field and the PAR model was able to handle the other
   fields appropriately without needing any indication from us.

Can I control the length of the sequences?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When learning the data, the PAR model also learned the distribution of
the lengths of the sequences, so each generated sequence may have a
different length:

.. ipython:: python
    :okwarning:

    model.sample(num_sequences=5).groupby('Symbol').size()

If we want to force a specific length to the generated sequences we can
pass the ``sequence_length`` argument to the ``sample`` method:

.. ipython:: python
    :okwarning:

    model.sample(num_sequences=5, sequence_length=100).groupby('Symbol').size()

Can I use timeseries without context?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes the timeseries datasets do not provide any additional
properties from the entities associated with each sequence, other than
the unique identifier of the entity.

Let’s simulate this situation by dropping the context columns from our
data.

.. ipython:: python
    :okwarning:

    no_context = data[['Symbol', 'Date', 'Open', 'Close', 'Volume']].copy()
    no_context.head()

In this case, we can simply skip the context columns when creating the
model, and PAR will be able to learn the timeseries without imposing any
conditions to them.

.. ipython:: python
    :okwarning:

    model = PAR(
        entity_columns=entity_columns,
        sequence_index=sequence_index,
        field_types=field_types,
    )
    model.fit(no_context)
    model.sample(num_sequences=1).head()

In this case, of course, we are not able to sample new sequences
conditioned on any value, but we are still able to force the symbols
that we want on the generated data by passing them in a
``pandas.DataFrame``

.. ipython:: python
    :okwarning:

    symbols = pd.DataFrame({
        'Symbol': ['TSLA']
    })
    model.sample(context=symbols).head()

What happens if there are no ``entity_columns`` either?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In some cases the timeseries datasets are made of a single timeseries
sequence with no identifiers of external entities. For example, suppose
we only had the data from one company:

.. ipython:: python
    :okwarning:

    tsla = no_context[no_context.Symbol == 'TSLA'].copy()
    del tsla['Symbol']
    tsla.head()

In this case, we can simply omit the ``entity_columns`` argument when
creating our PAR instance:

.. ipython:: python
    :okwarning:

    model = PAR(
        sequence_index=sequence_index,
    )
    model.fit(tsla)
    model.sample()

