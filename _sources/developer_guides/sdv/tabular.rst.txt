.. _developer_tabular_models:

Tabular Models
==============

SDV has all the necessary tools to model and sample data from individual tables with a collection
of tabular models, such as ``Copulas`` and ``CTGAN``, while keeping all the complexity associated
with their usage hidden away and offering a very simple and intuitive interface to the users.

All this functionalities are available in the ``sdv.tabular.BaseTabularModel``, which is inherited
and used by all the models implemented in the :ref:`sdv.tabular` sub-package.

Tabular Modeling Overview
-------------------------

The ``BaseTabularModel`` class offers a very simple API with only two methods, ``fit`` and
``sample``, which handle the processes of preparing the data for modeling, calling an underlying
model instance to learn the data and later on sample it, and finally post-processing the generated
data to make it as similar to the original one as possible.

A regular `modeling` and `sampling` cycle performed within one of the ``BaseTabularModel``
subclasses has the following steps:

1. An instance of the ``BaseTabularModel`` subclass is created. In this step, the user can either
   pass information about the data, such as field types or constraints, or a fully configured Table
   metadata object.
2. The ``BaseTabularModel.fit`` method is called with raw data from the table.
   During this step, the ``BaseTabularModel`` will:

   a. Fit the Table metadata object, unless it has already been fitted before.
   b. Store the number of rows that exist in the given data.
   c. Use the Table metadata to transform the data into numerical data, ready for the model.
      If the Table had any constraints defined using the ``transform`` strategy, this will also
      apply transformations to the data.
   d. Call the ``self._fit`` method, which has been implemented in the subclass, passing the
      numerical data.

3. The subclass ``_fit`` method will then fit an instance of the underlying model class. For
   example, an instance of a ``copulas.multivariate.GaussianCopula`` will be created and fitted
   to the numerical data.

4. In a later step, the ``BaseTabularModel.sample`` method will be called, optionally with an
   indication of the number of rows to sample.
   During this step, the ``BaseTabularModel`` will:

   a. Decide a number of rows to sample, which will either be the number of rows provided
      or the number of rows that the original table had.
   b. Call the ``self._sample`` method implemented by the subclass, which will use the
      underlying model to generate the indicated number of rows.
   c. Use the Table metadata to transform the sampled data back to the original format by passing
      the data to its ``revert_transform`` method. This will also revert any transformations
      performed by the Constraints that use the ``transform`` strategy.
   d. If there is any Constraint that is using the ``reject_sampling`` strategy, use the
      Table metadata to drop the invalid rows and repeat steps ``b`` and ``c`` until enough
      valid rows have been generated.

A part from the previous steps, the ``BaseTabularModel`` also offers a couple of minor
functionalities:

* ``get_metadata``: Returns the Table metadata object that has been fitted to the data.
* ``save``: Saves the complete Tabular Model in a file using ``pickle``.
* ``load``: Loads a previously saved model from a ``pickle`` file.

Implementing a Tabular Model
----------------------------

In order to implement a new Tabular Model, all you need is to create a class that inherits from
``dsv.tabular.base.BaseTabularModel`` and implement at least these two methods:

* ``_fit``: Gets clean numerical data as input and fits an underlying model.
* ``_sample``: Samples the indicated number of rows from the fitted model.

BaseTabularModel Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~

When creating a subclass we will probably want to capture some arguments to the ``__init__``
method.

The base ``__init__`` method implemented in the ``BaseTabularModel`` expects the following
arguments:

* ``field_names``:
  List of names of the fields that need to be modeled
  and included in the generated output data. Any additional
  fields found in the data will be ignored and will not be
  included in the generated output.
  If ``None``, all the fields found in the data are used.
* ``field_types``:
  Dictionary specifying the data types and subtypes
  of the fields that will be modeled. Field types and subtypes
  combinations must be compatible with the SDV Metadata Schema.
* ``field_transformers``:
  Dictionary specifying which transformers to use for each field.
  Available transformers are:

    * ``integer``: Uses a ``NumericalTransformer`` of dtype ``int``.
    * ``float``: Uses a ``NumericalTransformer`` of dtype ``float``.
    * ``categorical``: Uses a ``CategoricalTransformer`` without gaussian noise.
    * ``categorical_fuzzy``: Uses a ``CategoricalTransformer`` adding gaussian noise.
    * ``one_hot_encoding``: Uses a ``OneHotEncodingTransformer``.
    * ``label_encoding``: Uses a ``LabelEncodingTransformer``.
    * ``boolean``: Uses a ``BooleanTransformer``.
    * ``datetime``: Uses a ``DatetimeTransformer``.

* ``anonymize_fields``:
  Dict specifying which fields to anonymize and what faker
  category they belong to.
* ``primary_key``:
  Name of the field which is the primary key of the table.
* ``constraints``:
  List of Constraint objects or dicts.
* ``table_metadata``:
  Table metadata instance or dict representation.
  If given alongside any other metadata-related arguments, an
  exception will be raised.
  If not given at all, it will be built using the other
  arguments or learned from the data.

Subclasses can extend this list by adding their own arguments, or even simply implement their own
``__init__`` method. However, capturing these explicitly and passing them to the
``super().__init__`` method is the recommended way to initialize a ``BaseTabularModel`` subclass.

We can see such implementation in the ``__init__`` method of the ``sdv.tabular.ctgan.CTGAN`` model,
which adds a few arguments to the class but still captures all the other arguments and calls the
``super().__init__`` method with them:

.. code-block:: Python

    def __init__(self, field_names=None, field_types=None, field_transformers=None,
                 anonymize_fields=None, primary_key=None, constraints=None, table_metadata=None,
                 epochs=300, log_frequency=True, embedding_dim=128, generator_dim=(256, 256),
                 discriminator_dim=(256, 256), l2scale=1e-6, batch_size=500):
        super().__init__(
            field_names=field_names,
            primary_key=primary_key,
            field_types=field_types,
            anonymize_fields=anonymize_fields,
            constraints=constraints,
            table_metadata=table_metadata
        )
        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim
        self._l2scale = l2scale
        self._batch_size = batch_size
        self._epochs = epochs
        self._log_frequency = log_frequency

By doing these, not only can the ``CTGAN`` take advantage of all the functionalities from the base
class, but also the signature and API reference exposes all the accepted arguments appropriately.

_fit method
~~~~~~~~~~~

The ``_fit`` method only expects one argument called ``table_data``, which is a
``pandas.DataFrame`` that contains numerical data only.

Within this method, you can perform any steps necessary to fit your model.
For example, we can see how the ``sdv.tabular.ctgan.CTGAN._fit`` method creates an instance of
the underlying model, ``CTGANSynthesizer``, and prepares the list of categorical columns that
it expects alongside the data.

.. code-block:: Python

    def _fit(self, table_data):
        """Fit the model to the table.

        Args:
            table_data (pandas.DataFrame):
                Data to be learned.
        """
        self._model = self._CTGAN_CLASS(
            embedding_dim=self._embedding_dim,
            generator_dim=self._generator_dim,
            discriminator_dim=self._discriminator_dim,
            l2scale=self._l2scale,
            batch_size=self._batch_size,
        )
        categoricals = [
            field
            for field, meta in self._metadata.get_fields().items()
            if meta['type'] == 'categorical'
        ]
        self._model.fit(
            table_data,
            epochs=self._epochs,
            discrete_columns=categoricals,
            log_frequency=self._log_frequency,
        )

.. note:: Here you can also see that some of the hyperparameters for the ``CTGANSynthesizer``
          class are being taken from the instance itself, where the ``__init__`` method stored
          them beforehand.

_sample method
~~~~~~~~~~~~~~

The ``_sample`` method only expects one argument called ``num_rows``, which is an integer that
indicates the number of rows that need to be sampled. In most cases, such as the ``CTGAN`` example
shown below, all this method does is call the ``sample`` method of the underlying model:

.. code-block:: Python

    def _sample(self, num_rows):
        """Sample the indicated number of rows from the model.

        Args:
            num_rows (int):
                Amount of rows to sample.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        return self._model.sample(num_rows)
