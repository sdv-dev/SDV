Relational Models
=================

The SDV Tabular Models are implemented in the subpackage `sdv.tabular`_.

BaseTabularModel
----------------

All the Tabular models in SDV inherit from the ``sdv.tabular.base.BaseTabularModel``.

The ``BaseTabularModel`` defines all the functionality that will be common across the different
Tabular Models, such as:

* Capturing all the user provided properties of the data that will be modeled.
* Building and keeping track of a ``Table`` metadata object using the given properties.
* Fit the ``Table`` object to the data to adapt the transformations, constraints and anonymization
  procedures to the actual values.
* Transforming the data before fitting the underlying models and reverting the transformations
  after these have sampled new data.
* Saving the models into a file using ``pickle`` and loading them back as instances afterwards.

The base ``__init__`` method implemented in the ``BaseTabularModel`` expects the following
arguments:


* ``field_names``:
    List of names of the fields that need to be modeled
    and included in the generated output data. Any additional
    fields found in the data will be ignored and will not be
    included in the generated output.
    If ``None``, all the fields found in the data are used.
* ``primary_key``:
    Specification about which field or fields are the
    primary key of the table and information about how
    to generate them.
* ``field_types``:
    Dictinary specifying the data types and subtypes
    of the fields that will be modeled. Field types and subtypes
    combinations must be compatible with the SDV Metadata Schema.
* ``anonymize_fields``:
    Dict specifying which fields to anonymize and what faker
    category they belong to.
* ``table_metadata``:
    Table metadata instance or dict representation.
    If given alongside any other metadata-related arguments, an
    exception will be raised.
    If not given at all, it will be built using the other
    arguments or learned from the data.

.. _sdv.tabular: /api_reference/tabular.html

