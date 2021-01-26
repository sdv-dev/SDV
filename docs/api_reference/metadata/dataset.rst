.. _sdv.metadata.dataset:

sdv.metadata.dataset
====================

.. currentmodule:: sdv.metadata.dataset

Metadata Creation
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   Metadata
   Metadata.add_field
   Metadata.add_relationship
   Metadata.add_table
   Metadata.set_primary_key
   Metadata.validate
   Metadata.visualize
   Metadata.to_dict
   Metadata.to_json

Metadata Navigation
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   Metadata.get_children
   Metadata.get_dtypes
   Metadata.get_field_meta
   Metadata.get_fields
   Metadata.get_foreign_keys
   Metadata.get_parents
   Metadata.get_primary_key
   Metadata.get_table_meta
   Metadata.get_tables

Metadata Usage
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   Metadata.transform
   Metadata.reverse_transform
   Metadata.load_table
   Metadata.load_tables
