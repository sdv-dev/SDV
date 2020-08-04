.. _sdv.metadata:

sdv.metadata
============

.. currentmodule:: sdv.metadata

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
   Metadata.get_foreign_key
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

Table Creation
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   Table
   Table.from_dict
   Table.from_json
   Table.set_model_kwargs
   Table.set_primary_key
   Table.to_dict
   Table.to_json

Table Navigation
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   Table.get_dtypes
   Table.get_fields
   Table.get_model_kwargs

Table Usage
~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   Table.filter_valid
   Table.fit
   Table.transform
   Table.reverse_transform
