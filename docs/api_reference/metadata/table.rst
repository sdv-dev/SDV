.. _sdv.metadata.table:

sdv.metadata.table
==================

.. currentmodule:: sdv.metadata.table

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
