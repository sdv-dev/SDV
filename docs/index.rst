Introduction
============

The goal of the Synthetic Data Vault (SDV) is to allow data scientists to navigate, model and
sample relational databases. The main access point of the library is  the class `SDV`, that wraps
the functionality of the three core classes: the `DataNavigator`, the `Modeler` and the `Sampler`.

Using these classes, users can get easy access to information about the relational database,
create generative models for tables in the database and sample rows from these models to produce
synthetic data.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   self
   installation
   quickstart

.. toctree::
   :caption: Advanced Usage
   :maxdepth: 3

   usage

.. toctree::
   :caption: References
   :titlesonly:

   api_reference
   contributing
   authors
   history

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
