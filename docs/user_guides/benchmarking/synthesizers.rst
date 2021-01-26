.. _sdgym_synthesizers:

Synthesizer functions
=====================

**SDGym** evaluates the performance of **Synthetic Data Generators**,
also called **Synthesizers**.

A Synthesizer is a Python function (or class method) that takes as input
a ``dict`` with table names and ``pandas.DataFrame`` instances, which we
call the *real* data, and outputs another ``dict`` with the same shape
entries and new ``pandas.DataFrame`` instances, filled with new
*synthetic* data that has the same format and mathematical properties as
the *real* data.

The complete list of inputs of the synthesizer is:

-  ``real_data``: a ``dict`` containing table names as keys and
   ``pandas.DataFrame`` instances as values.
-  ``metadata``: an instance of an ``sdv.Metadata`` with information
   about the dataset.

And the output is a new ``dict`` with the same tables that the
``real_data`` contains.

.. code:: python

   def synthesizer_function(real_data: dict[str, pandas.DataFrame],
                            metadata: sdv.Metadata) -> real_data: dict[str, pandas.DataFrame]:
       ...
       # do all necessary steps to learn from the real data
       # and produce new synthetic data that resembles it
       ...
       return synthetic_data

SDGym Synthesizers
------------------

Apart from the benchmark functionality, SDGym implements a collection of
Baseline Synthesizers which are either trivial baseline synthesizers or
integrations of synthesizers found in other libraries.

These Synthesizers are written as Python classes that can be imported
from the ``sdgym.synthesizers`` module and have a ``fit_sample`` method
with the signature indicated above, which can be directly passed to the
``sdgym.run`` function to benchmark them.
