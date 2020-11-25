.. _benchmarking_single_table:

Single Table Benchmarking
=========================

The benchmarking of synthetic data generators for single table datasets is
done using the `Synthetic Data Gym (SDGym) <https://github.com/sdv-dev/SDGym>`__,
a library from the Synthetic Data Vault  Project that offers a collection of both
real and simulated datasets to work on, *Machine Learning Efficacy* and *Bayesian
Likelihood* based metrics and a number of classical and novel synthetic data generators
to use as baselines to compare against.

.. note::

    The SDGym Library is not installed by default alongside `sdv`.
    If you want to use it, please install it using the ``pip install sdgym``
    command.

SDGym Synthesizers
------------------

**SDGym** evaluates the performance of **Synthesizers**.

A Synthesizer is a Python function (or class method) that takes as input
a numpy matrix with some data, which we call the *real* data, and
outputs another numpy matrix with the same shape, filled with new
*synthetic* data that has similar mathematical properties as the *real*
one.

Apart from the benchmark functionality and the SDV Tabular synthesizers, SDGym
implements a collection of Synthesizers which are either custom demo synthesizers
or re-implementations of synthesizers that have been presented in other publications.

More details about the implemented synthesizers can be found in the `SDGym Synthesizers
documentation <https://github.com/sdv-dev/SDGym/blob/master/SYNTHESIZERS.md>`__.

Benchmark datasets
------------------

**SDGym** evaluates the performance of **Synthetic Data Generators**
using datasets that are in three families:

-  Simulated data generated using Gaussian Mixtures
-  Simulated data generated using Bayesian Networks
-  Real world datasets

Further details about how these datasets were generated can be found in
the `Modeling Tabular data using Conditional
GAN <https://arxiv.org/abs/1907.00503>`__ paper and in the `SDGym datasets
documentation <https://github.com/sdv-dev/SDGym/blob/master/DATASETS.md>`__.

Current Leaderboard
-------------------

This is a summary of the current SDGym leaderboard showing the number
of datasets in which each Synthesizer obtained the best score.

Detailed leaderboard results for all the releases are available `in this
Google Docs
Spreadsheet <https://docs.google.com/spreadsheets/d/1iNJDVG_tIobcsGUG5Gn4iLa565vVhz2U/edit>`__.

Gaussian Mixture Simulated Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

============================== ===== ===== =====
Synthesizer                    0.2.2 0.2.1 0.2.0
============================== ===== ===== =====
CLBNSynthesizer                0     0.0   1.0
CTGAN                          0     N/E   N/E
CTGANSynthesizer               0     0.0   1.0
CopulaGAN                      0     N/E   N/E
GaussianCopulaCategorical      1     N/E   N/E
GaussianCopulaCategoricalFuzzy 0     N/E   N/E
GaussianCopulaOneHot           0     N/E   N/E
MedganSynthesizer              0     0.0   0.0
PrivBNSynthesizer              0     0.0   0.0
TVAESynthesizer                5     5.0   4.0
TableganSynthesizer            0     1.0   0.0
VEEGANSynthesizer              0     0.0   0.0
============================== ===== ===== =====

Bayesian Networks Simulated Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

============================== ===== ===== =====
Synthesizer                    0.2.2 0.2.1 0.2.0
============================== ===== ===== =====
CLBNSynthesizer                0     0.0   0.0
CTGAN                          0     N/E   N/E
CTGANSynthesizer               0     0.0   0.0
CopulaGAN                      0     N/E   N/E
GaussianCopulaCategorical      0     N/E   N/E
GaussianCopulaCategoricalFuzzy 0     N/E   N/E
GaussianCopulaOneHot           0     N/E   N/E
MedganSynthesizer              4     4.0   1.0
PrivBNSynthesizer              3     3.0   6.0
TVAESynthesizer                1     1.0   3.0
TableganSynthesizer            0     0.0   0.0
VEEGANSynthesizer              0     0.0   0.0
============================== ===== ===== =====

Real World Datasets
~~~~~~~~~~~~~~~~~~~

============================== ===== ===== =====
Synthesizer                    0.2.2 0.2.1 0.2.0
============================== ===== ===== =====
CLBNSynthesizer                0     0.0   0.0
CTGAN                          1     N/E   N/E
CTGANSynthesizer               0     3.0   3.0
CopulaGAN                      3     N/E   N/E
GaussianCopulaCategorical      0     N/E   N/E
GaussianCopulaCategoricalFuzzy 0     N/E   N/E
GaussianCopulaOneHot           0     N/E   N/E
MedganSynthesizer              0     0.0   0.0
PrivBNSynthesizer              0     0.0   0.0
TVAESynthesizer                4     5.0   5.0
TableganSynthesizer            0     0.0   0.0
VEEGANSynthesizer              0     0.0   0.0
============================== ===== ===== =====


Install
-------

The easiest and recommended way to install **SDGym** is using
`pip <https://pip.pypa.io/en/stable/>`__:

.. code:: bash

   pip install sdgym

This will pull and install the latest stable release from
`PyPi <https://pypi.org/>`__.

Usage
-----

Benchmarking your own synthesizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All you need to do in order to use the SDGym Benchmark, is to import
``sdgym`` and call its ``run`` function passing it your synthesizer
function and the settings that you want to use for the evaluation.

For example, if we want to evaluate a simple synthesizer function in the
``adult`` dataset we can execute:

.. code:: python3

   import numpy as np
   import sdgym

   def my_synthesizer_function(real_data, categorical_columns, ordinal_columns):
       """dummy synthesizer that just returns a permutation of the real data."""
       return np.random.permutation(real_data)

   scores = sdgym.run(synthesizers=my_synthesizer_function, datasets=['adult'])

The output of the ``sdgym.run`` function will be a ``pd.DataFrame``
containing the results obtained by your synthesizer on each dataset, as
well as the results obtained previously by the SDGym synthesizers:

::

                           adult/accuracy  adult/f1  ...  ring/test_likelihood
   IndependentSynthesizer         0.56530  0.134593  ...             -1.958888
   UniformSynthesizer             0.39695  0.273753  ...             -2.519416
   IdentitySynthesizer            0.82440  0.659250  ...             -1.705487
   ...                                ...       ...  ...                   ...
   my_synthesizer_function        0.64865  0.210103  ...             -1.964966

Benchmarking the SDGym Synthesizers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to run the SDGym benchmark on the SDGym Synthesizers you can
directly pass the corresponding class, or a list of classes, to the
``sdgym.run`` function.

For example, if you want to run the complete benchmark suite to evaluate
all the existing synthesizers you can run:

.. code:: python

   from sdgym.synthesizers import (
       CLBNSynthesizer, CTGANSynthesizer, IdentitySynthesizer, IndependentSynthesizer,
       MedganSynthesizer, PrivBNSynthesizer, TableganSynthesizer, TVAESynthesizer,
       UniformSynthesizer, VEEGANSynthesizer)

   all_synthesizers = [
       CLBNSynthesizer,
       IdentitySynthesizer,
       IndependentSynthesizer,
       MedganSynthesizer,
       PrivBNSynthesizer,
       TableganSynthesizer,
       CTGANSynthesizer,
       TVAESynthesizer,
       UniformSynthesizer,
       VEEGANSynthesizer,
   ]
   scores = sdgym.run(synthesizers=all_synthesizers)

.. warning:: This will take A LOT of time to run on a single machine!

For further details about all the arguments and possibilities that the
``benchmark`` function offers please refer to the `SDGym benchmark
documentation <https://github.com/sdv-dev/SDGym/blob/master/BENCHMARK.md>`__
