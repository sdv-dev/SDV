.. _single_table:

Single Table Data
=================

**SDV** supports modeling single table datasets. It provides unique
features for making it easy for the user to learn models and synthesize
datasets. Some important features of ``sdv.tabular`` include:

-  Support for tables with primary key

-  Support to anonymize certain fields like addresses, emails, phone
   numbers, names and other PII information.

-  Support for a number of different data types - categorical,
   numerical, discrete-ordinal and datetimes.

-  Support multiple types of statistical and deep learning models:

   -  GaussianCopula: A tool to model multivariate distributions using
      `copula
      functions <https://en.wikipedia.org/wiki/Copula_%28probability_theory%29>`__.
      Based on our `Copulas
      Library <https://github.com/sdv-dev/Copulas>`__.

   -  CTGAN: A GAN-based Deep Learning data synthesizer that can generate
      synthetic tabular data with high fidelity. Based on our `CTGAN
      Library <https://github.com/sdv-dev/CTGAN>`__.

.. toctree::
    :titlesonly:

    data_description
    models
    constraints
