The Synthetic Data Vault
========================

The Synthetic Data Vault is built as a collection of libraries which provide different
functionalities, from simple data transformation to complex state-of-the-art Generative
Deep Learning models.

SDV
---

* `Documentation <../index.html>`_
* `Source Code <https://github.com/sdv-dev/SDV>`_

SDV is the main library, which provides a user friendly interface to access all the
different functionalities of the project.

The SDV library allows you to:

* Model single tables using Copulas and Deep Learning based models.
* Model complex multi-table relational datasets using Copulas and unique recursive
  modeling techniques.
* Handle multiple data types and missing data with minimum user input.
* Use and define custom inter-column constraints and data validation rules.
* Define entire datasets with a custom and flexible Metadata JSON schema.

Development Guides
~~~~~~~~~~~~~~~~~~

* `Tabular Models <sdv/tabular.html>`_
* `Constraints <sdv/constraints.html>`_
* `Metadata JSON <sdv/metadata.html>`_


RDT
---

* `Documentation <https://sdv-dev.github.io/RDT/>`_
* `Source Code <https://github.com/sdv-dev/RDT>`_

RDT is a Python library used to transform data for data science libraries and preserve the
transformations in order to revert them as needed.

It is organized around the objects called `Transformers`, which implement a very simple and
familiar API with 4 methods:

- ``fit``: Learn the properties of the data.
- ``transform``: Transform the data.
- ``fit_transform``: Fit the transformer to the data and then transform it.
- ``reverse_transform``: Revert a previous transformation to go back to the original format.


* Getting Started
* Developer Guides
* Source Code


Copulas
-------

* `Documentation <https://sdv-dev.github.io/Copulas/>`_
* `Source Code <https://github.com/sdv-dev/Copulas>`_

Copulas is a Python library for modeling multivariate distributions and sampling from them using
copula functions. Given a table containing numerical data, we can use Copulas to learn the
distribution and later on generate new synthetic rows following the same statistical properties.

Some of the features provided by this library include:

* A variety of distributions for modeling univariate data.
* Multiple Archimedean copulas for modeling bivariate data.
* Gaussian and Vine copulas for modeling multivariate data.
* Automatic selection of univariate distributions and bivariate copulas.


CTGAN
-----

* `Documentation <https://sdv-dev.github.io/CTGAN/>`_
* `Source Code <https://github.com/sdv-dev/CTGAN>`_

CTGAN is a GAN-based data synthesizer that can generate synthetic tabular data with high fidelity.

It was presented in NeurIPS 2020 in the paper `Modeling Tabular data using Conditional GAN`_.

.. toctree::
    :hidden:

    sdv/index


.. _Modeling Tabular data using Conditional GAN: https://arxiv.org/abs/1907.00503
