.. _sdgym_install:

Installing SDGym
================

Install with pip
----------------

The easiest and recommended way to install **SDGym** is using
`pip <https://pip.pypa.io/en/stable/>`__:

.. code:: bash

   pip install sdgym

This will pull and install the latest stable release from
`PyPi <https://pypi.org/>`__.

Install with conda
------------------

**SDGym** can also be installed using
`conda <https://docs.conda.io/en/latest/>`__:

.. code:: bash

   conda install -c sdv-dev -c conda-forge sdgym

This will pull and install the latest stable release from
`Anaconda <https://anaconda.org/>`__.

Install from source
-------------------

If you want to install **SDGym** from source you need to first clone the
repository and then execute the ``make install`` command inside the
``stable`` branch. Note that this command works only on Unix based
systems like GNU/Linux and macOS:

.. code:: bash

   git clone https://github.com/sdv-dev/SDGym
   cd SDGym
   git checkout stable
   make install

Install for development
-----------------------

If you intend to modify the source code or contribute to the project you
will need to install it from the source using the
``make install-develop`` command. In this case, we recommend you to
branch from ``master`` first:

.. code:: bash

   git clone git@github.com:sdv-dev/SDGym
   cd SDGym
   git checkout master
   git checkout -b <your-branch-name>
   make install-develp

For more details about how to contribute to the project please visit the
`Contributing Guide <CONTRIBUTING.rst>`__.

Compile C++ dependencies
------------------------

Some of the third party synthesizers that SDGym offers, like the
``PrivBN``, require dependencies written in C++ that need to be compiled
before they can be used.

In order to be able to use them, please do:

1. Clone or download the SDGym repository to your local machine:

.. code:: bash

   git clone git@github.com:sdv-dev/SDGym.git
   cd SDGym

2. make sure to have installed all the necessary dependencies to compile
   C++. In Linux distributions based on Ubuntu, this can be done with
   the following command:

.. code:: bash

   sudo apt-get install build-essential

3. Trigger the C++ compilation:

.. code:: bash

   make compile

4. Add the path to the created ``privBayes.bin`` binary to the
   ``PRIVBAYES_BIN`` environment variable:

.. code:: bash

   export PRIVBAYES_BIN=$(pwd)/privBayes.bin

Run with Docker
------------------

We support using Docker to run **SDGym**. For more information
on how to do so, check :ref:`benchmarking_docker`.