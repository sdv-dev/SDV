.. _install:

.. highlight:: shell

Installation
============

Requirements
------------

**SDV** has been tested and is supported on **GNU/Linux**, **macOS** and **Windows** systems running
`Python 3.6, 3.7 and 3.8`_ installed.

Also, although it is not strictly required, the usage of a `virtualenv`_ is highly recommended in
order to avoid having conflicts with other software installed in the system where **SDV** is run.

Install using pip
-----------------

The easiest and recommended way to install **SDV** is using `pip`_:

.. code-block:: console

    pip install sdv

This will pull and install the latest stable release from `PyPI`_.

.. warning:: When installing on windows systems, pip may complain about not being able to
   find a valid version for ``PyTorch``. In this case, please install ``PyTorch`` manually
   following the `PyTorch installation instructions`_ and retry installing ``sdv`` again
   afterwards.

Install from source
-------------------

The source code of **SDV** can be downloaded from the `Github repository`_

You can clone the repository and install with the following command in your terminal:

You can clone the repository and install it from source by running ``make install`` on the
``stable`` branch:

.. code-block:: console

    git clone git://github.com/sdv-dev/SDV
    cd SDV
    git checkout stable
    make install

.. note:: The ``master`` branch of the SDV repository contains the latest development version.
          If you want to install the latest stable version, make sure not to omit the
          ``git checkout stable`` indicated above.

If you are installing **SDV** in order to modify its code, the installation must be done
from its sources, in the editable mode, and also including some additional dependencies in
order to be able to run the tests and build the documentation. Instructions about this process
can be found in the :ref:`contributing` guide.

.. _Python 3.6, 3.7 and 3.8: https://docs.python-guide.org/starting/installation/
.. _WSL: https://docs.microsoft.com/en-us/windows/wsl/install-win10
.. _virtualenv: https://virtualenv.pypa.io/en/latest/
.. _pip: https://pip.pypa.io
.. _PyPI: https://pypi.org/
.. _Github repository: https://github.com/sdv-dev/SDV
.. _PyTorch installation instructions: https://pytorch.org/get-started/locally/
