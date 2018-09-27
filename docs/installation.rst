.. highlight:: shell

============
Installation
============


From PyPi
---------

The simplest and recommended way to install SDV is using `pip`:

.. code-block:: console

    pip install sdv

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for SDV can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    git clone git://github.com/HDI-Project/sdv

Or download the `tarball`_:

.. code-block:: console

    curl  -OL https://github.com/HDI-Project/sdv/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    pip install .


.. _Github repo: https://github.com/HDI-Project/sdv
.. _tarball: https://github.com/HDI-Project/sdv/tarball/master


Development
-----------

If you are installing **SDV** in order to modify its code, the installation must be done
from its sources, in the editable mode, and also including some additional dependencies in
order to be able to run the tests and build the documentation:

.. code-block:: console

    pip install -e .[dev]