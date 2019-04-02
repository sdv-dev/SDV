.. highlight:: shell

============
Installation
============


From PyPi
---------

The simplest and recommended way to install SDV is using `pip`:

.. code-block:: console

    $ pip install sdv

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for SDV can be downloaded from the `Github repo`_.

You can either clone the ``stable`` branch form the public repository:

.. code-block:: console

    $ git clone --branch stable git://github.com/HDI-Project/SDV

Or download the `tarball`_:

.. code-block:: console

    $ curl  -OL https://github.com/HDI-Project/SDV/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ make install


.. _Github repo: https://github.com/HDI-Project/SDV
.. _tarball: https://github.com/HDI-Project/SDV/tarball/master


Development Setup
-----------------

If you want to make changes in `SDV` and contribute them, you will need to prepare
your environment to do so.

These are the required steps:

1. Fork the SDV `Github repo`_.

2. Clone your fork locally:

.. code-block:: console

    $ git clone git@github.com:your_name_here/SDV.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed,
   this is how you set up your fork for local development:

.. code-block:: console

    $ mkvirtualenv SDV
    $ cd SDV/
    $ make install-develop
