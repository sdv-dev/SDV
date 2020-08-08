.. raw:: html

   <p align="left">
   <img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt="DAI-Lab" />
   <i>An open source project from Data to AI Lab at MIT.</i>
   </p>

|Development Status| |PyPi Shield| |Travis CI Shield| |Coverage Status|
|Downloads| |Binder| |Slack|

SDV - The Synthetic Data Vault
==============================

**Date**: |today| **Version**: |version|

-  License: `MIT <https://github.com/sdv-dev/SDV/blob/master/LICENSE>`__
-  Development Status:
   `Pre-Alpha <https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha>`__
-  Documentation: https://sdv-dev.github.io/SDV
-  Homepage: https://github.com/sdv-dev/SDV

Overview
--------

The Synthetic Data Vault (SDV) is a tool that allows users to
statistically model tabular as well as entire multi-table, relational
datasets. Users can then use the statistical model to generate a
synthetic dataset. Synthetic data can be used to supplement, augment and
in some cases replace real data when training machine learning models.
Additionally, it enables the testing of machine learning or other data
dependent software systems without the risk of exposure that comes with
data disclosure. Underneath the hood it uses a unique hierarchical
generative modeling and recursive sampling techniques.

Features:
~~~~~~~~~

-  Modeling of single tables using Copulas and Deep Learning based
   models.
-  Modeling of complex multi-table relational datasets using Copulas and
   unique recursive modeling techniques.
-  Handling of multiple data types and missing data with minimum user
   input.
-  Support for pre-defined and custom constraints and data validation.
-  Definition of entire datasets with a custom and flexible Metadata
   JSON schema.

Coming soon:
~~~~~~~~~~~~

-  Time Series modeling with Autoregressive and Deep Learning models.

Try it out now!
---------------

If you want to quickly discover **SDV**, simply click the button below
and follow the tutorials!

|Binder|

Join our Slack Workspace
------------------------

If you want to be part of the SDV community to receive announcements of the latest releases,
ask questions, suggest new features or participate in the development meetings, please join
our Slack Workspace!

|Slack|

Explore SDV
-----------

* `Getting Started <getting_started/index.html>`_
* `User Guides <user_guides/index.html>`_
* `API Reference <api_reference/index.html>`_
* `Developer Guides <developer_guides/index.html>`_
* `Release Notes <history.html>`_

--------------

.. |Development Status| image:: https://img.shields.io/badge/Development%20Status-2%20--%20Pre--Alpha-yellow
   :target: https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha
.. |PyPi Shield| image:: https://img.shields.io/pypi/v/SDV.svg
   :target: https://pypi.python.org/pypi/SDV
.. |Travis CI Shield| image:: https://travis-ci.org/sdv-dev/SDV.svg?branch=master
   :target: https://travis-ci.org/sdv-dev/SDV
.. |Coverage Status| image:: https://codecov.io/gh/sdv-dev/SDV/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/sdv-dev/SDV
.. |Downloads| image:: https://pepy.tech/badge/sdv
   :target: https://pepy.tech/project/sdv
.. |Binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/sdv-dev/SDV/master?filepath=tutorials
.. |Slack| image:: https://img.shields.io/badge/Slack%20Workspace-Join%20now!-36C5F0?logo=slack
   :target: https://join.slack.com/t/sdv-space/shared_invite/zt-gdsfcb5w-0QQpFMVoyB2Yd6SRiMplcw


.. toctree::
    :maxdepth: 3
    :hidden:
    :titlesonly:

    getting_started/index
    user_guides/index
    api_reference/index
    developer_guides/index
    Release Notes <history>
