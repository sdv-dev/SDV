.. _multi_table_metrics:

Multi Table Metrics
===================

In this section we will show you which metrics exist for multi table datasets and
how to use them.

Let us start by loading some demo data that we will use to explore the different
metrics that exist.

.. ipython::
    :verbatim:

    In [1]: from sdv.metrics.demos import load_multi_table_demo

    In [2]: real_data, synthetic_data, metadata = load_multi_table_demo()

This will return us three objects:

The ``real_data``, which is a dict containing the ``SDV`` demo dataset, with three tables:

.. ipython::
    :verbatim:

    In [3]: real_data
    Out[3]:
    {'users':    user_id country gender  age
     0        0      US      M   34
     1        1      UK      F   23
     2        2      ES   None   44
     3        3      UK      M   22
     4        4      US      F   54
     5        5      DE      M   57
     6        6      BG      F   45
     7        7      ES   None   41
     8        8      FR      F   23
     9        9      UK   None   30,
     'sessions':    session_id  user_id  device       os
     0           0        0  mobile  android
     1           1        1  tablet      ios
     2           2        1  tablet  android
     3           3        2  mobile  android
     4           4        4  mobile      ios
     5           5        5  mobile  android
     6           6        6  mobile      ios
     7           7        6  tablet      ios
     8           8        6  mobile      ios
     9           9        8  tablet      ios,
     'transactions':    transaction_id  session_id           timestamp  amount  approved
     0               0           0 2019-01-01 12:34:32   100.0      True
     1               1           0 2019-01-01 12:42:21    55.3      True
     2               2           1 2019-01-07 17:23:11    79.5      True
     3               3           3 2019-01-10 11:08:57   112.1     False
     4               4           5 2019-01-10 21:54:08   110.0     False
     5               5           5 2019-01-11 11:21:20    76.3      True
     6               6           7 2019-01-22 14:44:10    89.5      True
     7               7           8 2019-01-23 10:14:09   132.1     False
     8               8           9 2019-01-27 16:09:17    68.0      True
     9               9           9 2019-01-29 12:10:48    99.9      True}

The ``synthetic_data``, which is a clone of the ``real_data`` which has been generated
by the ``HMA1`` relational model.

.. ipython::
    :verbatim:

    In [4]: synthetic_data
    Out[4]:
    {'users':    user_id country gender  age
     0        0      US      M   37
     1        1      US      M   57
     2        2      DE      F   56
     3        3      DE      F   43
     4        4      ES      M   30
     5        5      ES      F   38
     6        6      UK      F   30
     7        7      BG      F   30
     8        8      US      M   46
     9        9      US      F   19,
     'sessions':    session_id  user_id  device       os
     0           0        0  mobile      ios
     1           1        1  mobile  android
     2           2        2  mobile      ios
     3           3        3  tablet      ios
     4           4        5  mobile  android
     5           5        5  mobile  android
     6           6        8  mobile  android
     7           7        9  tablet  android
     8           8        9  tablet  android,
     'transactions':    transaction_id  session_id           timestamp      amount  approved
     0               0           0 2019-01-17 20:59:07   95.998821      True
     1               1           1 2019-01-04 12:25:37   92.812296      True
     2               2           2 2019-01-13 09:24:52   68.369142      True
     3               3           3 2019-03-01 15:57:44  561.468787      True
     4               4           3 2019-01-27 21:36:42  317.456320      True
     5               5           4 2019-01-02 18:10:06   84.950110      True
     6               6           5 2019-01-02 18:09:56   84.947125      True
     7               7           6 2019-01-01 15:32:01  101.137274     False
     8               8           7 2019-01-09 05:00:02   84.190594      True
     9               9           8 2019-01-09 04:20:06   83.926505      True}

And a ``metadata``, which is the ``dict`` representation of the dataset:

.. ipython::
    :verbatim:

    In [5]: metadata
    Out[5]:
    {'tables': {'users': {'primary_key': 'user_id',
       'fields': {'user_id': {'type': 'id', 'subtype': 'integer'},
        'country': {'type': 'categorical'},
        'gender': {'type': 'categorical'},
        'age': {'type': 'numerical', 'subtype': 'integer'}}},
      'sessions': {'primary_key': 'session_id',
       'fields': {'session_id': {'type': 'id', 'subtype': 'integer'},
        'user_id': {'ref': {'field': 'user_id', 'table': 'users'},
         'type': 'id',
         'subtype': 'integer'},
        'device': {'type': 'categorical'},
        'os': {'type': 'categorical'}}},
      'transactions': {'primary_key': 'transaction_id',
       'fields': {'transaction_id': {'type': 'id', 'subtype': 'integer'},
        'session_id': {'ref': {'field': 'session_id', 'table': 'sessions'},
         'type': 'id',
         'subtype': 'integer'},
        'timestamp': {'type': 'datetime', 'format': '%Y-%m-%d'},
        'amount': {'type': 'numerical', 'subtype': 'float'},
        'approved': {'type': 'boolean'}}}}}

These three elements, or their corresponding equivalents, are all you will need to
run most of the *Multi Table Metrics* on your own *Synthetic Dataset*.

Multi Table Metric Families
---------------------------

The *Multi Table Metrics* are grouped in multiple families:

* **Multi Single Table Metrics**: These metrics simply a *Single Table Metric* on each table
  in the dataset and report the average score obtained.
* **Parent-Child Detection Metrics**: These metrics de-normalize the child tables on each
  parent-child relationship found in the dataset and then apply a *Single Table Detection Metric*
  on the resulting tables. If there is more than one parent-child relationship in the dataset,
  the overall score is the average of the scores obtained by the *Single Table Detection Metric*
  on each one of them.

Multi Single Table Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~

These metrics simply apply a *Single Table Metric* on each table from the dataset and then
report back the average score obtained.

The list of such metrics is:

* ``CSTest``: Multi Single Table metric based on the Single Table CSTest metric.
* ``KSTest``: Multi Single Table metric based on the Single Table KSTest metric.
* ``KSTestExtended``: Multi Single Table metric based on the Single Table KSTestExtended metric.
* ``LogisticDetection``: Multi Single Table metric based on the Single Table LogisticDetection metric.
* ``SVCDetection``: Multi Single Table metric based on the Single Table SVCDetection metric.
* ``BNLikelihood``: Multi Single Table metric based on the Single Table BNLikelihood metric.
* ``BNLogLikelihood``: Multi Single Table metric based on the Single Table BNLogLikelihood metric.

Let's try to use the ``KSTestExtended`` metric:

.. ipython::
    :verbatim:

    In [6]: from sdv.metrics.relational import KSTestExtended

    In [7]: KSTestExtended.compute(real_data, synthetic_data)
    Out[7]: 0.8194444444444443

Parent Child Detection Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These metrics will de-normalize each parent-child relationship on the dataset and build
a table out of it. Afterwards, they will apply a *Single Table Detection Metric* on the
resulting tables and report back the average of the obtained scores.

Such metrics are:

* ``LogisticParentChildDetection``: Parent-child detection metric based on a ``LogisticDetection``.
* ``SVCParentChildDetection``: Parent-child detection metric based on a ``SVCDetection``.

Since these metrics need to explit the table relationships, you will need to pass the
``metadata`` dict alongside the ``real_data`` and ``synthetic_data``, so the metric
is able to extract the relationships from it.

Let's see an example using the ``LogisticParentChildDetection`` metric:

.. ipython::
    :verbatim:

    In [8]: from sdv.metrics.relational import LogisticParentChildDetection

    In [9]: LogisticParentChildDetection.compute(real_data, synthetic_data, metadata)
    Out[9]: 0.8472222222222222

If you want, instead of passing the ``metadata``, you can build a ``foreign_keys`` list
with a ``tuple`` for each relationship indicating:

* The parent table
* The parent field
* The child table
* The child field

.. ipython::
    :verbatim:

    In [10]: fks = [
        ...: ('users', 'user_id', 'sessions', 'user_id'),
        ...: ('sessions', 'session_id', 'transactions', 'session_id')
        ...: ]

    In [11]: LogisticParentChildDetection.compute(real_data, synthetic_data, foreign_keys=fks)
    Out[11]: 0.8333333333333333
