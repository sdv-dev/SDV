.. _timeseries_metrics:

Time Series Metrics
===================

In this section we will show you which metrics exist for time series datasets and
how to use them.

Let us start by loading some demo data that we will use to explore the different
metrics that exist.

.. ipython::
    :verbatim:

    In [1]: from sdv.metrics.demos import load_timeseries_demo

    In [2]: real_data, synthetic_data, metadata = load_timeseries_demo()


This will return us three objects:

The ``real_data``, which is the time series ``sunglasses`` demo dataset:


.. ipython::
    :verbatim:

    In [3]: real_data
    Out[3]:
              date  store_id      region  day_of_week  total_sales  nb_customers
    0   2020-06-01     68608    New York            0       736.19            43
    1   2020-06-02     68608    New York            1       777.31            45
    2   2020-06-03     68608    New York            2       921.22            54
    3   2020-06-04     68608    New York            3      1085.69            63
    4   2020-06-05     68608    New York            4      1476.30            86
    ..         ...       ...         ...          ...          ...           ...
    695 2020-06-03     75193  California            2      3100.43           182
    696 2020-06-04     75193  California            3      3244.34           190
    697 2020-06-05     75193  California            4      3593.84           211
    698 2020-06-06     75193  California            5      4518.98           265
    699 2020-06-07     75193  California            6      3429.37           201
    .
    [700 rows x 6 columns]

The ``synthetic_data``, which is a clone of the ``real_data`` which has been generated
by the ``PAR`` timeseries model.


.. ipython::
    :verbatim:

    In [4]: synthetic_data
    Out[4]:
              date  store_id      region  day_of_week  total_sales  nb_customers
    0   2020-06-01         0    New York            0   762.349155            43
    1   2020-06-02         0    New York            1   809.673681            42
    2   2020-06-03         0    New York            2   950.915155            60
    3   2020-06-04         0    New York            3  1137.295165            62
    4   2020-06-05         0    New York            4  1517.681393            94
    ..         ...       ...         ...          ...          ...           ...
    695 2020-06-03        99  California            1  3042.781939           174
    696 2020-06-04        99  California            2  3183.143605           185
    697 2020-06-05        99  California            4  3541.943027           180
    698 2020-06-06        99  California            5  2978.115245           169
    699 2020-06-07        99  California            6  5427.324077           141
    .
    [700 rows x 6 columns]


And a ``metadata``, which is the ``dict`` representation of the ``student_placements`` metadata.


.. ipython::
    :verbatim:

    In [5]: metadata
    Out[5]:
    {'fields': {'region': {'type': 'categorical'},
      'store_id': {'type': 'numerical', 'subtype': 'integer'},
      'nb_customers': {'type': 'numerical', 'subtype': 'integer'},
      'total_sales': {'type': 'numerical', 'subtype': 'float'},
      'date': {'type': 'datetime'},
      'day_of_week': {'type': 'numerical', 'subtype': 'integer'}},
     'entity_columns': ['store_id'],
     'sequence_index': 'date',
     'context_columns': ['region']}

These three elements, or their corresponding equivalents, are all you will need to
run most of the *Time Series Metrics* on your own *Synthetic Dataset*.

Time Series Metric Families
---------------------------

The *Time Series Metrics* are grouped in multiple families:

* **Detection Metrics**: These metrics try to train a Machine Learning Classifier that learns
  to distinguish the real data from the synthetic data, and report a score of how successful
  this classifier is.
* **Machine Learning Efficacy Metrics**: These metrics train a Machine Learning model on your
  synthetic data and later on evaluate the model performance on the real data. Since these
  metrics need to evaluate the performance of a Machine Learning model on the dataset, they
  work only on datasets that represent a Machine Learning problem.

Detection Metrics
~~~~~~~~~~~~~~~~~

The metrics of this family evaluate how hard it is to distinguish the synthetic data from the
real data by using a Machine Learning model. To do this, the metrics will shuffle the real
data and synthetic data together with flags indicating whether the data is real or synthetic,
and then cross validate a Machine Learning model that tries to predict this flag using a
rectified version of a ROC AUC score that returns values in the range [0, 1].
The output of the metrics will be the 1 minus the average of the score across all the cross
validation splits.

Such metrics are:

* ``sdv.metrics.timeseries.LSTMDetection``: Detection metric based on an ``LSTM`` classifier
  implemented using ``PyTorch``.
* ``sdv.metrics.timeseries.TSFCDetection``: Detection metric based on a
  ``TimeSeriesForestClassifier`` pipeline implemented using ``sktime``.

Let us execute these metrics on the loaded data:

.. ipython::
    :verbatim:

    In [15]: from sdv.metrics.timeseries import LSTMDetection, TSFCDetection

    In [7]: LSTMDetection.compute(real_data, synthetic_data, metadata)
    Out[7]: 0.5

    In [8]: TSFCDetection.compute(real_data, synthetic_data, metadata)
    Out[8]: 0.0

Machine Learning Efficacy Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This family of metrics will evaluate whether it is possible to replace the real data with
synthetic data in order to solve a Machine Learning Problem by learning a Machine Learning
model on both the synthetic data and real data and then comparing the score which they
obtain when evaluated on held out real data. The output is the score obtained by the model
fitted on synthetic data divided by the score obtained when fitted on real data.

.. note:: Since this metrics will be evaluated by trying to solve a Machine Learning problem,
  they can only be used on datasets that contain a target column that needs or can be predicted
  using the rest of the data, and the scores returned by the metrics will be useful only
  if the Machine Learning problem is relatively easy to solve. Otherwise, if the performance
  of the models when fitted on real data is too low, the output from these metrics may be
  meaningless.

The metrics on this family are organized by Machine Learning problem type and model.

* Classification Metrics:

    * ``TSFClassifierEfficacy``: Efficacy metric based on a ``TimeSeriesForestClassifier`` from
      ``sktime``.
    * ``LSTMClassifierEfficacy``: Efficacy metric based on an LSTM Classifier implemented using
      ``PyTorch``.

In order to run these metrics we will need to select a column from our dataset which we will
use as the target for the prediction problem. For example, in the demo dataset we can try to
predict the ``region`` of the store based on the sequence items.

Let's try that:

.. ipython::
    :verbatim:

    In [9]: from sdv.metrics.timeseries import TSFClassifierEfficacy

    In [10]: TSFClassifierEfficacy.compute(real_data, synthetic_data, metadata, target='region')
    Out[10]: 1


.. note:: Apart from passing the ``target`` variable as an argument, we can also store its
   value inside the ``metadata`` dict and pass it to the metric:

   .. ipython::
       :verbatim:

       In [11]: metadata['target'] = 'region'

       In [11]: TSFClassifierEfficacy.compute(real_data, synthetic_data, metadata, target='region')
       Out[11]: 1.0
