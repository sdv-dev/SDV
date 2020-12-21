.. _single_table_metrics:

Single Table Metrics
====================

In this section we will show you which metrics exist for single table datasets and
how to use them.

Let us start by loading some demo data that we will use to explore the different
metrics that exist.

.. ipython::
    :verbatim:

    In [1]: from sdv.metrics.demos import load_single_table_demo

    In [2]: real_data, synthetic_data, metadata = load_single_table_demo()


This will return us three objects:

The ``real_data``, which is the single table ``student_placements`` demo dataset:


.. ipython::
    :verbatim:

    In [3]: real_data
    Out[3]:
         student_id gender  second_perc  high_perc high_spec  degree_perc  ... mba_perc   salary  placed  start_date   end_date  duration
    0         17264      M        67.00      91.00  Commerce        58.00  ...    58.80  27000.0    True  2020-07-23 2020-10-12       3.0
    1         17265      M        79.33      78.33   Science        77.48  ...    66.28  20000.0    True  2020-01-11 2020-04-09       3.0
    2         17266      M        65.00      68.00      Arts        64.00  ...    57.80  25000.0    True  2020-01-26 2020-07-13       6.0
    3         17267      M        56.00      52.00   Science        52.00  ...    59.43      NaN   False         NaT        NaT       NaN
    4         17268      M        85.80      73.60  Commerce        73.30  ...    55.50  42500.0    True  2020-07-04 2020-09-27       3.0
    ..          ...    ...          ...        ...       ...          ...  ...      ...      ...     ...         ...        ...       ...
    210       17474      M        80.60      82.00  Commerce        77.60  ...    74.49  40000.0    True  2020-07-27 2020-10-20       3.0
    211       17475      M        58.00      60.00   Science        72.00  ...    53.62  27500.0    True  2020-01-23 2020-08-04       6.0
    212       17476      M        67.00      67.00  Commerce        73.00  ...    69.72  29500.0    True  2020-01-25 2020-08-05       6.0
    213       17477      F        74.00      66.00  Commerce        58.00  ...    60.23  20400.0    True  2020-01-19 2020-04-20       3.0
    214       17478      M        62.00      58.00   Science        53.00  ...    60.22      NaN   False         NaT        NaT       NaN
    .
    [215 rows x 17 columns]

The ``synthetic_data``, which is a clone of the ``real_data`` which has been generated
by the ``CTGAN`` tabular model.


.. ipython::
    :verbatim:

    In [4]: synthetic_data
    Out[4]:
         student_id gender  second_perc   high_perc high_spec  degree_perc  ...   mba_perc   salary  placed  start_date   end_date  duration
    0             0      F    41.361060   85.425072  Commerce    74.972674  ...  57.291083      NaN    True  2020-02-11 2020-08-02       3.0
    1             1      M    63.720169   99.059033  Commerce    62.769650  ...  79.068319      NaN   False         NaT        NaT       NaN
    2             2      M    58.473884   89.241528   Science    83.066328  ...  77.042950  26727.0    True  2020-02-13 2020-05-27       3.0
    3             3      F    77.232204  100.523788  Commerce    61.010445  ...  68.132991  22058.0    True  2020-09-24 2020-11-07       3.0
    4             4      F    54.067830  109.611537  Commerce    72.846753  ...  66.363138      NaN   False         NaT        NaT       NaN
    ..          ...    ...          ...         ...       ...          ...  ...        ...      ...     ...         ...        ...       ...
    210         210      M    58.981597   97.809826  Commerce    73.548889  ...  61.981631      NaN   False         NaT        NaT       NaN
    211         211      M    42.643139   75.259843  Commerce    72.478613  ...  55.746391      NaN   False         NaT        NaT       NaN
    212         212      M    58.202031  103.876132  Commerce    81.088376  ...  58.117902  28772.0    True  2020-01-23 2021-02-26       6.0
    213         213      M    53.939037   70.498207  Commerce    65.284175  ...  53.206451  25441.0    True  2020-06-13 2020-06-14       6.0
    214         214      M    35.696869  100.655357  Commerce    58.946189  ...  48.470545      NaN   False         NaT        NaT       NaN
    .
    [215 rows x 17 columns]


And a ``metadata``, which is the ``dict`` representation of the ``student_placements`` metadata.


.. ipython::
    :verbatim:

    In [5]: metadata
    Out[5]:
    {'fields': {'start_date': {'type': 'datetime', 'format': '%Y-%m-%d'},
      'end_date': {'type': 'datetime', 'format': '%Y-%m-%d'},
      'salary': {'type': 'numerical', 'subtype': 'integer'},
      'duration': {'type': 'categorical'},
      'student_id': {'type': 'id', 'subtype': 'integer'},
      'high_perc': {'type': 'numerical', 'subtype': 'float'},
      'high_spec': {'type': 'categorical'},
      'mba_spec': {'type': 'categorical'},
      'second_perc': {'type': 'numerical', 'subtype': 'float'},
      'gender': {'type': 'categorical'},
      'degree_perc': {'type': 'numerical', 'subtype': 'float'},
      'placed': {'type': 'boolean'},
      'experience_years': {'type': 'numerical', 'subtype': 'float'},
      'employability_perc': {'type': 'numerical', 'subtype': 'float'},
      'mba_perc': {'type': 'numerical', 'subtype': 'float'},
      'work_experience': {'type': 'boolean'},
      'degree_type': {'type': 'categorical'}},
     'constraints': [],
     'model_kwargs': {},
     'name': None,
     'primary_key': 'student_id',
     'sequence_index': None,
     'entity_columns': [],
     'context_columns': []}

These three elements, or their corresponding equivalents, are all you will need to
run most of the *Single Table Metrics* on your own *Synthetic Dataset*.

Single Table Metric Families
----------------------------

The *Single Table Metrics* are grouped in multiple families:

* **Statistical Metrics**: These are metrics that compare the tables by running different
  statistical tests on them. Some of them work by comparing multiple columns at once, while
  other compare the different individual columns separately and later on return an aggregated
  result.
* **Likelihood Metrics**: These metrics attempt to fit a probabilistic model to the
  real data and later on evaluate the likelihood of the synthetic data on it.
* **Detection Metrics**: These metrics try to train a Machine Learning Classifier that learns
  to distinguish the real data from the synthetic data, and report a score of how successful
  this classifier is.
* **Machine Learning Efficacy Metrics**: These metrics train a Machine Learning model on your
  synthetic data and later on evaluate the model performance on the real data. Since these
  metrics need to evaluate the performance of a Machine Learning model on the dataset, they
  work only on datasets that represent a Machine Learning problem.

Statistical Metrics
~~~~~~~~~~~~~~~~~~~

The metrics of this family compare the tables by running different types of statistical tests
on them.

In the most simple scenario, these metrics compare individual columns from the real table
with the corresponding column from the synthetic table, and at the end report the average
outcome from the test.

Such metrics are:

* ``sdv.metrics.tabular.KSTest``: This metric uses the two-sample Kolmogorov–Smirnov test
  to compare the distributions of continuous columns using the empirical CDF.
  The output for each column is 1 minus the KS Test D statistic, which indicates the maximum
  distance between the expected CDF and the observed CDF values.
* ``sdv.metrics.tabular.CSTest``: This metric uses the Chi-Squared test to compare the
  distributions of two discrete columns. The output for each column is the CSTest p-value,
  which indicates the probability of the two columns having been sampled from the same
  distribution.

Let us execute these two metrics on the loaded data:

.. ipython::
    :verbatim:

    In [6]: from sdv.metrics.tabular import CSTest, KSTest

    In [7]: CSTest.compute(real_data, synthetic_data)
    Out[7]: 0.8078084931103922

    In [8]: KSTest.compute(real_data, synthetic_data)
    Out[8]: 0.6372093023255814

In each case, the statistical test will be executed on all the compatible column (so, categorical
or boolean columns for ``CSTest`` and numerical columns for ``KSTest``), and report the average
score obtained.

.. note:: If your table does not contain any column of the compatible type, the output of
   either metric will be ``nan``.

We can also compute the metrics by calling the ``sdv.evaluate`` function passing either the
metric classes or their names:

.. ipython::
    :verbatim:

    In [9]: from sdv.evaluation import evaluate

    In [10]: evaluate(synthetic_data, real_data, metrics=['CSTest', 'KSTest'], aggregate=False)
    Out[10]:
       metric                                     name     score  min_value  max_value      goal
    0  CSTest                              Chi-Squared  0.807808        0.0        1.0  MAXIMIZE
    1  KSTest  Inverted Kolmogorov-Smirnov D statistic  0.637209        0.0        1.0  MAXIMIZE


Likelihood Metrics
~~~~~~~~~~~~~~~~~~

The metrics of this family compare the tables by fitting the real data to a probabilistic
model and afterwards compute the likelihood of the synthetic data belonging to the learned
distribution.

Such metrics are:

* ``sdv.metrics.tabular.BNLikelihood``: This metric fits a BayesianNetwork to the real
  data and then evaluates the average likelihood of the rows from the synthetic data on it.
* ``sdv.metrics.tabular.BNLogLikelihood``: This metric fits a BayesianNetwork to the real
  data and then evaluates the average log likelihood of the rows from the synthetic data on it.
* ``sdv.metrics.tabular.GMLogLikelihood``: This metric fits multiple GaussianMixture models to
  the real data and then evaluates the average log likelihood of the synthetic data on them.

.. note:: These metrics do not accept missing data, so we will replace all the missing
   values with a 0 before executing them.

Let us execute these metrics on the loaded data:

.. ipython::
    :verbatim:

    In [11]: from sdv.metrics.tabular import BNLikelihood, BNLogLikelihood, GMLogLikelihood

    In [12]: BNLikelihood.compute(real_data.fillna(0), synthetic_data.fillna(0))
    Out[12]: 0.004311090583670755

    In [13]: BNLogLikelihood.compute(real_data.fillna(0), synthetic_data.fillna(0))
    Out[13]: -14.62132601319649

    In [14]: GMLogLikelihood.compute(real_data.fillna(0), synthetic_data.fillna(0))
    Out[14]: -35024.711762921426

Detection Metrics
~~~~~~~~~~~~~~~~~

The metrics of this family evaluate how hard it is to distinguish the synthetic data from the
real data by using a Machine Learning model. To do this, the metrics will shuffle the real
data and synthetic data together with flags indicating whether the data is real or synthetic,
and then cross validate a Machine Learning model that tries to predict this flag.
The output of the metrics will be the 1 minus the average ROC AUC score across all the cross
validation splits.

Such metrics are:

* ``sdv.metrics.tabular.LogisticDetection``: Detection metric based on a ``LogisticRegression``
  classifier from ``scikit-learn``.
* ``sdv.metrics.tabular.SVCDetection``: Detection metric based on a ``SVC`` classifier from
  ``scikit-learn``.

Let us execute these metrics on the loaded data:

.. ipython::
    :verbatim:

    In [15]: from sdv.metrics.tabular import LogisticDetection, SVCDetection

    In [16]: LogisticDetection.compute(real_data, synthetic_data)
    Out[16]: 0.0

    In [17]: SVCDetection.compute(real_data, synthetic_data)
    Out[17]: 0.0009056395989102128

Machine Learning Efficacy Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This family of metrics will evaluate whether it is possible to replace the real data with
synthetic data in order to solve a Machine Learning Problem by learning a Machine Learning
model on the synthetic data and then evaluating the score which it obtains when evaluated
on the real data.

.. note:: Since this metrics will be evaluated by trying to solve a Machine Learning problem,
  they can only be used on datasets that contain a target column that needs or can be predicted
  using the rest of the data, and the scores obtained by the metrics will be inversely
  proportional to how hard that Machine Problem is.

The metrics on this family are organized by Machine Learning problem type and model.

* Binary Classification Metrics:

  * ``BinaryDecisionTreeClassifier``
  * ``BinaryAdaBoostClassifier``
  * ``BinaryLogisticRegression``
  * ``BinaryMLPClassifier``

* Multiclass Classification Metrics:

  * ``MulticlassDecisionTreeClassifier``
  * ``MulticlassMLPClassifier``

* Regression Metrics:

  * ``LinearRegressionClassifier``
  * ``MLPRegressor``

In order to run these metrics we will need to select a column from our dataset which we will
use as the target for the prediction problem. For example, in the demo dataset there are multiple
columns that can be used as possible targets for a Machine Learning problem:

* ``work_experience`` and ``placed`` can be used for binary classification problems.
* ``high_spec``, ``degree_type``, ``mba_spec`` and ``duration`` can be used for multiclass
  classification problems.
* ``second_perc``, ``high_perc``, ``degree_perc``, ``experience_years``, ``employability_perc``,
  ``mba_perc`` and ``salary`` can be used for regression problems.

Let's select the ``mba_spect`` column as the target for our problem and let the Machine Learning
Efficacy Metric attempt to predict it using the rest of the columns.

.. ipython::
    :verbatim:

    In [18]: from sdv.metrics.tabular import MulticlassDecisionTreeClassifier

    In [19]: MulticlassDecisionTreeClassifier.compute(real_data, synthetic_data, target='mba_spec')
    Out[19]: 0.5581012959477294

Notice that the value returned by the metric does not only depend on how good our synthetic data
is, but also on how hard the Machine Learning problem that we are trying to solve is. For reference,
we may want to compare this result with the one obtained when trying to make the prediction
using real data as input. For this, we will need to split the data into train and test partitions
and call the metric replacing the real data and synthetic data with the test and training data
respectively.

.. ipython::
    :verbatim:

    In [20]: train = real_data.sample(int(len(real_data) * 0.75))

    In [21]: test = real_data[~real_data.index.isin(train.index)]

    In [22]: MulticlassDecisionTreeClassifier.compute(test, train, target='mba_spec')
    Out[22]: 0.5703908682116914

.. note:: Apart from passing the ``target`` variable as an argument, we can also store its
   value inside the ``metadata`` dict and pass it to the metric:

   .. ipython::
       :verbatim:

       In [23]: metadata['target'] = 'mba_spec'

       In [24]: MulticlassDecisionTreeClassifier.compute(real_data, synthetic_data, metadata)
       Out[24]: 0.5767075571709829
