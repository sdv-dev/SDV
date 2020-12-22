.. _sdv.metrics:

sdv.metrics
===========

sdv.metrics.demos
-----------------

.. currentmodule:: sdv.metrics.demos

.. autosummary::
   :toctree: api/

    load_single_table_demo
    load_multi_table_demo
    load_timeseries_demo

sdv.metrics.tabular
-------------------

.. currentmodule:: sdv.metrics.tabular

.. autosummary::
   :toctree: api/

    SingleTableMetric
    SingleTableMetric.get_subclasses
    MultiColumnPairsMetric
    MultiColumnPairsMetric.get_subclasses
    MultiSingleColumnMetric
    MultiSingleColumnMetric.get_subclasses

Single Table BayesianNetwork Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

    BNLikelihood
    BNLikelihood.get_subclasses
    BNLikelihood.compute
    BNLogLikelihood
    BNLogLikelihood.get_subclasses
    BNLogLikelihood.compute

Single Table Statistical Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

    CSTest
    CSTest.get_subclasses
    CSTest.compute
    KSTest
    KSTest.get_subclasses
    KSTest.compute
    KSTestExtended
    KSTestExtended.get_subclasses
    KSTestExtended.compute
    ContinuousKLDivergence
    ContinuousKLDivergence.get_subclasses
    ContinuousKLDivergence.compute
    DiscreteKLDivergence
    DiscreteKLDivergence.get_subclasses
    DiscreteKLDivergence.compute

Single Table GaussianMixture Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

    GMLogLikelihood
    GMLogLikelihood.get_subclasses
    GMLogLikelihood.compute

Single Table Detection Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

    DetectionMetric
    DetectionMetric.get_subclasses
    ScikitLearnClassifierDetectionMetric
    ScikitLearnClassifierDetectionMetric.get_subclasses
    LogisticDetection
    LogisticDetection.get_subclasses
    LogisticDetection.compute
    SVCDetection
    SVCDetection.get_subclasses
    SVCDetection.compute

Single Table Efficacy Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

    MLEfficacyMetric
    MLEfficacyMetric.get_subclasses
    BinaryEfficacyMetric
    BinaryEfficacyMetric.get_subclasses
    BinaryDecisionTreeClassifier
    BinaryDecisionTreeClassifier.get_subclasses
    BinaryDecisionTreeClassifier.compute
    BinaryAdaBoostClassifier
    BinaryAdaBoostClassifier.get_subclasses
    BinaryAdaBoostClassifier.compute
    BinaryLogisticRegression
    BinaryLogisticRegression.get_subclasses
    BinaryLogisticRegression.compute
    BinaryMLPClassifier
    BinaryMLPClassifier.get_subclasses
    BinaryMLPClassifier.compute
    MulticlassEfficacyMetric
    MulticlassEfficacyMetric
    MulticlassDecisionTreeClassifier
    MulticlassDecisionTreeClassifier.get_subclasses
    MulticlassDecisionTreeClassifier.compute
    MulticlassMLPClassifier
    MulticlassMLPClassifier.get_subclasses
    MulticlassMLPClassifier.compute
    RegressionEfficacyMetric
    RegressionEfficacyMetric
    LinearRegression
    LinearRegression.get_subclasses
    LinearRegression.compute
    MLPRegressor
    MLPRegressor.get_subclasses
    MLPRegressor.compute

sdv.metrics.relational
----------------------

.. currentmodule:: sdv.metrics.relational

.. autosummary::
   :toctree: api/

    MultiTableMetric
    MultiTableMetric.get_subclasses
    MultiSingleTableMetric
    MultiSingleTableMetric.get_subclasses

Multi Table BayesianNetwork Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

    BNLikelihood
    BNLikelihood.get_subclasses
    BNLikelihood.compute
    BNLogLikelihood
    BNLogLikelihood.get_subclasses
    BNLogLikelihood.compute

Multi Table Statistical Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

    CSTest
    CSTest.get_subclasses
    CSTest.compute
    KSTest
    KSTest.get_subclasses
    KSTest.compute
    KSTestExtended
    KSTestExtended.get_subclasses
    KSTestExtended.compute

Multi Table Detection Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

    DetectionMetric
    DetectionMetric.get_subclasses
    LogisticDetection
    LogisticDetection.get_subclasses
    LogisticDetection.compute
    SVCDetection
    SVCDetection.get_subclasses
    SVCDetection.compute
    ParentChildDetectionMetric
    ParentChildDetectionMetric.get_subclasses
    LogisticParentChildDetection
    LogisticParentChildDetection.get_subclasses
    LogisticParentChildDetection.compute
    SVCParentChildDetection
    SVCParentChildDetection.get_subclasses
    SVCParentChildDetection.compute
