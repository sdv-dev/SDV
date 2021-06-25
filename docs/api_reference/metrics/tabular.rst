.. _sdv.metrics.tabular:

sdv.metrics.tabular
===================

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

Single Table Privacy Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

    CategoricalPrivacyMetric
    CategoricalPrivacyMetric.get_subclasses
    NumericalPrivacyMetric
    NumericalPrivacyMetric.get_subclasses
    CategoricalCAP
    CategoricalCAP.get_subclasses
    CategoricalCAP.compute
    CategoricalZeroCAP
    CategoricalZeroCAP.get_subclasses
    CategoricalZeroCAP.compute
    CategoricalGeneralizedCAP
    CategoricalGeneralizedCAP.get_subclasses
    CategoricalGeneralizedCAP.compute
    CategoricalKNN
    CategoricalKNN.get_subclasses
    CategoricalKNN.compute
    CategoricalNB
    CategoricalNB.get_subclasses
    CategoricalNB.compute
    CategoricalRF
    CategoricalRF.get_subclasses
    CategoricalRF.compute
    CategoricalSVM
    CategoricalSVM.get_subclasses
    CategoricalSVM.compute
    NumericalMLP
    NumericalMLP.get_subclasses
    NumericalMLP.compute
    NumericalLR
    NumericalLR.get_subclasses
    NumericalLR.compute
    NumericalSVR
    NumericalSVR.get_subclasses
    NumericalSVR.compute
    CategoricalEnsemble
    CategoricalEnsemble.get_subclasses
    CategoricalEnsemble.compute
    NumericalRadiusNearestNeighbor
    NumericalRadiusNearestNeighbor.get_subclasses
    NumericalRadiusNearestNeighbor.compute