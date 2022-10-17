.. _evaluation_framework:

Evaluation Framework
====================

The `SDMetrics library <https://docs.sdv.dev/sdmetrics/>`__ includes reports, metrics and
visualizations that you can use to evaluate your synthetic data.

Required Information
--------------------

To use the SDMetrics library, you'll need:

1. Real data, loaded as a pandas DataFrame
2. Synthetic data, loaded as a pandas DataFrame
3. Metadata, represented as a dictionary format

We can get started using the demo data

.. ipython:: python
    :okwarning:

    from sdv.demo import load_tabular_demo
    from sdv.lite import TabularPreset

    metadata_obj, real_data = load_tabular_demo('student_placements', metadata=True)

    model = TabularPreset(metadata=metadata_obj, name='FAST_ML')
    model.fit(real_data)

    synthetic_data = model.sample(num_rows=real_data.shape[0])

After the previous steps, we will have two tables

- ``real_data``, containing data about student placements

.. ipython:: python
    :okwarning:

    real_data.head()

- ``synthetic_data``, containing synthesized students with the same format and mathematical
  properties as the original

.. ipython:: python
    :okwarning:

    synthetic_data.head()

We can also convert metadata to a Python dictionary by calling the ``to_dict`` method

.. ipython:: python
    :okwarning:

    metadata_dict = metadata_obj.to_dict()

Computing an overall score
--------------------------

Use the ``sdmetrics`` library to generate a Quality Report. This report evaluates the shapes
of the columns (marginal distributions) and the pairwise trends between the columns (correlations).

.. ipython:: python
    :okwarning:

    from sdmetrics.reports.single_table import QualityReport

    report = QualityReport()
    report.generate(real_data, synthetic_data, metadata_dict)

The report uses information in the metadata to select which metrics to apply to your data. The
final score is a number between 0 and 1, where 0 indicates the lowest quality and 1 indicates
the highest.

How was the obtained score computed?
------------------------------------

The report includes a breakdown for every property that it computed.

.. ipython:: python
    :okwarning:

    report.get_details(property_name='Column Shapes')

In the detailed view, you can see the quality score for each column of the table. Based on the data
type, different metrics may be used for the computation.

For more information about the Quality Report, see the `SDMetrics Docs 
<https://docs.sdv.dev/sdmetrics/reports/quality-report>`__.

Can I apply different metrics?
------------------------------

Outside of reports, the SDMetrics library contains a variety of metrics that you can apply
manually. For example the `NewRowSynthesis metric <https://docs.sdv.dev/sdmetrics/metrics/metrics-glossary/newrowsynthesis>`__
measures whether each row in the synthetic data is novel or whether it exactly matches a row in
the real data.

.. ipython:: python
    :okwarning

    from sdmetrics.single_table import NewRowSynthesis

    NewRowSynthesis.compute(real_data, synthetic_data, metadata_dict)

See the `SDMetrics Glossary <https://docs.sdv.dev/sdmetrics/metrics/metrics-glossary>`__ for a full
list of metrics that you can apply.
