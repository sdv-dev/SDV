.. _evaluation_framework:

Evaluation Framework
====================

**SDV** contains a *Synthetic Data Evaluation Framework* that facilitates
the task of evaluating the quality of your *Synthetic Dataset* by
applying multiple *Synthetic Data Metrics* on it and reporting results
in a comprehensive way.

Using the SDV Evaluation Framework
----------------------------------

To evaluate the quality of synthetic data we essentially need two things:
*real* data and *synthetic* data that pretends to resemble it.

Let us start by loading a demo table and generate a synthetic replica of
it using the ``GaussianCopula`` model.

.. ipython:: python
    :okwarning:

    from sdv.demo import load_tabular_demo
    from sdv.tabular import GaussianCopula

    real_data = load_tabular_demo('student_placements')

    model = GaussianCopula()
    model.fit(real_data)
    synthetic_data = model.sample(len(real_data))

After the previous steps we will have two tables:

-  ``real_data``: A table containing data about student placements

.. ipython:: python
    :okwarning:

    real_data.head()


-  ``synthetic_data``: A synthetically generated table that contains
   data in the same format and with similar statistical properties as
   the ``real_data``.

.. ipython:: python
    :okwarning:

    synthetic_data.head()


.. note:: For more details about this process, please visit the :ref:`gaussian_copula` guide.

Computing an overall score
~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest way to see how similar the two tables are is to import the
``sdv.evaluation.evaluate`` function and run it passing both the
``synthetic_data`` and the ``real_data`` tables.

.. ipython:: python
    :okwarning:

    from sdv.evaluation import evaluate

    evaluate(synthetic_data, real_data)


The output of this function call will be a number between 0 and 1 that
will indicate how similar the two tables are, being 0 the worst and 1
the best possible score.

How was the obtained score computed?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``evaluate`` function applies a collection of pre-configured metric
functions and returns the average of the scores that the data obtained
on each one of them. In most scenarios this can be enough to get an idea
about the similarity of the two tables, but you might want to explore
the metrics in more detail.

In order to see the different metrics that were applied you can pass and
additional argument ``aggregate=False``, which will make the
``evaluate`` function return a dictionary with the scores that each one
of the metrics functions returned:

.. ipython:: python
    :okwarning:

    evaluate(synthetic_data, real_data, aggregate=False)


Can I control which metrics are applied?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, the ``evaluate`` function will apply all the metrics that
are included within the SDV Evaluation framework. However, the list of
metrics that are applied can be controlled by passing a list with the
names of the metrics that you want to apply.

For example, if you were interested on obtaining only the ``CSTest``
metric you can call the ``evaluate`` function as follows:

.. ipython:: python
    :okwarning:

    evaluate(synthetic_data, real_data, metrics=['CSTest'])


Or, if we want to see the scores separately:

.. ipython:: python
    :okwarning:

    evaluate(synthetic_data, real_data, metrics=['CSTest'], aggregate=False)


For more details about all the metrics that exist for the different data modalities
please check the corresponding guides.



The `SDMetrics library <https://docs.sdv.dev/sdmetrics/>` includes reports, metrics and
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
<https://docs.sdv.dev/sdmetrics/reports/quality-report>`.

Can I apply different metrics?
------------------------------

Outside of reports, the SDMetrics library contains a variety of metrics that you can apply
manually. For example the `NewRowSynthesis metric <https://docs.sdv.dev/sdmetrics/metrics/metrics-glossary/newrowsynthesis>`
measures whether each row in the synthetic data is novel or whether it exactly matches a row in
the real data.

.. ipython:: python
    :okwarning

    from sdmetrics.single_table import NewRowSynthesis

    NewRowSynthesis.compute(real_data, synthetic_data, metadata_dict)

See the `SDMetrics Glossary <https://docs.sdv.dev/sdmetrics/metrics/metrics-glossary>` for a full
list of metrics that you can apply.
