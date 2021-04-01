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
    synthetic_data = model.sample()

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

For example, if you were interested on obtaining only the ``CSTest`` and
``KSTest`` metrics you can call the ``evaluate`` function as follows:

.. ipython:: python
    :okwarning:

    evaluate(synthetic_data, real_data, metrics=['CSTest', 'KSTest'])


Or, if we want to see the scores separately:

.. ipython:: python
    :okwarning:

    evaluate(synthetic_data, real_data, metrics=['CSTest', 'KSTest'], aggregate=False)


For more details about all the metrics that exist for the different data modalities
please check the corresponding guides.
