.. _evaluation:

Evaluating Synthetic Data
=========================

A very common question when someone starts using **SDV** to generate
synthetic data is: *"How good is the data that I just generated?"*

In order to answer this question, **SDV** has a collection of metrics
and tools that allow you to compare the *real* that you provided and the
*synthetic* data that you generated using **SDV** or any other tool.

In this guide we will show you how to perform this evaluation and how to
explore the different metrics that exist.

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
will indicate us how similar the two tables are, being 0 the worst and 1
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

For example, if you were interested on obtaining only the ``cstest`` and
``kstest`` metrics you can call the ``evaluate`` function as follows:

.. ipython:: python
    :okwarning:

    evaluate(synthetic_data, real_data, metrics=['cstest', 'kstest'])






Or, if we want to see the scores separately:

.. ipython:: python
    :okwarning:

    evaluate(synthetic_data, real_data, metrics=['cstest', 'kstest'], aggregate=False)






The complete list of possible metrics is:

-  ``cstest``: This metric compares the distributions of all the
   categorical columns of the table by using a Chi-squared test and
   returns the average of the ``p-values`` obtained across all the
   columns. If the tables that you are evaluating do not contain any
   categorical columns the result will be ``nan``.
-  ``kstest``: This metric compares the distributions of all the
   numerical columns of the table with a two-sample Kolmogorov–Smirnov
   test using the empirical CDF and returns the average of the
   ``p-values`` obtained across all the columns. If the tables that you
   are evaluating do not contain any numerical columns the result will
   be ``nan``.
-  ``logistic_detection``: This metric tries to use a Logistic
   Regression classifier to detect whether each row is real or synthetic
   and then evaluates its performance using an Area under the ROC curve
   metric. The returned score is 1 minus the ROC AUC score obtained
   by the classifier.
-  ``svc_detection``: This metric tries to use an Support Vector
   Classifier to detect whether each row is real or synthetic and then
   evaluates its performance using an Area under the ROC curve metric.
   The returned score is 1 minus the ROC AUC score obtained by the classifier.
