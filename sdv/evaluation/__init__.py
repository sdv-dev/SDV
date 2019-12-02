"""Tools for evaluate the synthesized data.

The main function in this module is `evaluate` whose usage has been shown in the README.

``evaluate`` works by using a series of descriptors on each column for both datasets, and
then applying metrics on the generated descriptors.

A descriptor is a function ``descriptor(column: pandas.Series) -> pandas.Series`` whose input and
return value are ``pandas.Series``.

A metric is a function ``metric(expected: numpy.ndarray, observed: numpy.ndarray) -> float`` that
takes to numpy arrays and returns a float.

So, if you want to use ``evaluate`` with your custom descriptors and/or metrics, you can
call it with:

.. code-block:: python

    def my_descriptor_function(column):
       # All necessary steps here
       return description

    def my_custom_metric(expected, observed):
       # All necessary steps here
       return metric

    my_descriptors = [
       my_descriptor_function
    ]

    my_metrics = [
       my_custom_metric
    ]

    evaluate(real, samples, descriptors=my_descriptors, metrics=my_metrics)

        my_custom_metric   6.040444e+32
        dtype: float64

"""

import pandas as pd

from sdv.evaluation.descriptors import DESCRIPTORS
from sdv.evaluation.metrics import DEFAULT_METRICS
from sdv.metadata import Metadata


def get_descriptor_values(real, synth, descriptor, table_name=None):
    """Compute the descriptor values for the given tables.

    Args:
        real (pandas.DataFrame):
            Real data.
        synth (pandas.DataFrame):
            Synthesized data.
        descriptor (callable or str):
            Callable that accepts columns and returns real-values.
        table_name (str):
            Table name to format the described output name. Defaults to ``None``.

    Return:
        pandas.DataFrame:
            It will contain the descriptor output for each column as columns.

    """
    real_values = list()
    synth_values = list()

    for column_name in real:
        try:
            described_real_column = pd.Series(descriptor(real[column_name]))
            described_synth_column = pd.Series(descriptor(synth[column_name]))

            if table_name:
                column_name = '{}_{}'.format(table_name, column_name)

            described_name = '{}_{}'.format(descriptor.__name__, column_name)
            if len(described_real_column) > 1:
                described_name = described_name + '_'
                described_real_column = described_real_column.add_prefix(described_name)
                described_synth_column = described_synth_column.add_prefix(described_name)
            else:
                described_real_column.index = [described_name]
                described_synth_column.index = [described_name]

            real_values.append(described_real_column.T)
            synth_values.append(described_synth_column.T)
        except TypeError:
            pass

    real_values = pd.concat(real_values, axis=0, sort=False)
    synth_values = pd.concat(synth_values, axis=0, sort=False)
    return pd.concat([real_values, synth_values], axis=1, sort=True, ignore_index=True).T


def get_descriptors_table(real, synth, metadata, table_name, descriptors=DESCRIPTORS):
    """Score the synthesized data using the given metrics and descriptors.

    Args:
        real (pandas.DataFrame):
            Table of real data.
        synth (pandas.DataFrame):
            Table of synthesized data.
        metadata (Metadata):
            Metadata object to get column names from a table without ids.
        descriptors (dict[str, callable]):
            Dictionary of descriptors.
        table_name (str):
            Table name to format the described output name. Defaults to ``None``.

    Return:
        pandas.DataFrame:
            2-column DataFrame whose index the name of the descriptors applied to the tables.

    """

    described = list()
    for descriptor in descriptors:
        if isinstance(descriptor, str):
            descriptor, dtypes = DESCRIPTORS[descriptor]
        elif isinstance(descriptor, tuple):
            descriptor, dtypes = descriptor
        else:
            dtypes = ['int', 'float', 'object', 'bool', 'datetime64']

        table_dtypes = pd.Series(metadata.get_dtypes(table_name))
        cols = table_dtypes[table_dtypes.isin(dtypes)].index

        if not cols.empty:
            described.append(
                get_descriptor_values(real.get(cols), synth.get(cols), descriptor, table_name)
            )

    return pd.concat(described, axis=1).fillna(0)


def _validate_arguments(synth, real, metadata, root_path, table_name):
    """Validate arguments before compute descriptors values.

    If ``metadata`` is an instance of dict create the ``Metadata`` object.
    If ``metadata`` is ``None``, validate that ``real`` has to be a ``pandas.DataFrane``.

    If ``real`` is ``None`` load all the tables and assert that ``synth`` is a ``dict``.
    Otherwise, ``real`` and ``synth`` must be of the same type.

    If ``synth`` is not a ``dict``, create a dictionary using the ``table_name``.
    If ``synth`` is not a ``dict``, create a dictionary using the ``table_name``.

    Assert that ``synth`` and ``real`` must have the same tables.

    Args:
        synth (dict or pandas.DataFrame):
        real (dict or pandas.DataFrame):
        metadata (str, dict or Metadata):
        root_path (str):
        table_name (str):

    returns:
        tuple (dict, dict, Metadata):
            Processed tables and Metadata oject.
    """
    if isinstance(metadata, dict):
        metadata = Metadata(metadata, root_path)
    elif metadata is None:
        if not isinstance(real, pd.DataFrame):
            raise TypeError('If metadata is None, `real` has to be a DataFrame')

        metadata = Metadata()
        metadata.add_table(table_name, data=real)

    if real is None:
        real = metadata.load_tables()
        if not isinstance(synth, dict):
            raise TypeError('If `real` is `None`, `synth` must be a dict')

    elif not isinstance(synth, type(real)):
        raise TypeError('`real` and `synth` must be of the same type')

    if not isinstance(synth, dict):
        synth = {table_name: synth}

    if not isinstance(real, dict):
        real = {table_name: real}

    if not set(real.keys()) == set(synth.keys()):
        raise ValueError('real and synthetic dataset must have the same tables')

    return synth, real, metadata


def evaluate(synth, real=None, metadata=None, descriptors=DESCRIPTORS.values(),
             metrics=DEFAULT_METRICS, root_path=None, table_name=None, by_tables=True):
    """Compute stats metric for all tables.

    Args:
        synth (dict[str, pandas.DataFrame] or pandas.DataFrame):
            Map of names and tables of synthesized data.
        real (dict[str, pandas.DataFrame] or pandas.DataFrame):
            Map of names and tables of real data. Defaults to ``None``.
        metadata (str, dict or Metadata):
            String or dictionary to instance a Metadata object or a Metadata itself.
            Defaults to ``None``.
        descriptors (list[callable]):
            List of descriptors.
        metrics (list[callable]):
            List of metrics.
        root_path (str):
            Relative path to find the metadata.json file when needed. Defaults to ``None``.
        table_name (str):
            Table name to be evaluated, only used when ``synth`` is a ``pandas.DataFrame``
            and ``real`` is ``None``. Defaults to None.
        by_tables (bool):
            Flag to return a ``pandas.DataFrame`` when ``True``, ``pandas.Series`` otherwise.

    Return:
        pandas.Series or pandas.DataFrame:
            It has the metrics as index, and the scores as values.
    """
    synth, real, metadata = _validate_arguments(synth, real, metadata, root_path, table_name)

    results = dict()
    for name, real_table in real.items():
        synth_table = synth[name]
        table_scores = get_descriptors_table(
            real_table, synth_table, metadata, name, descriptors)
        results[name] = table_scores

    if not by_tables:
        results = {0: pd.concat(list(results.values()), axis=1)}

    if not isinstance(metrics, (list, tuple)):
        metrics = [metrics]

    scores = dict()
    for name, described in results.items():
        table_scores = dict()
        real_descriptors = described.iloc[0, :]
        synth_descriptors = described.iloc[1, :]
        for metric in metrics:
            table_scores[metric.__name__] = metric(real_descriptors, synth_descriptors)

        scores[name] = table_scores

    scores = pd.DataFrame(scores).T

    if len(scores) > 1:
        mean = scores.mean()
        mean.name = '__mean__'
        scores = scores.append(mean)

    return scores
