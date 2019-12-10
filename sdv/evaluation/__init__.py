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

DEFAULT_DTYPES = ['int', 'float', 'object', 'bool', 'datetime64']
DESCRIBE_COLUMNS = ['table', 'column', 'descriptor', 'value', 'statistic']
EVALUATE_INDEX = ['table', 'column', 'descriptor', 'value']


def _describe_column(column, descriptor):
    """Compute the descriptor statistics for the given column.

    Args:
        column (pandas.Series):
            Data to describe.
        descriptor (callable):
            Callable that accepts a single column and returns one or more real-values.

    Returns:
        pandas.DataFrame:
            Column statistics as a ``pandas.DataFrame`` with 3 columns:
                * ``statistic``: Value of the statistic.
                * ``value``: specific column value to which the statistic relates.
                * ``descriptor``: Name of the descriptor used to compute the statistics.
    """
    column_stats = pd.Series(descriptor(column))
    column_stats.name = 'statistic'
    column_stats.index.name = 'value'
    column_stats = column_stats.reset_index()
    column_stats['descriptor'] = descriptor.__name__

    return column_stats


def _describe_columns(table_data, descriptor):
    """Compute the descriptor values for the given table.

    Args:
        table_data (pandas.DataFrame):
            Data to describe.
        descriptor (callable):
            Callable that accepts a single column and returns one or more real-values.

    Returns:
        pandas.DataFrame:
            Table statistics as a ``pandas.DataFrame`` with 4 columns:
                * ``statistic``: Value of the statistic.
                * ``value``: specific column value to which the statistic relates.
                * ``descriptor``: Name of the descriptor used to compute the statistics.
                * ``column``: Name of the column being described.
    """
    table_stats = list()

    for column_name in table_data.columns:
        try:
            column_stats = _describe_column(table_data[column_name], descriptor)
            column_stats['column'] = column_name
            table_stats.append(column_stats)
        except TypeError:
            pass

    return pd.concat(table_stats, ignore_index=True, sort=False)


def _describe_table(table_data, table_dtypes, descriptors):
    """Get stats for the given table using the descriptors.

    Args:
        table (pandas.DataFrame):
            Table to describe.
        table_dtypes (list[str]):
            List of column dtypes extracted from the table metadata.
        descriptors (list[callabel, list]):
           List of descriptors and supported dtypes.

    Return:
        pandas.DataFrame:
            Table statistics as a ``pandas.DataFrame`` with 4 columns:
                * ``statistic``: Value of the statistic.
                * ``value``: specific column value to which the statistic relates.
                * ``descriptor``: Name of the descriptor used to compute the statistics.
                * ``column``: Name of the column being described.
    """
    table_stats = list()
    for descriptor, dtypes in descriptors:
        valid_columns = table_dtypes[table_dtypes.isin(dtypes)].index
        if not valid_columns.empty:
            table_stats.append(_describe_columns(table_data[valid_columns], descriptor))

    return pd.concat(table_stats, ignore_index=True, sort=False)


def _get_descriptor_tuples(descriptors):
    if not descriptors:
        return list(DESCRIPTORS.values())

    tuples = list()
    for descriptor in descriptors:
        if isinstance(descriptor, str):
            descriptor, dtypes = DESCRIPTORS[descriptor]
        elif isinstance(descriptor, tuple):
            descriptor, dtypes = descriptor
        else:
            dtypes = DEFAULT_DTYPES

        tuples.append((descriptor, dtypes))

    return tuples


def describe(tables, metadata, descriptors=None):
    """Compute statistics for all tables.

    Args:
        tables (dict[str, pandas.DataFrame]):
            Mapping of table names and the corresponding pd.DataFrames.
        metadata (Metadata):
            Dataset Metadata.
        descriptors (list[callable]):
            List of descriptors.

    Return:
        pandas.Series or pandas.DataFrame:
            It has the metrics as index, and the scores as values.
    """
    stats = list()
    descriptors = _get_descriptor_tuples(descriptors)
    for table_name, table_data in tables.items():
        table_dtypes = pd.Series(metadata.get_dtypes(table_name))
        table_stats = _describe_table(table_data, table_dtypes, descriptors)
        table_stats['table'] = table_name
        stats.append(table_stats)

    return pd.concat(stats, ignore_index=True, sort=False)[DESCRIBE_COLUMNS]


def _validate_arguments(synth, real, metadata, root_path, table_name):
    """Validate arguments needed to compute descriptors values.

    If ``metadata`` is an instance of dict create the ``Metadata`` object.
    If ``metadata`` is ``None``, ``real`` has to be a ``pandas.DataFrane``.

    If ``real`` is ``None`` load all the tables and assert that ``synth`` is a ``dict``.
    Otherwise, ``real`` and ``synth`` must be of the same type.

    If ``synth`` is not a ``dict``, create a dictionary using the ``table_name``.

    Assert that ``synth`` and ``real`` must have the same tables.

    Args:
        synth (dict or pandas.DataFrame):
            Synthesized data.
        real (dict, pandas.DataFrame or None):
            Real data.
        metadata (str, dict, Metadata or None):
            Metadata instance or details needed to build it.
        root_path (str):
            Path to the metadata file.
        table_name (str):
            Table name used to prepare the metadata object, real and synth dict.

    Returns:
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


def evaluate(synth, real=None, metadata=None, root_path=None, descriptors=DESCRIPTORS.keys(),
             metrics=DEFAULT_METRICS, table_name=None, by_tables=False):
    """Compute stats metric for all tables.

    Args:
        synth (dict[str, pandas.DataFrame] or pandas.DataFrame):
            Map of names and tables of synthesized data.
        real (dict[str, pandas.DataFrame] or pandas.DataFrame):
            Map of names and tables of real data.
        metadata (str, dict, Metadata or None):
            Metadata instance or details needed to build it.
        root_path (str):
            Relative path to find the metadata.json file when needed.
        descriptors (list[callable]):
            List of descriptors.
        metrics (list[callable]):
            List of metrics.
        table_name (str):
            Table name to be evaluated, only used when ``synth`` is a ``pandas.DataFrame``
            and ``real`` is ``None``.
        by_tables (bool):
            Whether to compute the metrics by table or over the whole dataset.

    Return:
        pandas.Series or pandas.DataFrame:
            It has the metrics as index, and the scores as values.
    """
    synth, real, metadata = _validate_arguments(synth, real, metadata, root_path, table_name)

    real_stats = describe(real, metadata)
    synth_stats = describe(synth, metadata)

    stats = pd.DataFrame({
        'real': real_stats.set_index(EVALUATE_INDEX)['statistic'],
        'synth': synth_stats.set_index(EVALUATE_INDEX)['statistic']
    }).fillna(0).reset_index()

    if not by_tables:
        stats['table'] = 0

    if not isinstance(metrics, (list, tuple)):
        metrics = [metrics]

    scores = dict()
    for table in stats['table'].unique():
        table_scores = dict()
        table_stats = stats[stats['table'] == table]
        real_stats = table_stats['real']
        synth_stats = table_stats['synth']
        for metric in metrics:
            table_scores[metric.__name__] = metric(real_stats, synth_stats)

        scores[table] = table_scores

    scores = pd.DataFrame(scores).T

    if len(scores) > 1:
        mean = scores.mean()
        mean.name = 'mean'
        scores = scores.append(mean)

    return scores
