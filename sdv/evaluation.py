"""Tools to evaluate the synthesized data."""

import pandas as pd
import sdmetrics

from sdv.metadata.dataset import Metadata


def _validate_arguments(synthetic_data, real_data, metadata, root_path, table_name):
    """Validate arguments needed to compute descriptors values.

    If ``metadata`` is an instance of dict create the ``Metadata`` object.
    If ``metadata`` is ``None``, ``real_data`` has to be a ``pandas.DataFrane``.

    If ``real_data`` is ``None`` load all the tables and assert that ``synthetic_data`` is
    a ``dict``. Otherwise, ``real_data`` and ``synthetic_data`` must be of the same type.

    If ``synthetic_data`` is not a ``dict``, create a dictionary using the ``table_name``.

    Assert that ``synthetic_data`` and ``real_data`` must have the same tables.

    Args:
        synthetic_data (dict or pandas.DataFrame):
            Synthesized data.
        real_data (dict, pandas.DataFrame or None):
            Real data.
        metadata (str, dict, Metadata or None):
            Metadata instance or details needed to build it.
        root_path (str):
            Path to the metadata file.
        table_name (str):
            Table name used to prepare the metadata object, real_data and synthetic_data dict.

    Returns:
        tuple (dict, dict, Metadata):
            Processed tables and Metadata oject.
    """
    if isinstance(metadata, dict):
        metadata = Metadata(metadata, root_path)
    elif metadata is None:
        if not isinstance(real_data, pd.DataFrame):
            raise TypeError('If metadata is None, `real_data` has to be a DataFrame')

        metadata = Metadata()
        metadata.add_table(table_name, data=real_data)

    if real_data is None:
        real_data = metadata.load_tables()
        if not isinstance(synthetic_data, dict):
            raise TypeError('If `real_data` is `None`, `synthetic_data` must be a dict')

    elif not isinstance(synthetic_data, type(real_data)):
        raise TypeError('`real_data` and `synthetic_data` must be of the same type')

    if not isinstance(synthetic_data, dict):
        synthetic_data = {table_name: synthetic_data}

    if not isinstance(real_data, dict):
        real_data = {table_name: real_data}

    if not set(real_data.keys()) == set(synthetic_data.keys()):
        raise ValueError('real_data and synthetic dataset must have the same tables')

    if len(real_data.keys()) < len(metadata.get_tables()):
        meta_dict = {
            table: metadata.get_table_meta(table)
            for table in real_data.keys()
        }
        metadata = Metadata({'tables': meta_dict})

    return synthetic_data, real_data, metadata.to_dict()


def _select_metrics(synthetic_data, metrics):
    if isinstance(synthetic_data, dict):
        modality = 'multi-table'
        metric_classes = sdmetrics.multi_table.MultiTableMetric.get_subclasses()
    else:
        modality = 'single-table'
        metric_classes = sdmetrics.single_table.SingleTableMetric.get_subclasses()

    if metrics is None:
        return metric_classes, modality

    final_metrics = {}
    for metric in metrics:
        if isinstance(metric, str):
            try:
                final_metrics[metric] = metric_classes[metric]
            except KeyError:
                raise ValueError(f'Unknown {modality} metric: {metric}')

    return final_metrics, modality


def evaluate(synthetic_data, real_data=None, metadata=None, root_path=None,
             table_name=None, metrics=None, aggregate=True):
    """Apply multiple metrics at once.

    Args:
        synthetic_data (dict[str, pandas.DataFrame] or pandas.DataFrame):
            Map of names and tables of synthesized data. When evaluating a single table,
            a single ``pandas.DataFrame`` can be passed alone.
        real_data (dict[str, pandas.DataFrame] or pandas.DataFrame):
            Map of names and tables of real data. When evaluating a single table,
            a single ``pandas.DataFrame`` can be passed alone.
        metadata (str, dict, Metadata or None):
            Metadata instance or details needed to build it.
        root_path (str):
            Relative path to find the metadata.json file when needed.
        metrics (list[str]):
            List of metric names to apply.
        table_name (str):
            Table name to be evaluated, only used when ``synthetic_data`` is a
            ``pandas.DataFrame`` and ``real_data`` is ``None``.
        aggregate (bool):
            If ``get_report`` is ``False``, whether to compute the mean of all the scores to
            return a single float value or return a ``dict`` containing the score that each
            metric obtained. Defaults to ``True``.

    Return:
        float or sdmetrics.MetricsReport
    """
    metrics, modality = _select_metrics(synthetic_data, metrics)

    synthetic_data, real_data, metadata = _validate_arguments(
        synthetic_data, real_data, metadata, root_path, table_name)

    if modality == 'single-table':
        table = list(metadata['tables'].keys())[0]
        metadata = metadata['tables'][table]
        real_data = real_data[table]
        synthetic_data = synthetic_data[table]

    scores = sdmetrics.compute_metrics(metrics, real_data, synthetic_data, metadata=metadata)
    scores.dropna(inplace=True)

    if aggregate:
        return scores.score.mean()

    return scores
