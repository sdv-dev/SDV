"""Tools to evaluate the synthesized data."""

import numpy as np
import pandas as pd
from sdmetrics.detection.tabular import LogisticDetector, SVCDetector
from sdmetrics.report import MetricsReport
from sdmetrics.statistical import CSTest, KSTest

from sdv.metadata import Metadata


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

    if len(real.keys()) < len(metadata.get_tables()):
        meta_dict = {
            table: metadata.get_table_meta(table)
            for table in real.keys()
        }
        metadata = Metadata({'tables': meta_dict})

    return synth, real, metadata


def _tabular_metric(sdmetric, synthetic, real, metadata=None, details=False):
    if metadata is None:
        metadata = Metadata()
        metadata.add_table(None, real)
        real = {None: real}
        synthetic = {None: synthetic}

    metrics = sdmetric.metrics(metadata, real, synthetic)
    if details:
        return list(metrics)

    return np.mean([metric.value for metric in metrics])


def _cstest(synthetic, real, metadata=None, details=False):
    return _tabular_metric(CSTest(), synthetic, real, metadata, details)


def _kstest(synthetic, real, metadata=None, details=False):
    return _tabular_metric(KSTest(), synthetic, real, metadata, details)


def _logistic_detection(synthetic, real, metadata=None, details=False):
    return _tabular_metric(LogisticDetector(), synthetic, real, metadata, details)


def _svc_detection(synthetic, real, metadata=None, details=False):
    return _tabular_metric(SVCDetector(), synthetic, real, metadata, details)


METRICS = {
    'cstest': _cstest,
    'kstest': _kstest,
    'logistic_detection': _logistic_detection,
    'svc_detection': _svc_detection,
}


def evaluate(synthetic_data, real_data=None, metadata=None, root_path=None,
             table_name=None, metrics=None, get_report=False, aggregate=True):
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
            Table name to be evaluated, only used when ``synth`` is a ``pandas.DataFrame``
            and ``real`` is ``None``.
        get_report (bool):
            Whether to return the complete SDMetrics report or only the overall average score
            of each metric applied. Defaults to ``False``.
        aggregate (bool):
            If ``get_report`` is ``False``, whether to compute the mean of all the scores to
            return a single float value or return a ``dict`` containing the score that each
            metric obtained. Defaults to ``True``.

    Return:
        float or sdmetrics.MetricsReport
    """
    synth, real, metadata = _validate_arguments(
        synthetic_data, real_data, metadata, root_path, table_name)

    if metrics is None:
        metrics = METRICS.keys()

    computed = {}
    for metric in metrics:
        computed[metric] = METRICS[metric](synth, real, metadata, details=get_report)

    if get_report:
        report = MetricsReport()
        for metrics in computed.values():
            report.add_metrics(metrics)

        return report

    elif aggregate:
        return np.nanmean(list(computed.values()))

    return computed
