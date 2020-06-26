"""Tools to evaluate the synthesized data."""

import pandas as pd
import sdmetrics

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


def evaluate(synth, real=None, metadata=None, root_path=None, table_name=None, get_report=False):
    """Compute a score using SDMetrics.

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
        get_report (bool):
            whether to return the complete SDMetrics report. If False (default), only
            the overall score is returned.

    Return:
        float or sdmetrics.MetricsReport
    """

    synth, real, metadata = _validate_arguments(synth, real, metadata, root_path, table_name)

    report = sdmetrics.evaluate(metadata, real, synth)

    if get_report:
        return report

    return report.overall()
