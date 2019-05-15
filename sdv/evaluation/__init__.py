"""Tools for evaluate the synthesized data."""
from collections import defaultdict

import pandas as pd

from sdv.evaluation.descriptors import DESCRIPTORS
from sdv.evaluation.metrics import DEFAULT_METRICS


def get_descriptor_values(real, synth, descriptors):
    """Compute the descriptor values for the given tables.

    Args:
        real(pandas.DataFrame): Real data.
        synth(pandas.DataFrame): Synthesized data.
        descriptor(callable): Callable that accepts columns and returns real-values.

    Return:
        pandas.DataFrame: It will contain the descriptor output for each column as columns, 
                          .

    """
    real_values = real.apply(descriptor, axis=0).to_frame().T
    synth_values = synth.apply(descriptor, axis=0).to_frame().T

    return pd.concat([real_values, synth_values], axis=0, ignore_index=True)


def score_descriptors_table(
    real, synth, descriptors=DESCRIPTORS.values(), metrics=DEFAULT_METRICS
):
    """Score the synthesized data using the given metrics and descriptors.

    Args:
        real(pandas.DataFrame): Table of real data.
        synth(pandas.DataFrame): Table of synthesized data.
        descriptors(list(callable)): List of descriptors.
        metrics(list(callable)): List of metrics.

    Return:
        pandas.DataFrame:
            DataFrame whose columns are the names of the metrics, as index the name of the
            descriptor, and as a values, the value of the metric applied to the descriptor of both
            tables.

    """
    metric_values = defaultdict(list)
    index = []
    columns = [metric.__name__ for metric in metrics]
    for descriptor in descriptors:
        index.append(descriptor.__name__)
        real_descriptor, synth_descriptor = get_descriptor_values(real, synth, descriptor)
        for metric in metrics:
            metric_value = metric(real_descriptor, synth_descriptor)
            metric_values[metric.__name__].append(metric_value)

    return pd.DataFrame(metric_values, index=index, columns=columns)


def score_descriptors_dataset(
    real, synth, descriptors=DESCRIPTORS.values(), metrics=DEFAULT_METRICS
):
    """Compute stats metric for all tables.

    Args:
        real(dict[str, pandas.DataFrame]): Map of names and tables of real data.
        synth(dict[str, pandas.DataFrame]): Map of names and tables of synthesized data.
        descriptors(list(callable)): List of descriptors.
        metrics(list(callable)): List of metrics.

    Return:
        dict[str, pandas.DataFrame]:
            Dictionary with the table name as keys and as values a DataFrame whose columns are
            the names of the metrics, as index the name of the descriptor, and as a values,
            the value of the metric applied to the descriptor of both tables.

    """
    assert real.keys() == synth.keys(), "real and synthetic dataset must have the same tables"

    result = {}
    for name, real_data in real.items():
        synth_data = synth[name]
        result[name] = score_descriptors_table(
            real_data, synth_data, descriptors=descriptors, metrics=metrics)

    return result
