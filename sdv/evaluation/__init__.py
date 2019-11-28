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

from sdv.evaluation.descriptors import DESCRIPTORS, DTypes
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


def _dtype_in_dtypes(a, b):
    for item in b:
        if a == item.value:
            return True

    return False


def get_descriptors_table(real, synth, metadata, descriptors=DESCRIPTORS, table_name=None):
    """Score the synthesized data using the given metrics and descriptors.

    Args:
        real (pandas.DataFrame):
            Table of real data.
        synth (pandas.DataFrame):
            Table of synthesized data.
        descriptors (dict[str, callable]):
            Dictionary of descriptors.

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
            dtypes = DTypes.values()

        if table_name:
            table_dtypes = metadata.get_dtypes(table_name)
            cols = [k for k, v in table_dtypes.items() if _dtype_in_dtypes(v, dtypes)]
        else:
            cols = list(real.columns)

        if cols:
            described.append(
                get_descriptor_values(real.get(cols), synth.get(cols), descriptor, table_name)
            )

    return pd.concat(described, axis=1)


def evaluate(metadata, synth, real=None, descriptors=DESCRIPTORS.values(),
             metrics=DEFAULT_METRICS, root_path=None, table_name=None):
    """Compute stats metric for all tables.

    Args:
        metadata (str, dict or Metadata):
            ...
        real (dict[str, pandas.DataFrame] or pandas.DataFrame):
            Map of names and tables of real data.
        synth (dict[str, pandas.DataFrame] or pandas.DataFrame):
            Map of names and tables of synthesized data.
        descriptors (list[callable]):
            List of descriptors.
        metrics (list[callable]):
            List of metrics.

    Return:
        pandas.Series:
            It has the metrics as index, and the scores as values.
    """
    if not isinstance(metadata, Metadata):
        metadata = Metadata(metadata, root_path)

    if isinstance(real, pd.DataFrame):
        described = get_descriptors_table(real, synth, metadata, descriptors, table_name)

    if isinstance(real, dict):
        if not set(real.keys()) == set(synth.keys()):
            raise ValueError('real and synthetic dataset must have the same tables')

        result = list()
        for name, real_table in real.items():
            synth_table = synth[name]
            result.append(get_descriptors_table(real_table, synth_table, metadata,
                                                descriptors, name))

        described = pd.concat(result, axis=1).fillna(0)

    real_descriptors = described.iloc[0, :]
    synth_descriptors = described.iloc[1, :]
    scores = dict()

    for metric in metrics:
        scores[metric.__name__] = metric(real_descriptors, synth_descriptors)

    return pd.Series(scores), real_descriptors, synth_descriptors
