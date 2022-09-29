"""Utility functions for tabular models."""

import numpy as np


def detect_discrete_columns(metadata, data):
    """Detect th discrete columns in a dataset.

    Args:
        metadata (sdv.metadata.SingleTableMetadata):
            Metadata that belongs to the given ``data``.

        data (pandas.DataFrame):
            ``pandas.DataFrame`` that matches the ``metadata``.

    Returns:
        discrete_columns (list):
            A list of discrete columns to be used with some of ``sdv`` synthesizers.
    """
    discrete_columns = []

    for column in data.columns:
        if column in metadata._columns:
            if metadata._columns[column]['sdtype'] == 'categorical':
                discrete_columns.append(column)

        else:
            column_data = data[column].dropna()
            dtype = column_data.infer_objects().dtype
            try:
                kind = np.dtype(dtype).kind
            except TypeError:
                kind = 'O'

            if kind in ['O', 'b']:
                discrete_columns.append(column)

    return discrete_columns
