"""Utility functions for tabular models."""


def detect_discrete_columns(metadata, data):
    """Detect the discrete columns in a dataset.

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
            if metadata._columns[column]['sdtype'] not in ['numerical', 'datetime']:
                discrete_columns.append(column)

        else:
            column_data = data[column].dropna()
            if set(column_data.unique()) == {0.0, 1.0}:
                column_data = column_data.astype(bool)

            try:
                dtype = column_data.infer_objects().dtype.kind
                if dtype in ['O', 'b']:
                    discrete_columns.append(column)

            except Exception:
                discrete_columns.append(column)

    return discrete_columns
