import pandas as pd


def _prepare_data_visualization(data, metadata, column_names, sample_size):
    """Prepare the data for a plot.

    Args:
        data (pd.DataFrame or None):
            The data to be prepared.
        metadata (Metadata):
            The metadata of the data.
        column_names (str or list[str]):
            The column names to plot.
        sample_size (int or None):
            The number of samples to plot. If ``None``, use the whole dataset.

    Returns:
        pd.DataFrame or None:
            The prepared data.
    """
    if data is None:
        return None

    col_names = column_names if isinstance(column_names, list) else [column_names]
    data = data.copy()
    for column_name in col_names:
        sdtype = metadata.columns[column_name]['sdtype']
        if sdtype == 'datetime':
            datetime_format = metadata.columns[column_name].get('datetime_format')
            data[column_name] = pd.to_datetime(data[column_name], format=datetime_format)

    if sample_size and sample_size < len(data):
        data = data.sample(n=sample_size)

    return data
