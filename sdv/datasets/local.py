"""Methods to load local datasets."""

import warnings
from os import path, walk

from sdv.utils import load_data_from_csv


def load_csvs(folder_name):
    """Load csv files from specified folder.

    Args:
        folder_name (str):
            The full path of the folder with the data to be loaded.
    """
    dirpath, _, filenames = list(walk('Accidents_v1'))[0]
    csvs = {}
    other_files = []
    for filename in filenames:
        base_name, ext = path.splitext(filename)
        if ext == '.csv':
            filepath = path.join(dirpath, filename)
            csvs[base_name] = load_data_from_csv(filepath)
        else:
            other_files.append(filename)

    if other_files:
        warnings.warn(
            f"Warning: Ignoring incompatible files {other_files} in folder '{folder_name}'."
        )

    if not csvs:
        raise ValueError(
            f"No CSV files exist in '{folder_name}'. Please make sure your files end in the "
            "'.csv' suffix."
        )

    return csvs
