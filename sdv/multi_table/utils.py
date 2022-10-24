"""Utility functions for handling multi-table data."""

from pathlib import Path

from sdv.utils import load_data_from_csv


def validate_file_exists(filepath, table_name):
    """Validate a file path doesn't exist."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise ValueError(
            f"No data found for table '{table_name}'. Please check the filepath ('{filepath}')."
        )


def load_from_csv(table_name_to_filepath):
    """Load csv files from specified filepaths.

    Args:
        table_name_to_filepath (dict):
            Dictionary mapping the table names to the file paths where their data is located.
    """
    data = {}
    for table_name, path in table_name_to_filepath.items():
        validate_file_exists(path, table_name)
        table_data = load_data_from_csv(path)
        data[table_name] = table_data

    return data
