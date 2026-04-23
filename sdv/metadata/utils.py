"""Tools to generate strings from regular expressions."""

import json
from pathlib import Path


def read_json(filepath):
    """Validate and open a file path."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise ValueError(
            f"A file named '{filepath.name}' does not exist. Please specify a different filename."
        )

    with open(filepath, 'r', encoding='utf-8') as metadata_file:
        return json.load(metadata_file)


def validate_file_does_not_exist(filepath):
    """Validate a file path doesn't exist."""
    filepath = Path(filepath)
    if filepath.exists():
        raise ValueError(
            f"A file named '{filepath.name}' already exists in this folder. Please specify "
            'a different filename.'
        )


def _validate_file_mode(mode):
    possible_modes = ['write', 'overwrite']
    if mode not in possible_modes:
        raise ValueError(f"Mode '{mode}' must be in {possible_modes}.")


def _format_metadata_value(value):
    """Format a value for display, quoting only strings.

    Args:
        value:
            The value to format. Boolean and None are returned as their
            string representation; all other values are wrapped in single quotes.

    Returns:
        str:
            The formatted value as a string.
    """
    if isinstance(value, bool) or value is None:
        return str(value)
    return f"'{value}'"


def _format_column_metadata(sdtype_info):
    """Format a column's metadata dictionary as a display string, with sdtype first.

    Args:
        sdtype_info (dict):
            A dictionary of column metadata (`{'sdtype': 'ssn', 'pii': False}`).

    Returns:
        str:
            A comma-separated `key=value` string with 'sdtype' first.
            (`sdtype='numerical', computer_representation='Float'`)
    """
    parts = [f'{k}={_format_metadata_value(v)}' for k, v in sdtype_info.items()]
    parts.sort(key=lambda p: not p.startswith('sdtype='))
    return ', '.join(parts)
