"""Tools to generate strings from regular expressions."""

import json
from pathlib import Path


def read_json(filepath):
    """Validate and open a file path."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise ValueError(
            f"A file named '{filepath.name}' does not exist. "
            'Please specify a different filename.'
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
