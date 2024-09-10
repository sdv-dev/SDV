"""Utility functions for the benchmarking."""

import sys

from tests._external.gdrive_utils import read_excel

BENCHMARK_FILE_ID = '1mrvIok6G5P0x88m2_TjOtqcQ-p4iEAk8PTuG6Hpp5Uk'


def get_python_version():
    """Get the current python version."""
    python_version = sys.version_info
    python_version = f'{python_version.major}.{python_version.minor}'
    return python_version


def get_previous_result(dtype, method):
    """Return previous result for a given ``dtype`` and method."""
    data = read_excel(BENCHMARK_FILE_ID)
    python_version = get_python_version()
    df = data[python_version]
    filtered_row = df[df['dtype'] == dtype]
    value = filtered_row[method].to_numpy()[0]
    return value
