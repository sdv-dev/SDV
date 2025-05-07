"""Utils for testing."""

import contextlib

import pandas as pd

from sdv.logging import get_sdv_logger
from sdv.metadata.metadata import Metadata


class DataFrameMatcher:
    """Match a given Pandas DataFrame in a mock function call."""

    def __init__(self, df):
        self.df = df

    def __eq__(self, other):
        pd.testing.assert_frame_equal(self.df, other)
        return True


class DataFrameDictMatcher:
    """Match a given dictionary of pandas DataFrames in a mock function call."""

    def __init__(self, data):
        self.data = data

    def __eq__(self, other):
        """Assert the data keys match, then use pandas to assert the values are equal."""
        assert self.data.keys() == other.keys()
        for key in self.data:
            pd.testing.assert_frame_equal(self.data[key], other[key])

        return True


class SeriesMatcher:
    """Match a given Pandas Series in a mock function call."""

    def __init__(self, series):
        self.series = series

    def __eq__(self, other):
        pd.testing.assert_series_equal(self.series, other)
        return True


def get_multi_table_metadata():
    """Return a ``MultiTableMetadata`` object to be used with tests."""
    dict_metadata = {
        'tables': {
            'nesreca': {
                'primary_key': 'id_nesreca',
                'columns': {
                    'upravna_enota': {'sdtype': 'id'},
                    'id_nesreca': {'sdtype': 'id'},
                    'nesreca_val': {'sdtype': 'numerical'},
                },
            },
            'oseba': {
                'columns': {
                    'upravna_enota': {'sdtype': 'id'},
                    'id_nesreca': {'sdtype': 'id'},
                    'oseba_val': {'sdtype': 'numerical'},
                }
            },
            'upravna_enota': {
                'primary_key': 'id_upravna_enota',
                'columns': {
                    'id_upravna_enota': {'sdtype': 'id'},
                    'upravna_val': {'sdtype': 'numerical'},
                },
            },
        },
        'relationships': [
            {
                'parent_table_name': 'upravna_enota',
                'parent_primary_key': 'id_upravna_enota',
                'child_table_name': 'nesreca',
                'child_foreign_key': 'upravna_enota',
            },
            {
                'parent_table_name': 'upravna_enota',
                'parent_primary_key': 'id_upravna_enota',
                'child_table_name': 'oseba',
                'child_foreign_key': 'upravna_enota',
            },
            {
                'parent_table_name': 'nesreca',
                'parent_primary_key': 'id_nesreca',
                'child_table_name': 'oseba',
                'child_foreign_key': 'id_nesreca',
            },
        ],
        'METADATA_SPEC_VERSION': 'MULTI_TABLE_V1',
    }

    return Metadata.load_from_dict(dict_metadata)


def get_multi_table_data():
    """Return a dictionary containing some data for multi table."""
    data = {
        'nesreca': pd.DataFrame({
            'id_nesreca': list(range(4)),
            'upravna_enota': list(range(4)),
            'nesreca_val': list(range(4)),
        }),
        'oseba': pd.DataFrame({
            'upravna_enota': list(range(4)),
            'id_nesreca': list(range(4)),
            'oseba_val': list(range(4)),
        }),
        'upravna_enota': pd.DataFrame({
            'id_upravna_enota': list(range(4)),
            'upravna_val': list(range(4)),
        }),
    }

    return data


@contextlib.contextmanager
def catch_sdv_logs(caplog, level, logger):
    """Context manager to capture logs from an SDV logger."""
    logger = get_sdv_logger(logger)
    orig_level = logger.level
    logger.setLevel(level)
    logger.addHandler(caplog.handler)
    try:
        yield
    finally:
        logger.setLevel(orig_level)
        logger.removeHandler(caplog.handler)


def run_pattern(constraint, data, metadata):
    """Run a constraint."""
    constraint.validate(data, metadata)
    updated_metadata = constraint.get_updated_metadata(metadata)
    constraint.fit(data, metadata)
    transformed = constraint.transform(data)
    reverse_transformed = constraint.reverse_transform(transformed)

    return updated_metadata, transformed, reverse_transformed
