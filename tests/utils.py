"""Utils for testing."""

import contextlib
from copy import deepcopy
from functools import lru_cache

import pandas as pd
from rdt.transformers.utils import learn_rounding_digits

from sdv.datasets.demo import download_demo
from sdv.logging import get_sdv_logger
from sdv.metadata.metadata import Metadata
from sdv.multi_table import HMASynthesizer
from sdv.single_table import GaussianCopulaSynthesizer

RANGE_KEYS = {
    'range_is_nullable',
    'range_min',
    'range_max',
    'range_values',
    'decimal_places',
}


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
    """Return a multi-table ``Metadata`` object to be used with tests."""
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
        'METADATA_SPEC_VERSION': 'V1',
    }

    return Metadata.load_from_dict(dict_metadata)


def get_simplified_multi_table_metadata():
    """Return a simplified ``Metadata`` object to be used with HMA tests."""
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
        'METADATA_SPEC_VERSION': 'V1',
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


def run_constraint(constraint, data, metadata):
    """Run a constraint."""
    constraint.validate(data, metadata)
    updated_metadata = constraint.get_updated_metadata(metadata)
    constraint.fit(data, metadata)
    transformed = constraint.transform(data)
    reverse_transformed = constraint.reverse_transform(transformed)

    return updated_metadata, transformed, reverse_transformed


def run_copula(data, metadata, constraints=None):
    synthesizer = GaussianCopulaSynthesizer(metadata)
    if constraints:
        synthesizer.add_constraints(constraints=constraints)
    synthesizer.fit(data)

    return synthesizer


def run_hma(data, metadata, constraints=None):
    synthesizer = HMASynthesizer(metadata)
    if constraints:
        synthesizer.add_constraints(constraints=constraints)
    synthesizer.fit(data)

    return synthesizer


@lru_cache
def _download_demo(modality, dataset_name):
    return download_demo(modality, dataset_name)


def download_test_demo(modality, dataset_name):
    """Download demo datasets with caching.

    Args:
        modality:
            The modality of the dataset: 'single_table', 'multi_table', 'sequential'.
        dataset_name:
            Name of the dataset to download.
    """
    data, metadata = _download_demo(modality, dataset_name)
    return deepcopy(data), deepcopy(metadata)


def compare_metadata(metadata, expected_metadata):
    """Compare metadata, allowing detected range fields to be omitted from expected metadata."""
    actual = metadata.to_dict() if isinstance(metadata, Metadata) else deepcopy(metadata)
    expected = (
        expected_metadata.to_dict()
        if isinstance(expected_metadata, Metadata)
        else deepcopy(expected_metadata)
    )

    for table_name, table in actual['tables'].items():
        for column_name, column in table['columns'].items():
            expected_column = expected['tables'][table_name]['columns'][column_name]
            for key in RANGE_KEYS:
                if key not in expected_column:
                    column.pop(key, None)

    assert actual == expected


def compare_ranges(metadata, data):
    """Check that detected ranges are consistent with the source data."""
    metadata = metadata.to_dict() if isinstance(metadata, Metadata) else metadata
    for table_name, table in metadata['tables'].items():
        primary_key = table.get('primary_key')
        primary_keys = {primary_key} if isinstance(primary_key, str) else set(primary_key or [])
        for column_name, column in table['columns'].items():
            sdtype = column.get('sdtype')
            range_keys = set(column) & RANGE_KEYS
            if column_name in primary_keys or sdtype == 'unknown':
                assert not range_keys
                continue

            column_data = data[table_name][column_name]
            clean_data = column_data.dropna()

            if 'range_is_nullable' in column:
                assert column['range_is_nullable'] == column_data.isna().any()

            if 'range_values' in column:
                assert set(column['range_values']) == set(clean_data)

            if 'range_min' in column:
                if column['sdtype'] == 'datetime':
                    assert pd.to_datetime(column['range_min']) == pd.to_datetime(clean_data).min()
                    assert pd.to_datetime(column['range_max']) == pd.to_datetime(clean_data).max()
                else:
                    assert column['range_min'] == clean_data.min()
                    assert column['range_max'] == clean_data.max()

            if 'decimal_places' in column:
                assert column['decimal_places'] == learn_rounding_digits(column_data)
