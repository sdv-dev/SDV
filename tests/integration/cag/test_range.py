import re

import numpy as np
import pandas as pd
import pytest

from sdv.cag import Range
from sdv.cag._errors import PatternNotMetError
from sdv.metadata import Metadata
from sdv.multi_table import HMASynthesizer
from sdv.single_table.copulas import GaussianCopulaSynthesizer
from tests.utils import run_pattern


@pytest.fixture()
def data():
    return pd.DataFrame({
        'A': [1, 2, 3, 1, 2, 1],
        'B': [10, 20, 30, 10, 20, 10],
        'C': [100, 200, 300, 100, 200, 100],
    })


@pytest.fixture()
def metadata():
    return Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'numerical'},
            'B': {'sdtype': 'numerical'},
            'C': {'sdtype': 'numerical'},
        }
    })


@pytest.fixture()
def pattern():
    return Range(
        low_column_name='A',
        middle_column_name='B',
        high_column_name='C',
    )


@pytest.fixture()
def data_datetime():
    return pd.DataFrame({
        'A': [
            pd.Timestamp('2024-01-01'),
            pd.Timestamp('2024-01-02'),
            pd.Timestamp('2024-01-03'),
            pd.Timestamp('2024-01-01'),
            pd.Timestamp('2024-01-02'),
            pd.Timestamp('2024-01-01'),
        ],
        'B': [
            pd.Timestamp('2024-01-02'),
            pd.Timestamp('2024-01-03'),
            pd.Timestamp('2024-01-04'),
            pd.Timestamp('2024-01-05'),
            pd.Timestamp('2024-01-06'),
            pd.Timestamp('2024-01-07'),
        ],
        'C': [
            pd.Timestamp('2024-01-03'),
            pd.Timestamp('2024-01-04'),
            pd.Timestamp('2024-01-05'),
            pd.Timestamp('2024-01-06'),
            pd.Timestamp('2024-01-07'),
            pd.Timestamp('2024-01-08'),
        ],
    })


@pytest.fixture()
def metadata_datetime():
    return Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'datetime'},
            'B': {'sdtype': 'datetime'},
            'C': {'sdtype': 'datetime'},
        }
    })


@pytest.fixture()
def data_multi(data):
    return {
        'table1': data,
        'table2': pd.DataFrame({'id': range(5)}),
    }


@pytest.fixture()
def metadata_multi():
    return Metadata.load_from_dict({
        'tables': {
            'table1': {
                'columns': {
                    'A': {'sdtype': 'numerical'},
                    'B': {'sdtype': 'numerical'},
                    'C': {'sdtype': 'numerical'},
                }
            },
            'table2': {
                'columns': {
                    'id': {'sdtype': 'id'},
                }
            },
        }
    })


@pytest.fixture()
def pattern_multi():
    return Range(
        low_column_name='A',
        middle_column_name='B',
        high_column_name='C',
        table_name='table1',
    )


@pytest.fixture()
def data_multi_datetime(data_datetime):
    return {
        'table1': data_datetime,
        'table2': pd.DataFrame({'id': range(5)}),
    }


@pytest.fixture()
def metadata_multi_datetime():
    return Metadata.load_from_dict({
        'tables': {
            'table1': {
                'columns': {
                    'A': {'sdtype': 'datetime'},
                    'B': {'sdtype': 'datetime'},
                    'C': {'sdtype': 'datetime'},
                }
            },
            'table2': {
                'columns': {
                    'id': {'sdtype': 'id'},
                }
            },
        }
    })


def test_range_pattern_integers(data, metadata, pattern):
    """Test that Range pattern works with integer columns."""
    # Run
    updated_metadata, transformed, reverse_transformed = run_pattern(pattern, data, metadata)

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'numerical'},
            'A#B': {'sdtype': 'numerical'},
            'B#C': {'sdtype': 'numerical'},
            'A#B#C.nan_component': {'sdtype': 'categorical'},
        }
    }).to_dict()
    assert expected_updated_metadata == updated_metadata.to_dict()
    assert list(transformed.columns) == ['A', 'A#B', 'B#C', 'A#B#C.nan_component']
    pd.testing.assert_frame_equal(data, reverse_transformed)


def test_range_pattern_with_nans(metadata, pattern):
    """Test that Range pattern works with NaNs."""
    # Setup
    data = pd.DataFrame({
        'A': [None, 2, np.nan, 1, 2, 1],
        'B': [np.nan, 20, 30, 10, 20, None],
        'C': [None, 200, 300, 100, 200, None],
    })

    # Run
    updated_metadata, transformed, reverse_transformed = run_pattern(pattern, data, metadata)

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'numerical'},
            'A#B': {'sdtype': 'numerical'},
            'B#C': {'sdtype': 'numerical'},
            'A#B#C.nan_component': {'sdtype': 'categorical'},
        }
    }).to_dict()
    assert expected_updated_metadata == updated_metadata.to_dict()
    assert list(transformed.columns) == ['A', 'A#B', 'B#C', 'A#B#C.nan_component']
    pd.testing.assert_frame_equal(
        pd.concat([data.iloc[:2], data.iloc[3:]]),
        pd.concat([reverse_transformed.iloc[:2], reverse_transformed.iloc[3:]]),
    )
    assert 1 < reverse_transformed.iloc[2]['B'] < 30
    assert pd.isna(reverse_transformed.iloc[2]['A'])
    assert 100 < reverse_transformed.iloc[2]['C'] < 300


def test_range_pattern_datetime(data_datetime, metadata_datetime, pattern):
    """Test that Range pattern works with datetime columns."""
    # Setup
    data = data_datetime
    metadata = metadata_datetime

    # Run
    updated_metadata, transformed, reverse_transformed = run_pattern(pattern, data, metadata)

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'datetime'},
            'A#B': {'sdtype': 'numerical'},
            'B#C': {'sdtype': 'numerical'},
            'A#B#C.nan_component': {'sdtype': 'categorical'},
        }
    }).to_dict()
    assert expected_updated_metadata == updated_metadata.to_dict()
    assert list(transformed.columns) == ['A', 'A#B', 'B#C', 'A#B#C.nan_component']

    # Check that the timestamps are very close to each other
    for col in ['A', 'B', 'C']:
        diff = (data[col] - reverse_transformed[col]).abs()
        assert (diff.dt.total_seconds() < 1e-6).all()


def test_range_pattern_datetime_nans(metadata_datetime, pattern):
    """Test that Range pattern works with datetime columns with NaNs."""
    # Setup
    metadata = metadata_datetime
    data = pd.DataFrame({
        'A': [
            np.nan,
            pd.Timestamp('2024-01-02'),
            None,
            pd.Timestamp('2024-01-01'),
            pd.Timestamp('2024-01-02'),
            pd.Timestamp('2024-01-01'),
        ],
        'B': [
            pd.Timestamp('2024-01-02'),
            None,
            np.nan,
            pd.Timestamp('2024-01-05'),
            pd.Timestamp('2024-01-06'),
            pd.Timestamp('2024-01-07'),
        ],
        'C': [
            pd.Timestamp('2024-01-03'),
            None,
            None,
            pd.Timestamp('2024-01-06'),
            pd.Timestamp('2024-01-07'),
            pd.Timestamp('2024-01-08'),
        ],
    })

    # Run
    updated_metadata, transformed, reverse_transformed = run_pattern(pattern, data, metadata)

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'datetime'},
            'A#B': {'sdtype': 'numerical'},
            'B#C': {'sdtype': 'numerical'},
            'A#B#C.nan_component': {'sdtype': 'categorical'},
        }
    }).to_dict()
    assert expected_updated_metadata == updated_metadata.to_dict()
    assert list(transformed.columns) == ['A', 'A#B', 'B#C', 'A#B#C.nan_component']

    pd.testing.assert_series_equal(data['A'], reverse_transformed['A'])
    assert pd.Timestamp('2024-01-01') < reverse_transformed['B'][0] < pd.Timestamp('2024-01-07')
    assert pd.isna(reverse_transformed['B'][1])
    assert pd.isna(reverse_transformed['B'][2])

    diff = (data['B'][3:] - reverse_transformed['B'][3:]).abs()
    assert (diff.dt.total_seconds() < 1e-6).all()

    assert reverse_transformed['B'][3] == reverse_transformed['C'][3] - pd.Timedelta(days=1)

    diff = (data['B'][3:] - reverse_transformed['B'][3:]).abs()
    assert (diff.dt.total_seconds() < 1e-6).all()


def test_range_pattern_with_multi_table(
    data_multi,
    metadata_multi,
):
    """Test that Range pattern works with multi-table data."""
    # Setup
    data = data_multi
    metadata = metadata_multi
    pattern = Range(
        low_column_name='A',
        middle_column_name='B',
        high_column_name='C',
        table_name='table1',
    )

    # Run
    updated_metadata, transformed, reverse_transformed = run_pattern(pattern, data, metadata)

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'tables': {
            'table1': {
                'columns': {
                    'A': {'sdtype': 'numerical'},
                    'A#B': {'sdtype': 'numerical'},
                    'B#C': {'sdtype': 'numerical'},
                    'A#B#C.nan_component': {'sdtype': 'categorical'},
                }
            },
            'table2': {
                'columns': {
                    'id': {'sdtype': 'id'},
                }
            },
        }
    }).to_dict()
    assert expected_updated_metadata == updated_metadata.to_dict()
    assert list(transformed['table1'].columns) == ['A', 'A#B', 'B#C', 'A#B#C.nan_component']
    assert set(data.keys()) == set(reverse_transformed.keys())
    for table_name, table in data.items():
        pd.testing.assert_frame_equal(table, reverse_transformed[table_name])


def test_range_multiple_patterns():
    """Test that Range pattern works with multiple patterns."""
    # Setup
    data = pd.DataFrame({
        'low': [1, 2, 3, 1, 2, 1],
        'mid': [5, 6, 7, 8, 9, 9],
        'high1': [10, 20, 30, 10, 20, 10],
        'high2': [10, 20, 30, 10, 20, 10],
    })
    metadata = Metadata.load_from_dict({
        'columns': {
            'low': {'sdtype': 'numerical'},
            'mid': {'sdtype': 'numerical'},
            'high1': {'sdtype': 'numerical'},
            'high2': {'sdtype': 'numerical'},
        }
    })
    pattern1 = Range(
        low_column_name='low',
        middle_column_name='mid',
        high_column_name='high1',
    )
    pattern2 = Range(
        low_column_name='low',
        middle_column_name='mid',
        high_column_name='high2',
    )

    # Run
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.add_cag(patterns=[pattern1, pattern2])
    synthesizer.fit(data)
    samples = synthesizer.sample(100)
    updated_metadata = synthesizer.get_metadata('modified')
    original_metadata = synthesizer.get_metadata('original')

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'columns': {
            'low': {'sdtype': 'numerical'},
            'low#mid': {'sdtype': 'numerical'},
            'mid#high1': {'sdtype': 'numerical'},
            'high2': {'sdtype': 'numerical'},
            'low#mid#high1.nan_component': {'sdtype': 'categorical'},
        }
    }).to_dict()
    assert expected_updated_metadata == updated_metadata.to_dict()

    assert original_metadata.to_dict() == metadata.to_dict()

    assert all(samples['low'] < samples['mid'])
    assert all(samples['mid'] < samples['high1'])
    assert all(samples['mid'] < samples['high2'])


def test_range_multiple_patterns_different_mid_columns():
    """Test that Range pattern works with multiple patterns."""
    # Setup
    data = pd.DataFrame({
        'low': [1, 2, 3, 1, 2, 1],
        'mid1': [5, 6, 7, 8, 9, 9],
        'mid2': [5, 6, 7, 8, 9, 9],
        'high1': [10, 20, 30, 10, 20, 10],
        'high2': [10, 20, 30, 10, 20, 10],
    })
    metadata = Metadata.load_from_dict({
        'columns': {
            'low': {'sdtype': 'numerical'},
            'mid1': {'sdtype': 'numerical'},
            'mid2': {'sdtype': 'numerical'},
            'high1': {'sdtype': 'numerical'},
            'high2': {'sdtype': 'numerical'},
        }
    })
    pattern1 = Range(
        low_column_name='low',
        middle_column_name='mid1',
        high_column_name='high1',
    )
    pattern2 = Range(
        low_column_name='low',
        middle_column_name='mid2',
        high_column_name='high2',
    )

    # Run
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.add_cag(patterns=[pattern1, pattern2])
    synthesizer.fit(data)
    samples = synthesizer.sample(100)
    updated_metadata = synthesizer.get_metadata('modified')
    original_metadata = synthesizer.get_metadata('original')

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'columns': {
            'low': {'sdtype': 'numerical'},
            'low#mid1': {'sdtype': 'numerical'},
            'mid1#high1': {'sdtype': 'numerical'},
            'mid2#high2': {'sdtype': 'numerical'},
            'low#mid2': {'sdtype': 'numerical'},
            'low#mid1#high1.nan_component': {'sdtype': 'categorical'},
            'low#mid2#high2.nan_component': {'sdtype': 'categorical'},
        }
    }).to_dict()
    assert expected_updated_metadata == updated_metadata.to_dict()

    assert original_metadata.to_dict() == metadata.to_dict()

    assert all(samples['low'] < samples['mid1'])
    assert all(samples['mid1'] < samples['high1'])
    assert all(samples['low'] < samples['mid2'])
    assert all(samples['mid2'] < samples['high2'])


def test_validate_cag(data, metadata, pattern):
    """Test validate_cag works with synthetic data generated with Range."""
    # Setup
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.add_cag(patterns=[pattern])
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample(100)

    # Run
    synthesizer.validate_cag(synthetic_data=synthetic_data)

    # Assert
    assert all(synthetic_data['A'] < synthetic_data['B'])
    assert all(synthetic_data['B'] < synthetic_data['C'])


def test_validate_cag_raises(data, metadata, pattern):
    """Test validate_cag raises an error with bad synthetic data with Range."""
    # Setup
    synthetic_data = pd.DataFrame({
        'A': data['B'],
        'B': data['A'],
        'C': data['C'],
    })
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.add_cag(patterns=[pattern])
    synthesizer.fit(data)
    msg = re.escape('The range requirement is not met for row indices: 0, 1, 2, 3, 4, +1 more')

    # Run and Assert
    with pytest.raises(PatternNotMetError, match=msg):
        synthesizer.validate_cag(synthetic_data=synthetic_data)


def test_validate_cag_multi(
    data_multi,
    metadata_multi,
    pattern_multi,
):
    """Test validate_cag with synthetic data generated with Range with multitable numerical data."""
    data = data_multi
    metadata = metadata_multi
    pattern = pattern_multi
    synthesizer = HMASynthesizer(metadata)
    synthesizer.add_cag(patterns=[pattern])
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample(100)

    # Run
    synthesizer.validate_cag(synthetic_data=synthetic_data)

    # Assert
    assert all(synthetic_data['table1']['A'] < synthetic_data['table1']['B'])
    assert all(synthetic_data['table1']['B'] < synthetic_data['table1']['C'])


def test_validate_cag_multi_datetime(
    data_multi_datetime,
    metadata_multi_datetime,
    pattern_multi,
):
    """Test validate_cag with synthetic data generated with Range with multitable datetime data."""
    # Setup
    data = data_multi_datetime
    metadata = metadata_multi_datetime
    pattern = pattern_multi
    synthesizer = HMASynthesizer(metadata)
    synthesizer.add_cag(patterns=[pattern])
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample(100)

    # Run
    synthesizer.validate_cag(synthetic_data=synthetic_data)

    # Assert
    assert all(synthetic_data['table1']['A'] < synthetic_data['table1']['B'])
    assert all(synthetic_data['table1']['B'] < synthetic_data['table1']['C'])


def test_validate_cag_multi_raises(data_multi, metadata_multi, pattern_multi):
    """Test validate_cag raises an error with bad multitable synthetic data with Range."""
    # Setup
    data = data_multi
    metadata = metadata_multi
    pattern = pattern_multi
    synthetic_data = {
        'table1': {
            'A': data['table1']['B'],
            'B': data['table1']['A'],
            'C': data['table1']['C'],
        },
        'table2': pd.DataFrame({'id': range(5)}),
    }
    synthesizer = HMASynthesizer(metadata)
    synthesizer.add_cag(patterns=[pattern])
    synthesizer.fit(data)
    msg = re.escape(
        "Table 'table1': The range requirement is not met for row indices: 0, 1, 2, 3, 4, +1 more"
    )

    # Run and Assert
    with pytest.raises(PatternNotMetError, match=msg):
        synthesizer.validate_cag(synthetic_data=synthetic_data)
