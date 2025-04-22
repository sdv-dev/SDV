import numpy as np
import pandas as pd

from sdv.cag import Range
from sdv.metadata import Metadata
from sdv.single_table.copulas import GaussianCopulaSynthesizer
from tests.utils import run_pattern


def test_range_pattern_integers():
    """Test that Range pattern works with integer columns."""
    # Setup
    data = pd.DataFrame({
        'A': [1, 2, 3, 1, 2, 1],
        'B': [10, 20, 30, 10, 20, 10],
        'C': [100, 200, 300, 100, 200, 100],
    })
    metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'numerical'},
            'B': {'sdtype': 'numerical'},
            'C': {'sdtype': 'numerical'},
        }
    })
    pattern = Range(
        low_column_name='A',
        middle_column_name='B',
        high_column_name='C',
    )

    # Run
    updated_metadata, transformed, reverse_transformed = run_pattern(pattern, data, metadata)

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'numerical'},
            'A#B': {'sdtype': 'numerical'},
            'B#C': {'sdtype': 'numerical'},
        }
    }).to_dict()
    assert expected_updated_metadata == updated_metadata.to_dict()
    assert list(transformed.columns) == ['A', 'A#B', 'B#C']
    pd.testing.assert_frame_equal(data, reverse_transformed)


def test_range_pattern_with_nans():
    """Test that Range pattern works with NaNs."""
    # Setup
    data = pd.DataFrame({
        'A': [None, 2, np.nan, 1, 2, 1],
        'B': [np.nan, 20, 30, 10, 20, None],
        'C': [None, 200, 300, 100, 200, None],
    })
    metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'numerical'},
            'B': {'sdtype': 'numerical'},
            'C': {'sdtype': 'numerical'},
        }
    })

    pattern = Range(
        low_column_name='A',
        middle_column_name='B',
        high_column_name='C',
    )

    # Run
    updated_metadata, transformed, reverse_transformed = run_pattern(pattern, data, metadata)

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'numerical'},
            'A#B': {'sdtype': 'numerical'},
            'B#C': {'sdtype': 'numerical'},
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


def test_range_pattern_datetime():
    """Test that Range pattern works with datetime columns."""
    # Setup
    data = pd.DataFrame({
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
    metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'datetime'},
            'B': {'sdtype': 'datetime'},
            'C': {'sdtype': 'datetime'},
        }
    })
    pattern = Range(
        low_column_name='A',
        middle_column_name='B',
        high_column_name='C',
    )

    # Run
    updated_metadata, transformed, reverse_transformed = run_pattern(pattern, data, metadata)

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'datetime'},
            'A#B': {'sdtype': 'numerical'},
            'B#C': {'sdtype': 'numerical'},
        }
    }).to_dict()
    assert expected_updated_metadata == updated_metadata.to_dict()
    assert list(transformed.columns) == ['A', 'A#B', 'B#C']

    # Check that the timestamps are very close to each other
    for col in ['A', 'B', 'C']:
        diff = (data[col] - reverse_transformed[col]).abs()
        assert (diff.dt.total_seconds() < 1e-6).all()


def test_range_pattern_datetime_nans():
    """Test that Range pattern works with datetime columns with NaNs."""
    # Setup
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
    metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'datetime'},
            'B': {'sdtype': 'datetime'},
            'C': {'sdtype': 'datetime'},
        }
    })
    pattern = Range(
        low_column_name='A',
        middle_column_name='B',
        high_column_name='C',
    )

    # Run
    updated_metadata, transformed, reverse_transformed = run_pattern(pattern, data, metadata)

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'datetime'},
            'A#B': {'sdtype': 'numerical'},
            'B#C': {'sdtype': 'numerical'},
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


def test_range_pattern_with_multi_table():
    """Test that Range pattern works with multi-table data."""
    # Setup
    data = {
        'table1': pd.DataFrame({
            'A': [1, 2, 3, 1, 2, 1],
            'B': [10, 20, 30, 10, 20, 10],
            'C': [100, 200, 300, 100, 200, 100],
        }),
        'table2': pd.DataFrame({'id': range(5)}),
    }
    metadata = Metadata.load_from_dict({
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
    assert list(transformed['table1'].columns) == ['A', 'A#B', 'B#C']
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
        }
    }).to_dict()
    assert expected_updated_metadata == updated_metadata.to_dict()

    assert original_metadata.to_dict() == metadata.to_dict()

    assert all(samples['low'] < samples['mid1'])
    assert all(samples['mid1'] < samples['high1'])
    assert all(samples['low'] < samples['mid2'])
    assert all(samples['mid2'] < samples['high2'])
