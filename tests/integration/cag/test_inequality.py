import numpy as np
import pandas as pd
import pytest

from sdv.cag import Inequality
from sdv.cag._errors import PatternNotMetError
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer
from tests.utils import run_pattern


def test_inequality_pattern_integers():
    """Test that Inequality pattern works with integer columns."""
    # Setup
    data = pd.DataFrame({
        'A': [1, 2, 3, 1, 2, 1],
        'B': [10, 20, 30, 10, 20, 10],
    })
    metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'numerical'},
            'B': {'sdtype': 'numerical'},
        }
    })
    pattern = Inequality(
        low_column_name='A',
        high_column_name='B',
    )

    # Run
    updated_metadata, transformed, reverse_transformed = run_pattern(pattern, data, metadata)

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'numerical'},
            'A#B': {'sdtype': 'numerical'},
        }
    }).to_dict()
    assert expected_updated_metadata == updated_metadata.to_dict()
    assert list(transformed.columns) == ['A', 'A#B']
    pd.testing.assert_frame_equal(data, reverse_transformed)


def test_inequality_pattern_with_nans():
    """Test that Inequality pattern works with NaNs."""
    # Setup
    data = pd.DataFrame({
        'A': [None, 2, np.nan, 1, 2, 1],
        'B': [np.nan, 20, 30, 10, 20, None],
    })
    metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'numerical'},
            'B': {'sdtype': 'numerical'},
        }
    })

    pattern = Inequality(
        low_column_name='A',
        high_column_name='B',
    )

    # Run
    updated_metadata, transformed, reverse_transformed = run_pattern(pattern, data, metadata)

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'numerical'},
            'A#B': {'sdtype': 'numerical'},
        }
    }).to_dict()
    assert expected_updated_metadata == updated_metadata.to_dict()
    assert list(transformed.columns) == ['A', 'A#B', 'A#B.nan_component']
    pd.testing.assert_frame_equal(
        pd.concat([data.iloc[:2], data.iloc[3:]]),
        pd.concat([reverse_transformed.iloc[:2], reverse_transformed.iloc[3:]]),
    )
    assert 1 < reverse_transformed.iloc[2]['B'] < 30
    assert pd.isna(reverse_transformed.iloc[2]['A'])


def test_inequality_pattern_datetime():
    """Test that Inequality pattern works with datetime columns."""
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
    })
    metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'datetime'},
            'B': {'sdtype': 'datetime'},
        }
    })
    pattern = Inequality(
        low_column_name='A',
        high_column_name='B',
    )

    # Run
    updated_metadata, transformed, reverse_transformed = run_pattern(pattern, data, metadata)

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'datetime'},
            'A#B': {'sdtype': 'numerical'},
        }
    }).to_dict()
    assert expected_updated_metadata == updated_metadata.to_dict()
    assert list(transformed.columns) == ['A', 'A#B']

    # Check that the timestamps are very close to each other
    for col in ['A', 'B']:
        diff = (data[col] - reverse_transformed[col]).abs()
        assert (diff.dt.total_seconds() < 1e-6).all()


def test_inequality_pattern_datetime_nans():
    """Test that Inequality pattern works with datetime columns with NaNs."""
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
    })
    metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'datetime'},
            'B': {'sdtype': 'datetime'},
        }
    })
    pattern = Inequality(
        low_column_name='A',
        high_column_name='B',
    )

    # Run
    updated_metadata, transformed, reverse_transformed = run_pattern(pattern, data, metadata)

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'datetime'},
            'A#B': {'sdtype': 'numerical'},
        }
    }).to_dict()
    assert expected_updated_metadata == updated_metadata.to_dict()
    assert list(transformed.columns) == ['A', 'A#B', 'A#B.nan_component']

    pd.testing.assert_series_equal(data['A'], reverse_transformed['A'])
    assert pd.Timestamp('2024-01-01') < reverse_transformed['B'][0] < pd.Timestamp('2024-01-07')
    assert pd.isna(reverse_transformed['B'][1])
    assert pd.isna(reverse_transformed['B'][2])

    diff = (data['B'][3:] - reverse_transformed['B'][3:]).abs()
    assert (diff.dt.total_seconds() < 1e-6).all()


def test_inequality_pattern_with_multi_table():
    """Test that Inequality pattern works with multi-table data."""
    # Setup
    data = {
        'table1': pd.DataFrame({
            'A': [1, 2, 3, 1, 2, 1],
            'B': [10, 20, 30, 10, 20, 10],
        }),
        'table2': pd.DataFrame({'id': range(5)}),
    }
    metadata = Metadata.load_from_dict({
        'tables': {
            'table1': {
                'columns': {
                    'A': {'sdtype': 'numerical'},
                    'B': {'sdtype': 'numerical'},
                }
            },
            'table2': {
                'columns': {
                    'id': {'sdtype': 'id'},
                }
            },
        }
    })
    pattern = Inequality(
        low_column_name='A',
        high_column_name='B',
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
    assert list(transformed['table1'].columns) == ['A', 'A#B']
    assert set(data.keys()) == set(reverse_transformed.keys())
    for table_name, table in data.items():
        pd.testing.assert_frame_equal(table, reverse_transformed[table_name])


def test_inequality_with_numerical():
    """Test it works with numerical columns."""
    # Setup
    data = pd.DataFrame({
        'A': [1, 2, 3, 1, 2, 1],
        'B': [10, 20, 30, 10, 20, 10],
    })
    metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'numerical'},
            'B': {'sdtype': 'numerical'},
        }
    })
    pattern = Inequality(
        low_column_name='A',
        high_column_name='B',
    )
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.add_cag(patterns=[pattern])

    # Run
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample(num_rows=10)

    # Assert
    assert (synthetic_data['A'] < synthetic_data['B']).all()
    assert len(synthetic_data) == 10


def test_inequality_with_timestamp_and_date():
    """Test that the inequality pattern passes without strict boundaries.

    This test checks if the `Inequality` pattern can handle two columns
    with different datetime formats when `strict_boundaries` is set to `False`.
    The pattern allows the `SUBMISSION_TIMESTAMP` column to be less than
    or equal to the `DUE_DATE` column, even when they differ in precision but end
    within the same day.
    """
    # Setup
    data = pd.DataFrame(
        data={
            'SUBMISSION_TIMESTAMP': [
                '2016-07-10 17:04:00',
                '2016-07-11 13:23:00',
                '2016-07-12 08:45:30',
                '2016-07-11 12:00:00',
                '2016-07-12 10:30:00',
            ],
            'DUE_DATE': ['2016-07-10', '2016-07-11', '2016-07-12', '2016-07-13', '2016-07-14'],
        }
    )

    metadata = Metadata.load_from_dict({
        'tables': {
            'table': {
                'columns': {
                    'SUBMISSION_TIMESTAMP': {
                        'sdtype': 'datetime',
                        'datetime_format': '%Y-%m-%d %H:%M:%S',
                    },
                    'DUE_DATE': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
                }
            }
        }
    })
    synthesizer = GaussianCopulaSynthesizer(metadata)
    pattern = Inequality(
        low_column_name='SUBMISSION_TIMESTAMP',
        high_column_name='DUE_DATE',
        strict_boundaries=False,
    )

    synthesizer.add_cag(patterns=[pattern])

    # Run
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample(num_rows=10)

    # Assert
    synthetic_data['SUBMISSION_TIMESTAMP'] = pd.to_datetime(
        synthetic_data['SUBMISSION_TIMESTAMP'], errors='coerce'
    )
    synthetic_data['DUE_DATE'] = pd.to_datetime(synthetic_data['DUE_DATE'], errors='coerce')
    invalid_rows = synthetic_data[
        synthetic_data['SUBMISSION_TIMESTAMP'].dt.date > synthetic_data['DUE_DATE'].dt.date
    ]
    assert invalid_rows.empty


def test_inequality_with_timestamp_and_date_strict_boundaries():
    """Test that the inequality pattern fails with strict boundaries.

    This test evaluates the `Inequality` pattern when `strict_boundaries`
    is set to `True`. The `SUBMISSION_TIMESTAMP` column values must be strictly
    less than the `DUE_DATE` values to satisfy the pattern. If any
    `SUBMISSION_TIMESTAMP` matches or exceeds the `DUE_DATE`, an error should
    be raised.
    """
    # Setup
    data = pd.DataFrame(
        data={
            'SUBMISSION_TIMESTAMP': [
                '2016-07-10 17:04:00',
                '2016-07-11 13:23:00',
                '2016-07-12 08:45:30',
                '2016-07-11 12:00:00',
                '2016-07-12 10:30:00',
            ],
            'DUE_DATE': ['2016-07-10', '2016-07-11', '2016-07-12', '2016-07-13', '2016-07-14'],
        }
    )

    metadata = Metadata.load_from_dict({
        'tables': {
            'table': {
                'columns': {
                    'SUBMISSION_TIMESTAMP': {
                        'sdtype': 'datetime',
                        'datetime_format': '%Y-%m-%d %H:%M:%S',
                    },
                    'DUE_DATE': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
                }
            }
        }
    })

    synthesizer = GaussianCopulaSynthesizer(metadata)
    pattern = Inequality(
        low_column_name='SUBMISSION_TIMESTAMP',
        high_column_name='DUE_DATE',
        strict_boundaries=True,
    )
    synthesizer.add_cag(patterns=[pattern])

    # Run and Assert
    error_msg = 'The inequality requirement is not met for row indices: '
    with pytest.raises(PatternNotMetError) as error:
        synthesizer.fit(data)
        assert error_msg in error


def test_inequality_pattern_date_less_than_timestamp_strict_boundaries():
    """Test that the inequality pattern fails when date is less than timestamp.

    This case evaluates the `Inequality` pattern with
    `strict_boundaries=True` when the `DUE_DATE` column contains dates, and the
    `SUBMISSION_TIMESTAMP` contains a timestamp. Since `DUE_DATE` lacks
    time precision, and `SUBMISSION_TIMESTAMP` contains a timestamp, it
    violates the pattern when the date is less than the timestamp.
    """
    # Setup
    data = pd.DataFrame(
        data={
            'SUBMISSION_TIMESTAMP': [
                '2024-11-12 02:34:45',
                '2024-11-13 10:45:30',
                '2024-11-14 14:30:00',
                '2024-11-15 16:20:10',
                '2024-11-16 08:00:00',
            ],
            'DUE_DATE': ['2024-11-12', '2024-11-13', '2024-11-14', '2024-11-15', '2024-11-16'],
        }
    )

    metadata = Metadata.load_from_dict({
        'tables': {
            'table': {
                'columns': {
                    'SUBMISSION_TIMESTAMP': {
                        'sdtype': 'datetime',
                        'datetime_format': '%Y-%m-%d %H:%M:%S',
                    },
                    'DUE_DATE': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
                }
            }
        }
    })

    synthesizer = GaussianCopulaSynthesizer(metadata)
    pattern = Inequality(
        low_column_name='DUE_DATE',
        high_column_name='SUBMISSION_TIMESTAMP',
        strict_boundaries=True,
    )
    synthesizer.add_cag(patterns=[pattern])

    # Run and Assert
    error_msg = 'The inequality requirement is not met for row indices: '
    with pytest.raises(PatternNotMetError) as error:
        synthesizer.fit(data)
        assert error_msg in error


def test_inequality_pattern_timestamp_less_than_date_strict_boundaries():
    """Test that the inequality pattern fails when timestamp is less than date.

    This case evaluates the `Inequality` pattern with
    `strict_boundaries=True` when the `SUBMISSION_TIMESTAMP` column contains a
    timestamp, and the `DUE_DATE` contains a date. This case should violate the
    pattern, as the `SUBMISSION_TIMESTAMP` (which includes time) is not
    strictly less than the `DUE_DATE` (which has no time).
    """
    # Setup
    data = pd.DataFrame(
        data={
            'SUBMISSION_TIMESTAMP': [
                '2024-11-12 02:34:45',
                '2024-11-13 08:23:15',
                '2024-11-14 11:10:10',
                '2024-11-15 13:55:30',
                '2024-11-16 09:05:00',
            ],
            'DUE_DATE': ['2024-11-12', '2024-11-13', '2024-11-14', '2024-11-15', '2024-11-16'],
        }
    )

    metadata = Metadata.load_from_dict({
        'tables': {
            'table': {
                'columns': {
                    'SUBMISSION_TIMESTAMP': {
                        'sdtype': 'datetime',
                        'datetime_format': '%Y-%m-%d %H:%M:%S',
                    },
                    'DUE_DATE': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
                }
            }
        }
    })

    synthesizer = GaussianCopulaSynthesizer(metadata)
    pattern = Inequality(
        low_column_name='SUBMISSION_TIMESTAMP',
        high_column_name='DUE_DATE',
        strict_boundaries=True,
    )
    synthesizer.add_cag(patterns=[pattern])

    # Run and Assert
    err_msg = 'The inequality requirement is not met for row indices: '
    with pytest.raises(PatternNotMetError) as error:
        synthesizer.fit(data)
        assert err_msg in error


def test_inequality_pattern_date_less_than_timestamp_no_strict_boundaries():
    """Test that the inequality pattern passes when date is less than timestamp.

    This case evaluates the `Inequality` pattern with
    `strict_boundaries=False`. Since `strict_boundaries` is set to `False`, we
    assume that a date in the `DUE_DATE` column should be treated as the
    beginning of the day,
    so there will be no violation if the `SUBMISSION_TIMESTAMP` is later on the
    same day.
    """
    # Setup
    data = pd.DataFrame(
        data={
            'SUBMISSION_TIMESTAMP': [
                '2024-11-12 02:34:45',
                '2024-11-13 10:45:30',
                '2024-11-14 14:30:00',
                '2024-11-15 16:20:10',
                '2024-11-16 08:00:00',
            ],
            'DUE_DATE': ['2024-11-12', '2024-11-13', '2024-11-14', '2024-11-15', '2024-11-16'],
        }
    )

    metadata = Metadata.load_from_dict({
        'tables': {
            'table': {
                'columns': {
                    'SUBMISSION_TIMESTAMP': {
                        'sdtype': 'datetime',
                        'datetime_format': '%Y-%m-%d %H:%M:%S',
                    },
                    'DUE_DATE': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
                }
            }
        }
    })

    synthesizer = GaussianCopulaSynthesizer(metadata)
    pattern = Inequality(
        low_column_name='DUE_DATE',
        high_column_name='SUBMISSION_TIMESTAMP',
        strict_boundaries=False,
    )
    synthesizer.add_cag(patterns=[pattern])

    # Run
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample(10)

    # Assert
    synthetic_data['SUBMISSION_TIMESTAMP'] = pd.to_datetime(
        synthetic_data['SUBMISSION_TIMESTAMP'], errors='coerce'
    )
    synthetic_data['DUE_DATE'] = pd.to_datetime(synthetic_data['DUE_DATE'], errors='coerce')
    invalid_rows = synthetic_data[
        synthetic_data['SUBMISSION_TIMESTAMP'].dt.date > synthetic_data['DUE_DATE'].dt.date
    ]
    assert invalid_rows.empty


def test_inequality_pattern_timestamp_less_than_date_no_strict_boundaries():
    """Test that the inequality pattern passes when timestamp is less than date.

    This case evaluates the `Inequality` pattern with
    `strict_boundaries=False`. Since `strict_boundaries` is set to `False`, we
    assume that a date in the `DUE_DATE` column should be treated as the end of
    the day, so there will be no violation if the `SUBMISSION_TIMESTAMP` is
    earlier on the same day.
    """
    # Setup
    data = pd.DataFrame(
        data={
            'SUBMISSION_TIMESTAMP': [
                '2024-11-12 02:34:45',
                '2024-11-13 08:23:15',
                '2024-11-14 11:10:10',
                '2024-11-15 13:55:30',
                '2024-11-16 09:05:00',
            ],
            'DUE_DATE': ['2024-11-12', '2024-11-13', '2024-11-14', '2024-11-15', '2024-11-16'],
        }
    )

    metadata = Metadata.load_from_dict({
        'tables': {
            'table': {
                'columns': {
                    'SUBMISSION_TIMESTAMP': {
                        'sdtype': 'datetime',
                        'datetime_format': '%Y-%m-%d %H:%M:%S',
                    },
                    'DUE_DATE': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
                }
            }
        }
    })

    synthesizer = GaussianCopulaSynthesizer(metadata)
    pattern = Inequality(
        low_column_name='SUBMISSION_TIMESTAMP',
        high_column_name='DUE_DATE',
        strict_boundaries=False,
    )
    synthesizer.add_cag(patterns=[pattern])

    # Run
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample(10)

    # Assert
    synthetic_data['SUBMISSION_TIMESTAMP'] = pd.to_datetime(
        synthetic_data['SUBMISSION_TIMESTAMP'], errors='coerce'
    )
    synthetic_data['DUE_DATE'] = pd.to_datetime(synthetic_data['DUE_DATE'], errors='coerce')
    invalid_rows = synthetic_data[
        synthetic_data['SUBMISSION_TIMESTAMP'].dt.date > synthetic_data['DUE_DATE'].dt.date
    ]
    assert invalid_rows.empty
