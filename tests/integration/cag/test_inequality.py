import re

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_object_dtype

from sdv.cag import Inequality
from sdv.cag._errors import PatternNotMetError
from sdv.datasets.demo import download_demo
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer
from tests.utils import run_copula, run_hma, run_pattern


@pytest.fixture()
def data():
    return pd.DataFrame({
        'A': [1, 2, 3, 1, 2, 1],
        'B': [10, 20, 30, 10, 20, 10],
    })


@pytest.fixture()
def metadata():
    return Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'numerical'},
            'B': {'sdtype': 'numerical'},
        }
    })


@pytest.fixture()
def pattern():
    return Inequality(
        low_column_name='A',
        high_column_name='B',
    )


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
    return Inequality(
        low_column_name='A',
        high_column_name='B',
        table_name='table1',
    )


@pytest.fixture()
def data_reject():
    return pd.DataFrame({
        'low': [1, 2, 3, 1, 2, 1],
        'low2': [1, 2, 3, 1, 2, 1],
        'high': [10, 20, 30, 10, 20, 10],
    })


@pytest.fixture()
def metadata_reject():
    return Metadata.load_from_dict({
        'columns': {
            'low': {'sdtype': 'numerical'},
            'low2': {'sdtype': 'numerical'},
            'high': {'sdtype': 'numerical'},
        }
    })


@pytest.fixture()
def patterns_reject():
    # pattern1 drops high column, which pattern2 relies upon
    pattern1 = Inequality(
        low_column_name='low',
        high_column_name='high',
    )
    pattern2 = Inequality(
        low_column_name='low2',
        high_column_name='high',
    )
    return [pattern1, pattern2]


def test_inequality_pattern_integers(data, metadata, pattern):
    """Test that Inequality pattern works with integer columns."""
    # Run
    updated_metadata, transformed, reverse_transformed = run_pattern(pattern, data, metadata)

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'columns': {
            'A.fillna': {'sdtype': 'numerical'},
            'A#B': {'sdtype': 'numerical'},
            'A#B.nan_component': {'sdtype': 'categorical'},
        }
    }).to_dict()
    assert expected_updated_metadata == updated_metadata.to_dict()
    assert list(transformed.columns) == ['A.fillna', 'A#B', 'A#B.nan_component']
    pd.testing.assert_frame_equal(data, reverse_transformed)


def test_all_possible_nans_configurations(pattern, metadata):
    """Test it works with all possible NaN configurations."""
    # Setup
    data = pd.DataFrame(data={'A': [0, 1, np.nan, np.nan, 2], 'B': [2, np.nan, 3, np.nan, 3]})

    # Run
    synthesizer = run_copula(data, metadata, [pattern])
    synthetic_data = synthesizer.sample(10000)

    # Assert
    assert (~(pd.isna(synthetic_data['A'])) & ~(pd.isna(synthetic_data['B']))).any()
    assert ((pd.isna(synthetic_data['A'])) & ~(pd.isna(synthetic_data['B']))).any()
    assert (~(pd.isna(synthetic_data['A'])) & (pd.isna(synthetic_data['B']))).any()
    assert (~(pd.isna(synthetic_data['A'])) & ~(pd.isna(synthetic_data['B']))).any()


def test_inequality_pattern_with_nans(metadata, pattern):
    """Test that Inequality pattern works with NaNs."""
    # Setup
    data = pd.DataFrame({
        'A': [None, 2, np.nan, 1, 2, 1],
        'B': [np.nan, 20, 30, 10, 20, None],
    })

    # Run
    updated_metadata, transformed, reverse_transformed = run_pattern(pattern, data, metadata)

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'columns': {
            'A.fillna': {'sdtype': 'numerical'},
            'A#B': {'sdtype': 'numerical'},
            'A#B.nan_component': {'sdtype': 'categorical'},
        }
    }).to_dict()
    assert expected_updated_metadata == updated_metadata.to_dict()
    assert list(transformed.columns) == ['A.fillna', 'A#B', 'A#B.nan_component']
    pd.testing.assert_frame_equal(
        pd.concat([data.iloc[:2], data.iloc[3:]]),
        pd.concat([reverse_transformed.iloc[:2], reverse_transformed.iloc[3:]]),
    )
    assert 1 < reverse_transformed.iloc[2]['B'] < 30
    assert pd.isna(reverse_transformed.iloc[2]['A'])


def test_inequality_pattern_datetime(pattern):
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

    # Run
    updated_metadata, transformed, reverse_transformed = run_pattern(pattern, data, metadata)

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'columns': {
            'A.fillna': {'sdtype': 'datetime'},
            'A#B': {'sdtype': 'numerical'},
            'A#B.nan_component': {'sdtype': 'categorical'},
        }
    }).to_dict()
    assert expected_updated_metadata == updated_metadata.to_dict()
    assert list(transformed.columns) == ['A.fillna', 'A#B', 'A#B.nan_component']

    # Check that the timestamps are very close to each other
    for col in ['A', 'B']:
        diff = (data[col] - reverse_transformed[col]).abs()
        assert (diff.dt.total_seconds() < 1e-6).all()


def test_inequality_pattern_datetime_nans(pattern):
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

    # Run
    updated_metadata, transformed, reverse_transformed = run_pattern(pattern, data, metadata)

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'columns': {
            'A.fillna': {'sdtype': 'datetime'},
            'A#B': {'sdtype': 'numerical'},
            'A#B.nan_component': {'sdtype': 'categorical'},
        }
    }).to_dict()
    assert expected_updated_metadata == updated_metadata.to_dict()
    assert list(transformed.columns) == ['A.fillna', 'A#B', 'A#B.nan_component']

    pd.testing.assert_series_equal(data['A'], reverse_transformed['A'])
    assert pd.Timestamp('2024-01-01') < reverse_transformed['B'][0] < pd.Timestamp('2024-01-07')
    assert pd.isna(reverse_transformed['B'][1])
    assert pd.isna(reverse_transformed['B'][2])

    diff = (data['B'][3:] - reverse_transformed['B'][3:]).abs()
    assert (diff.dt.total_seconds() < 1e-6).all()


def test_inequality_pattern_with_multi_table(data_multi, metadata_multi, pattern_multi):
    """Test that Inequality pattern works with multi-table data."""
    # Setup
    data = data_multi
    metadata = metadata_multi
    pattern = pattern_multi

    # Run
    updated_metadata, transformed, reverse_transformed = run_pattern(pattern, data, metadata)

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'tables': {
            'table1': {
                'columns': {
                    'A.fillna': {'sdtype': 'numerical'},
                    'A#B': {'sdtype': 'numerical'},
                    'A#B.nan_component': {'sdtype': 'categorical'},
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
    assert list(transformed['table1'].columns) == ['A.fillna', 'A#B', 'A#B.nan_component']
    assert set(data.keys()) == set(reverse_transformed.keys())
    for table_name, table in data.items():
        pd.testing.assert_frame_equal(table, reverse_transformed[table_name])


def test_inequality_with_numerical(data, metadata, pattern):
    """Test it works with numerical columns."""
    # Run
    synthesizer = run_copula(data, metadata, [pattern])
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
    pattern = Inequality(
        low_column_name='SUBMISSION_TIMESTAMP',
        high_column_name='DUE_DATE',
        strict_boundaries=False,
    )

    # Run
    synthesizer = run_copula(data, metadata, [pattern])
    synthetic_data = synthesizer.sample(num_rows=10)

    # Assert
    assert is_object_dtype(synthetic_data['SUBMISSION_TIMESTAMP'].dtype)
    synthetic_data['SUBMISSION_TIMESTAMP'] = pd.to_datetime(
        synthetic_data['SUBMISSION_TIMESTAMP'], format='%Y-%m-%d %H:%M:%S'
    )
    assert is_object_dtype(synthetic_data['DUE_DATE'].dtype)
    synthetic_data['DUE_DATE'] = pd.to_datetime(synthetic_data['DUE_DATE'], format='%Y-%m-%d')
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

    pattern = Inequality(
        low_column_name='SUBMISSION_TIMESTAMP',
        high_column_name='DUE_DATE',
        strict_boundaries=True,
    )

    # Run and Assert
    error_msg = 'The inequality requirement is not met for row indices: '
    with pytest.raises(PatternNotMetError) as error:
        run_copula(data, metadata, [pattern])
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

    pattern = Inequality(
        low_column_name='DUE_DATE',
        high_column_name='SUBMISSION_TIMESTAMP',
        strict_boundaries=True,
    )

    # Run and Assert
    error_msg = 'The inequality requirement is not met for row indices: '
    with pytest.raises(PatternNotMetError) as error:
        run_copula(data, metadata, [pattern])
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

    pattern = Inequality(
        low_column_name='SUBMISSION_TIMESTAMP',
        high_column_name='DUE_DATE',
        strict_boundaries=True,
    )

    # Run and Assert
    err_msg = 'The inequality requirement is not met for row indices: '
    with pytest.raises(PatternNotMetError) as error:
        run_copula(data, metadata, [pattern])
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

    pattern = Inequality(
        low_column_name='DUE_DATE',
        high_column_name='SUBMISSION_TIMESTAMP',
        strict_boundaries=False,
    )

    # Run
    synthesizer = run_copula(data, metadata, [pattern])
    synthetic_data = synthesizer.sample(10)

    # Assert
    assert is_object_dtype(synthetic_data['SUBMISSION_TIMESTAMP'].dtype)
    synthetic_data['SUBMISSION_TIMESTAMP'] = pd.to_datetime(
        synthetic_data['SUBMISSION_TIMESTAMP'], format='%Y-%m-%d %H:%M:%S'
    )
    assert is_object_dtype(synthetic_data['DUE_DATE'].dtype)
    synthetic_data['DUE_DATE'] = pd.to_datetime(synthetic_data['DUE_DATE'], format='%Y-%m-%d')
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

    pattern = Inequality(
        low_column_name='SUBMISSION_TIMESTAMP',
        high_column_name='DUE_DATE',
        strict_boundaries=False,
    )

    # Run
    synthesizer = run_copula(data, metadata, [pattern])
    synthetic_data = synthesizer.sample(10)

    # Assert
    assert is_object_dtype(synthetic_data['SUBMISSION_TIMESTAMP'].dtype)
    synthetic_data['SUBMISSION_TIMESTAMP'] = pd.to_datetime(
        synthetic_data['SUBMISSION_TIMESTAMP'], format='%Y-%m-%d %H:%M:%S'
    )
    assert is_object_dtype(synthetic_data['DUE_DATE'].dtype)
    synthetic_data['DUE_DATE'] = pd.to_datetime(synthetic_data['DUE_DATE'], format='%Y-%m-%d')
    invalid_rows = synthetic_data[
        synthetic_data['SUBMISSION_TIMESTAMP'].dt.date > synthetic_data['DUE_DATE'].dt.date
    ]
    assert invalid_rows.empty


@pytest.mark.skip(reason='This test is failing because of the time component in the B column.')
def test_inequality_unequal_datetime_formats_strings():
    """Test that the inequality pattern works with unequal datetime formats."""
    # Setup
    data = pd.DataFrame({
        'A': ['2020-01-01', '2020-01-02', '2020-01-03'],
        'B': ['2020-01-02 10:00:00', '2020-01-03 13:00:00', '2020-01-04 6:00:00'],
    })
    metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
            'B': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d %H:%M:%S'},
        }
    })
    pattern = Inequality(low_column_name='A', high_column_name='B')

    # Run
    sample = run_copula(data, metadata, [pattern]).sample(10)

    # Assert
    assert is_object_dtype(sample['A'].dtype)
    assert is_object_dtype(sample['B'].dtype)

    col_A = pd.to_datetime(sample['A'], format='%Y-%m-%d')
    col_B = pd.to_datetime(sample['B'], format='%Y-%m-%d %H:%M:%S')
    assert all(col_A <= col_B)
    assert any(col_B.dt.time.astype(str) != '00:00:00')


@pytest.mark.skip(reason='This test is failing because of the time component in the B column.')
def test_inequality_unequal_datetime_formats():
    """Test that the inequality pattern works with unequal datetime formats."""
    # Setup
    data = pd.DataFrame({
        'A': pd.to_datetime(['2020-01-01', np.nan, '2020-01-02']),
        'B': pd.to_datetime(['2020-01-02 10:00:00', '2020-01-03 13:00:00', np.nan]),
    })
    metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
            'B': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d %H:%M:%S'},
        }
    })
    pattern = Inequality(low_column_name='A', high_column_name='B')

    # Run
    sample = run_copula(data, metadata, [pattern]).sample(10)

    # Assert
    col_A = pd.to_datetime(sample['A'], format='%Y-%m-%d')
    col_B = pd.to_datetime(sample['B'], format='%Y-%m-%d %H:%M:%S')
    assert all(col_A <= col_B)
    assert any(col_B.dt.time.astype(str) != '00:00:00')


@pytest.mark.skip(reason='Timezone not supported for object dtype.')
def test_inequality_unequal_datetime_formats_timezone_aware():
    """Test that the inequality pattern works with timezone-aware datetime objects."""
    # Setup
    data = pd.DataFrame({
        'A': ['2020-01-01 UTC', '2020-01-02 UTC', '2020-01-03 UTC'],
        'B': ['2020-01-02 10:00:00 UTC', '2020-01-03 13:00:00 UTC', '2020-01-04 6:00:00 UTC'],
    })
    metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d %Z'},
            'B': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d %H:%M:%S %Z'},
        }
    })
    pattern = Inequality(low_column_name='A', high_column_name='B')

    # Run
    sample = run_copula(data, metadata, [pattern]).sample(10)

    # Assert
    assert is_object_dtype(sample['A'].dtype)
    assert is_object_dtype(sample['B'].dtype)

    col_A = pd.to_datetime(sample['A'], format='%Y-%m-%d')
    col_B = pd.to_datetime(sample['B'], format='%Y-%m-%d %H:%M:%S')
    assert all(col_A <= col_B)
    assert any(col_B.dt.time.astype(str) != '00:00:00')


@pytest.mark.skip(reason='Does not work if only one column has timezone.')
def test_inequality_unequal_datetime_formats_unequal_timezone():
    """Test that the inequality pattern works with timezone-aware datetime objects."""
    # Setup
    data = pd.DataFrame({
        'A': pd.to_datetime(['2020-01-01', np.nan, '2020-01-02']),
        'B': pd.to_datetime(['2020-01-02 10:00:00 UTC', '2020-01-03 13:00:00 UTC', np.nan]),
    })
    metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
            'B': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d %H:%M:%S %Z'},
        }
    })
    pattern = Inequality(low_column_name='A', high_column_name='B')

    # Run
    sample = run_copula(data, metadata, [pattern]).sample(10)

    # Assert
    assert is_object_dtype(sample['A'].dtype)
    assert is_object_dtype(sample['B'].dtype)

    col_A = pd.to_datetime(sample['A'], format='%Y-%m-%d')
    col_B = pd.to_datetime(sample['B'], format='%Y-%m-%d %H:%M:%S')
    assert all(col_A <= col_B)
    assert any(col_B.dt.time.astype(str) != '00:00:00')


def test_inequality_multiple_patterns():
    """Test that Inequality pattern works with multiple patterns."""
    # Setup
    data = pd.DataFrame({
        'low': [1, 2, 3, 1, 2, 1],
        'high1': [10, 20, 30, 10, 20, 10],
        'high2': [10, 20, 30, 10, 20, 10],
    })
    metadata = Metadata.load_from_dict({
        'columns': {
            'low': {'sdtype': 'numerical'},
            'high1': {'sdtype': 'numerical'},
            'high2': {'sdtype': 'numerical'},
        }
    })
    pattern1 = Inequality(
        low_column_name='low',
        high_column_name='high1',
    )
    pattern2 = Inequality(
        low_column_name='low.fillna',
        high_column_name='high2',
    )

    # Run
    synthesizer = run_copula(data, metadata, [pattern1, pattern2])
    samples = synthesizer.sample(100)
    updated_metadata = synthesizer.get_metadata('modified')
    original_metadata = synthesizer.get_metadata('original')

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'columns': {
            'low.fillna.fillna': {'sdtype': 'numerical'},
            'low#high1': {'sdtype': 'numerical'},
            'low.fillna#high2': {'sdtype': 'numerical'},
            'low#high1.nan_component': {'sdtype': 'categorical'},
            'low.fillna#high2.nan_component': {'sdtype': 'categorical'},
        }
    }).to_dict()
    assert expected_updated_metadata == updated_metadata.to_dict()

    assert original_metadata.to_dict() == metadata.to_dict()

    assert all(samples['low'] <= samples['high1'])
    assert all(samples['low'] <= samples['high2'])


def test_inequality_multiple_patterns_reject_sampling(
    data_reject, metadata_reject, patterns_reject
):
    """Test that Inequality pattern works with multiple patterns using reject sampling."""
    # Run
    synthesizer = run_copula(data_reject, metadata_reject, patterns_reject)
    samples = synthesizer.sample(10)
    updated_metadata = synthesizer.get_metadata('modified')
    original_metadata = synthesizer.get_metadata('original')

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'columns': {
            'low.fillna': {'sdtype': 'numerical'},
            'low#high': {'sdtype': 'numerical'},
            'low2': {'sdtype': 'numerical'},
            'low#high.nan_component': {'sdtype': 'categorical'},
        }
    }).to_dict()
    assert expected_updated_metadata == updated_metadata.to_dict()

    assert original_metadata.to_dict() == metadata_reject.to_dict()

    assert all(samples['low'] <= samples['high'])


def test_inequality_multiple_patterns_one_pattern_invalid_column():
    """Test that Inequality pattern works with multiple patterns."""
    # Setup
    values = np.random.randint(0, 100, size=1000)
    data = pd.DataFrame({
        'low': values,
        'mid': values + 1,
        'high': values + 2,
    })
    metadata = Metadata.load_from_dict({
        'columns': {
            'low': {'sdtype': 'numerical'},
            'mid': {'sdtype': 'numerical'},
            'high': {'sdtype': 'numerical'},
        }
    })
    pattern1 = Inequality(
        low_column_name='low',
        high_column_name='mid',
    )
    pattern2 = Inequality(
        low_column_name='mid',
        high_column_name='high',
    )

    # Run
    synthesizer = run_copula(data, metadata, [pattern1, pattern2])
    samples = synthesizer.sample(1000000)
    updated_metadata = synthesizer.get_metadata('modified')
    original_metadata = synthesizer.get_metadata('original')

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'columns': {
            'low.fillna': {'sdtype': 'numerical'},
            'low#mid': {'sdtype': 'numerical'},
            'low#mid.nan_component': {'sdtype': 'categorical'},
            'high': {'sdtype': 'numerical'},
        }
    }).to_dict()

    assert expected_updated_metadata == updated_metadata.to_dict()

    assert original_metadata.to_dict() == metadata.to_dict()

    assert all(samples['low'] <= samples['mid'])
    assert all(samples['mid'] <= samples['high'])


def test_inequality_many_patterns():
    """Test that Inequality pattern works with multiple patterns."""
    # Setup
    values = np.random.randint(0, 100, size=1000)
    data = pd.DataFrame({f'{i}': values + i for i in range(10)})
    metadata = Metadata.load_from_dict({
        'columns': {f'{i}': {'sdtype': 'numerical'} for i in range(10)}
    })
    patterns = [Inequality(low_column_name=f'{i}', high_column_name=f'{i + 1}') for i in range(9)]

    # Run
    synthesizer = run_copula(data, metadata, patterns)
    samples = synthesizer.sample(100)

    updated_metadata = synthesizer.get_metadata('modified')
    original_metadata = synthesizer.get_metadata('original')

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'columns': {
            '0.fillna': {'sdtype': 'numerical'},
            '0#1': {'sdtype': 'numerical'},
            '0#1.nan_component': {'sdtype': 'categorical'},
            '2.fillna': {'sdtype': 'numerical'},
            '2#3': {'sdtype': 'numerical'},
            '2#3.nan_component': {'sdtype': 'categorical'},
            '4.fillna': {'sdtype': 'numerical'},
            '4#5': {'sdtype': 'numerical'},
            '4#5.nan_component': {'sdtype': 'categorical'},
            '6.fillna': {'sdtype': 'numerical'},
            '6#7': {'sdtype': 'numerical'},
            '6#7.nan_component': {'sdtype': 'categorical'},
            '8.fillna': {'sdtype': 'numerical'},
            '8#9': {'sdtype': 'numerical'},
            '8#9.nan_component': {'sdtype': 'categorical'},
        }
    }).to_dict()

    assert expected_updated_metadata == updated_metadata.to_dict()
    assert original_metadata.to_dict() == metadata.to_dict()
    for i in range(9):
        assert all(samples[f'{i}'] <= samples[f'{i + 1}'])


def test_inequality_with_nan():
    """Test that Inequality pattern works with NaN values."""
    # Setup
    data, metadata = download_demo('single_table', 'fake_hotel_guests')
    inequality_cag = Inequality(
        low_column_name='checkin_date',
        high_column_name='checkout_date',
    )

    # Run
    synthesizer = run_copula(data, metadata, [inequality_cag])
    sampled_data = synthesizer.sample(100)
    synthesizer.validate(sampled_data)

    # Assert
    assert data['checkout_date'].isna().sum() > 0
    assert sampled_data['checkout_date'].isna().sum() > 0
    valid_dates = sampled_data[['checkin_date', 'checkout_date']].dropna()
    assert all(
        pd.to_datetime(valid_dates['checkin_date']) <= pd.to_datetime(valid_dates['checkout_date'])
    )


def test_validate_cag(data, metadata, pattern):
    """Test validate_cag works with synthetic data generated with Inequality."""
    # Setup
    synthesizer = run_copula(data, metadata, [pattern])
    synthetic_data = synthesizer.sample(100)

    # Run
    synthesizer.validate_cag(synthetic_data=synthetic_data)

    # Assert
    assert all(synthetic_data['A'] < synthetic_data['B'])


def test_validate_cag_raises(data, metadata, pattern):
    """Test validate_cag raises an error with bad synthetic data with Inequality."""
    # Setup
    synthetic_data = pd.DataFrame({
        'A': [10, 20, 30, 10, 20, 10],
        'B': [1, 2, 3, 1, 2, 1],
    })
    assert all(data['A'] < data['B'])
    assert all(synthetic_data['A'] > synthetic_data['B'])
    synthesizer = run_copula(data, metadata, [pattern])
    msg = re.escape('The inequality requirement is not met for row indices: 0, 1, 2, 3, 4, +1 more')

    # Run and Assert
    with pytest.raises(PatternNotMetError, match=msg):
        synthesizer.validate_cag(synthetic_data=synthetic_data)


def test_validate_cag_multi(data_multi, metadata_multi, pattern_multi):
    """Test validate_cag works with multitable synthetic data generated with Inequality."""
    # Setup
    synthesizer = run_hma(data_multi, metadata_multi, [pattern_multi])
    synthetic_data = synthesizer.sample(100)

    # Run
    synthesizer.validate_cag(synthetic_data=synthetic_data)

    # Assert
    assert all(synthetic_data['table1']['A'] < synthetic_data['table1']['B'])


def test_validate_cag_multi_with_reject(data_reject, metadata_reject, patterns_reject):
    """Test validate_cag works with reject sampling."""
    # Setup
    synthesizer = run_copula(data_reject, metadata_reject, patterns_reject)
    synthetic_data = synthesizer.sample(100)

    # Run
    synthesizer.validate_cag(synthetic_data=synthetic_data)

    # Assert
    assert all(synthetic_data['low'] <= synthetic_data['high'])


def test_validate_cag_multi_with_reject_raises(data_reject, metadata_reject, patterns_reject):
    """Test validate_cag raises an error due reject sampling pattern not matching."""
    # Setup
    synthesizer = run_copula(data_reject, metadata_reject, patterns_reject)

    # pattern 1 matches, but pattern 2 does not
    synthetic_data = pd.DataFrame({
        'low': [1, 2, 3, 1, 2, 1],
        'low2': [10, 20, 30, 10, 20, 10],
        'high': [5, 4, 6, 7, 8, 9],
    })
    msg = re.escape(
        'The inequality requirement is not met for row indices: 0, 1, 2, 3, 4, +1 more.'
    )

    # Run and Assert
    with pytest.raises(PatternNotMetError, match=msg):
        synthesizer.validate_cag(synthetic_data=synthetic_data)


def test_validate_cag_multi_raises(data_multi, metadata_multi, pattern_multi):
    """Test validate_cag raises an error with multitable data."""
    # Setup
    synthetic_data = {
        'table1': pd.DataFrame({
            'A': data_multi['table1']['B'],
            'B': data_multi['table1']['A'],
        }),
        'table2': pd.DataFrame({'id': range(5)}),
    }
    synthesizer = run_hma(data_multi, metadata_multi, [pattern_multi])
    msg = re.escape(
        "Table 'table1': The inequality requirement is not met for "
        'row indices: 0, 1, 2, 3, 4, +1 more'
    )

    # Run and Assert
    with pytest.raises(PatternNotMetError, match=msg):
        synthesizer.validate_cag(synthetic_data=synthetic_data)


def test_invalid_data():
    """Test that the Inequality pattern raises an error with invalid data."""
    # Setup
    data, metadata = download_demo('single_table', 'fake_hotel_guests')
    pattern = Inequality(
        low_column_name='checkin_date',
        high_column_name='checkout_date',
        strict_boundaries=False,
    )
    clean_data = data[~(data[['checkin_date', 'checkout_date']].isna().any(axis=1))]
    data_invalid = clean_data.copy()
    data_invalid.loc[0, 'checkin_date'] = '31 Dec 2020'
    expected_error_msg = re.escape('The inequality requirement is not met for row indices: [0]')

    # Run and Assert
    synthesizer = run_copula(clean_data, metadata, [pattern])
    with pytest.raises(PatternNotMetError, match=expected_error_msg):
        synthesizer.fit(data_invalid)


def test_auto_assign_transformer():
    """Test that the Inequality pattern works with auto-assign transformer."""
    # Setup
    data, metadata = download_demo('single_table', 'fake_hotel_guests')
    inequality_cag = Inequality(
        low_column_name='checkin_date',
        high_column_name='checkout_date',
    )
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.add_cag([inequality_cag])

    # Run
    synthesizer.auto_assign_transformers(data)

    # Assert
    expected_transformers = (
        "{'guest_email': AnonymizedFaker(provider_name='internet', function_name='email',"
        " locales=['en_US'], cardinality_rule='unique'),"
        " 'has_rewards': UniformEncoder(),"
        " 'room_type': UniformEncoder(),"
        " 'amenities_fee': FloatFormatter(learn_rounding_scheme=True, "
        'enforce_min_max_values=True),'
        " 'room_rate': FloatFormatter(learn_rounding_scheme=True,"
        ' enforce_min_max_values=True),'
        " 'billing_address': AnonymizedFaker(provider_name='address',"
        " function_name='address', locales=['en_US']),"
        " 'credit_card_number': AnonymizedFaker(provider_name='credit_card',"
        " function_name='credit_card_number', locales=['en_US']),"
        " 'checkin_date.fillna': UnixTimestampEncoder(datetime_format='%d %b %Y',"
        ' enforce_min_max_values=True),'
        " 'checkin_date#checkout_date': FloatFormatter(learn_rounding_scheme=True,"
        ' enforce_min_max_values=True),'
        " 'checkin_date#checkout_date.nan_component': UniformEncoder()}"
    )
    assert str(synthesizer.get_transformers()) == expected_transformers


def test_low_column_formatting_maintained():
    """Test that the Inequality pattern works with auto-assign transformer."""
    # Setup
    data, metadata = download_demo('single_table', 'fake_hotel_guests')
    inequality_cag = Inequality(
        low_column_name='amenities_fee',
        high_column_name='room_rate',
    )
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.add_cag([inequality_cag])

    # Run
    synthesizer.fit(data)
    sampled_data = synthesizer.sample(100)

    # Assert
    assert all(sampled_data['room_rate'].round(2) == sampled_data['room_rate'])
