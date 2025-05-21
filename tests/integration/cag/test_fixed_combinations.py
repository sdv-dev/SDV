"""Integration tests for FixedCombinations CAG pattern."""

import re

import numpy as np
import pandas as pd
import pytest

from sdv.cag import FixedCombinations
from sdv.cag._errors import PatternNotMetError
from sdv.metadata import Metadata
from sdv.single_table.copulas import GaussianCopulaSynthesizer
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
            'A': {'sdtype': 'categorical'},
            'B': {'sdtype': 'categorical'},
        }
    })


@pytest.fixture()
def pattern():
    return FixedCombinations(['A', 'B'])


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
                    'A': {'sdtype': 'categorical'},
                    'B': {'sdtype': 'categorical'},
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
    return FixedCombinations(['A', 'B'], table_name='table1')


def test_fixed_combinations_integers(data, metadata, pattern):
    """Test that FixedCombinations constraint works with integer columns."""
    # Run
    updated_metadata, transformed, reverse_transformed = run_pattern(pattern, data, metadata)

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'columns': {
            'A#B': {'sdtype': 'categorical'},
        }
    }).to_dict()
    assert expected_updated_metadata == updated_metadata.to_dict()
    assert list(transformed.columns) == ['A#B']
    pd.testing.assert_frame_equal(data, reverse_transformed)


def test_fixed_combinations_integers_copula(data, metadata, pattern):
    """Test that FixedCombinations constraint works with integer columns using Copula."""
    # Run
    synthesizer = run_copula(data, metadata, [pattern])
    synthetic_data = synthesizer.sample(1000)

    # Assert
    assert len(synthetic_data) == 1000
    pd.testing.assert_frame_equal(
        synthetic_data.drop_duplicates(ignore_index=True),
        data.drop_duplicates(ignore_index=True),
        check_like=True,
    )


def test_fixed_combinations_with_nans(metadata, pattern):
    """Test that FixedCombinations constraint works with NaNs."""
    # Setup
    data = pd.DataFrame({
        'A': pd.Categorical([1, 2, np.nan, 1, 2, 1]),
        'B': pd.Categorical([10, 20, 30, 10, 20, None]),
    })

    # Run
    updated_metadata, transformed, reverse_transformed = run_pattern(pattern, data, metadata)

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'columns': {
            'A#B': {'sdtype': 'categorical'},
        }
    }).to_dict()
    assert expected_updated_metadata == updated_metadata.to_dict()
    assert list(transformed.columns) == ['A#B']
    pd.testing.assert_frame_equal(data, reverse_transformed)


def test_fixed_combinations_with_nans_copula(metadata, pattern):
    """Test that FixedCombinations constraint works with nans using Copula."""
    # Setup
    data = pd.DataFrame({
        'A': [1, 2, np.nan, 1, 2, 1],
        'B': [10, 20, 30, 10, 20, 10],
    })

    # Run
    synthesizer = run_copula(data, metadata, [pattern])
    synthetic_data = synthesizer.sample(1000)

    # Assert
    assert len(synthetic_data) == 1000
    pd.testing.assert_frame_equal(
        synthetic_data.drop_duplicates(ignore_index=True),
        pd.DataFrame({
            'A': [1, np.nan, 2],
            'B': [10, 30, 20],
        }).drop_duplicates(ignore_index=True),
        check_like=True,
    )


def test_fixed_null_combinations_with_multi_table(data_multi, metadata_multi, pattern_multi):
    """Test that FixedCombinations constraint works with multi-table data."""
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
                    'A#B': {'sdtype': 'categorical'},
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
    assert list(transformed['table1'].columns) == ['A#B']
    assert set(data.keys()) == set(reverse_transformed.keys())
    for table_name, table in data.items():
        pd.testing.assert_frame_equal(table, reverse_transformed[table_name])


def test_fixed_combinations_multiple_patterns():
    """Test that FixedCombinations pattern works with multiple patterns."""
    # Setup
    data = pd.DataFrame({
        'A': [1, 2, 3, 1, 2, 1],
        'B': [10, 20, 30, 10, 20, 10],
        'C': [100, 200, 300, 100, 200, 100],
        'D': [1000, 2000, 3000, 1000, 2000, 1000],
    })
    metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'categorical'},
            'B': {'sdtype': 'categorical'},
            'C': {'sdtype': 'categorical'},
            'D': {'sdtype': 'categorical'},
        }
    })
    pattern1 = FixedCombinations(['A', 'B'])
    pattern2 = FixedCombinations(['C', 'D'])

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
            'A#B': {'sdtype': 'categorical'},
            'C#D': {'sdtype': 'categorical'},
        }
    }).to_dict()
    assert expected_updated_metadata == updated_metadata.to_dict()

    assert original_metadata.to_dict() == metadata.to_dict()

    # Get unique combinations from original data
    original_ab_combos = set(zip(data['A'], data['B']))
    original_cd_combos = set(zip(data['C'], data['D']))

    # Get unique combinations from synthetic data
    synthetic_ab_combos = set(zip(samples['A'], samples['B']))
    synthetic_cd_combos = set(zip(samples['C'], samples['D']))

    # Assert combinations match
    assert original_ab_combos == synthetic_ab_combos
    assert original_cd_combos == synthetic_cd_combos


def test_fixed_combinations_multiple_patterns_reject_sampling():
    """Test that FixedCombinations pattern works with multiple patterns and reject sampling."""
    # Setup
    data = pd.DataFrame({
        'A': [1, 2, 3, 1, 2, 1],
        'B': [10, 20, 30, 10, 20, 10],
        'C': [100, 200, 300, 100, 200, 100],
    })
    metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'categorical'},
            'B': {'sdtype': 'categorical'},
            'C': {'sdtype': 'categorical'},
        }
    })
    pattern1 = FixedCombinations(['A', 'B'])
    pattern2 = FixedCombinations(['A', 'C'])

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
            'A#B': {'sdtype': 'categorical'},
            'C': {'sdtype': 'categorical'},
        }
    }).to_dict()
    assert expected_updated_metadata == updated_metadata.to_dict()

    assert original_metadata.to_dict() == metadata.to_dict()

    # Get unique combinations from original data
    original_ab_combos = set(zip(data['A'], data['B']))
    original_ac_combos = set(zip(data['A'], data['C']))

    # Get unique combinations from synthetic data
    synthetic_ab_combos = set(zip(samples['A'], samples['B']))
    synthetic_ac_combos = set(zip(samples['A'], samples['C']))

    # Assert combinations match
    assert original_ab_combos == synthetic_ab_combos
    assert original_ac_combos == synthetic_ac_combos


def test_fixed_combinations_multiple_patterns_three_patterns():
    """Test that FixedCombinations pattern works with multiple patterns."""
    # Setup
    data = pd.DataFrame({
        'A': [1, 2, 3, 1, 2, 1],
        'B': [10, 20, 30, 10, 20, 10],
        'C': [100, 200, 300, 100, 200, 100],
        'D': [1000, 2000, 3000, 1000, 2000, 1000],
    })
    metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'categorical'},
            'B': {'sdtype': 'categorical'},
            'C': {'sdtype': 'categorical'},
            'D': {'sdtype': 'categorical'},
        }
    })
    pattern1 = FixedCombinations(['A', 'B'])
    pattern2 = FixedCombinations(['C', 'D'])
    pattern3 = FixedCombinations(['A', 'C'])

    # Run
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.add_cag(patterns=[pattern1, pattern2, pattern3])
    synthesizer.fit(data)
    samples = synthesizer.sample(100)
    updated_metadata = synthesizer.get_metadata('modified')
    original_metadata = synthesizer.get_metadata('original')

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'columns': {
            'A#B': {'sdtype': 'categorical'},
            'C#D': {'sdtype': 'categorical'},
        }
    }).to_dict()
    assert expected_updated_metadata == updated_metadata.to_dict()

    assert original_metadata.to_dict() == metadata.to_dict()

    # Get unique combinations from original data
    original_ab_combos = set(zip(data['A'], data['B']))
    original_cd_combos = set(zip(data['C'], data['D']))
    original_ac_combos = set(zip(data['A'], data['C']))

    # Get unique combinations from synthetic data
    synthetic_ab_combos = set(zip(samples['A'], samples['B']))
    synthetic_cd_combos = set(zip(samples['C'], samples['D']))
    synthetic_ac_combos = set(zip(samples['A'], samples['C']))

    # Assert combinations match
    assert original_ab_combos == synthetic_ab_combos
    assert original_cd_combos == synthetic_cd_combos
    assert original_ac_combos == synthetic_ac_combos


def test_fixed_combinations_multiple_patterns_three_patterns_reject_sampling():
    """Test that FixedCombinations pattern works with multiple patterns.

    Test that when the second pattern in the chain fails, the third pattern still works.
    """
    # Setup
    data = pd.DataFrame({
        'A': [1, 2, 3, 1, 2, 1],
        'B': [10, 20, 30, 10, 20, 10],
        'C': [100, 200, 300, 100, 200, 100],
        'D': [1000, 2000, 3000, 1000, 2000, 1000],
    })
    metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'categorical'},
            'B': {'sdtype': 'categorical'},
            'C': {'sdtype': 'categorical'},
            'D': {'sdtype': 'categorical'},
        }
    })
    pattern1 = FixedCombinations(['A', 'B'])
    pattern2 = FixedCombinations(['C', 'D'])
    pattern3 = FixedCombinations(['A', 'C'])

    # Run
    synthesizer = run_copula(data, metadata, [pattern1, pattern3, pattern2])
    samples = synthesizer.sample(100)
    updated_metadata = synthesizer.get_metadata('modified')
    original_metadata = synthesizer.get_metadata('original')

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'columns': {
            'A#B': {'sdtype': 'categorical'},
            'C#D': {'sdtype': 'categorical'},
        }
    }).to_dict()

    assert expected_updated_metadata == updated_metadata.to_dict()

    assert original_metadata.to_dict() == metadata.to_dict()

    # Get unique combinations from original data
    original_ab_combos = set(zip(data['A'], data['B']))
    original_cd_combos = set(zip(data['C'], data['D']))
    original_ac_combos = set(zip(data['A'], data['C']))

    # Get unique combinations from synthetic data
    synthetic_ab_combos = set(zip(samples['A'], samples['B']))
    synthetic_cd_combos = set(zip(samples['C'], samples['D']))
    synthetic_ac_combos = set(zip(samples['A'], samples['C']))

    # Assert combinations match
    assert original_ab_combos == synthetic_ab_combos
    assert original_cd_combos == synthetic_cd_combos
    assert original_ac_combos == synthetic_ac_combos


def test_validate_cag(data, metadata, pattern):
    """Test validate_cag works with synthetic data generated with FixedCombinations."""
    # Setup
    synthesizer = run_copula(data, metadata, [pattern])
    synthetic_data = synthesizer.sample(100)

    # Run
    synthesizer.validate_cag(synthetic_data=synthetic_data)

    # Assert
    original_ab_combos = set(zip(data['A'], data['B']))
    synthetic_ab_combos = set(zip(synthetic_data['A'], synthetic_data['B']))
    assert original_ab_combos == synthetic_ab_combos


def test_validate_cag_raises(data, metadata, pattern):
    """Test validate_cag raises an error with bad synthetic data with FixedCombinations."""
    # Setup
    synthetic_data = data.copy()
    synthetic_data['B'] = [11, 21, 31, 11, 21, 11]
    synthesizer = run_copula(data, metadata, [pattern])
    msg = re.escape(
        'The fixed combinations requirement is not met for row indices: 0, 1, 2, 3, 4, +1 more'
    )

    # Run and Assert
    with pytest.raises(PatternNotMetError, match=msg):
        synthesizer.validate_cag(synthetic_data=synthetic_data)


def test_validate_cag_multi(data_multi, metadata_multi, pattern_multi):
    """Test validate_cag works with multitable data generated with FixedCombinations."""
    # Setup
    synthesizer = run_hma(data_multi, metadata_multi, [pattern_multi])
    synthetic_data = synthesizer.sample(100)

    # Run
    synthesizer.validate_cag(synthetic_data=synthetic_data)

    # Assert
    original_ab_combos = set(zip(data_multi['table1']['A'], data_multi['table1']['B']))
    synthetic_ab_combos = set(zip(synthetic_data['table1']['A'], synthetic_data['table1']['B']))
    assert original_ab_combos == synthetic_ab_combos


def test_validate_cag_multi_raises(data_multi, metadata_multi, pattern_multi):
    """Test validate_cag raises an error with bad multitable data with FixedCombinations."""
    # Setup
    synthetic_data = {
        'table1': pd.DataFrame({
            'A': [1, 2, 3, 1, 2, 1],
            'B': [11, 21, 31, 11, 21, 11],
        }),
        'table2': pd.DataFrame({'id': range(5)}),
    }
    synthesizer = run_hma(data_multi, metadata_multi, [pattern_multi])
    msg = re.escape(
        "Table 'table1': The fixed combinations requirement is "
        'not met for row indices: 0, 1, 2, 3, 4, +1 more'
    )

    # Run and Assert
    with pytest.raises(PatternNotMetError, match=msg):
        synthesizer.validate_cag(synthetic_data=synthetic_data)
