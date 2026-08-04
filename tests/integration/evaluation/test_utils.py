import pandas as pd

from sdv.evaluation.utils import (
    NO_PII_OVERLAP_MESSAGE,
    get_combination_overlap,
    get_pii_overlap,
)
from sdv.metadata.metadata import Metadata
from sdv.single_table.copulas import GaussianCopulaSynthesizer


def _get_demographic_data():
    """Return real data and metadata for a table of quasi-identifiers."""
    real_data = pd.DataFrame({
        'date_of_birth': [
            '1990-01-01',
            '1985-06-15',
            '1972-11-30',
            '2001-03-22',
            '1968-09-08',
        ],
        'zipcode': [94301, 10001, 60614, 78701, 2139],
        'gender': ['M', 'F', 'F', 'M', 'F'],
    })

    metadata = Metadata()
    metadata.add_table('customer')
    metadata.add_column('date_of_birth', 'customer', sdtype='datetime', datetime_format='%Y-%m-%d')
    metadata.add_column('zipcode', 'customer', sdtype='numerical')
    metadata.add_column('gender', 'customer', sdtype='categorical')

    return real_data, metadata


def test_get_combination_overlap_end_to_end():
    """Test the overlap of a synthesizer's output against the real data."""
    # Setup
    real_data, metadata = _get_demographic_data()
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(real_data)
    synthetic_data = synthesizer.sample(10)
    column_names = ['date_of_birth', 'zipcode', 'gender']

    # Run
    result = get_combination_overlap(
        real_data={'customer': real_data},
        synthetic_data={'customer': synthetic_data},
        table_name='customer',
        column_names=column_names,
        verbose=False,
    )

    # Assert
    real_combinations = set(real_data[column_names].itertuples(index=False, name=None))
    assert isinstance(result, int)
    assert 0 <= result <= len(real_combinations)


def test_get_combination_overlap_with_identical_data(capsys):
    """Test that data copied from the real data overlaps completely."""
    # Setup
    real_data, _ = _get_demographic_data()
    column_names = ['date_of_birth', 'zipcode', 'gender']

    # Run
    result = get_combination_overlap(
        real_data={'customer': real_data},
        synthetic_data={'customer': real_data.copy()},
        table_name='customer',
        column_names=column_names,
    )

    # Assert
    captured = capsys.readouterr()
    assert result == 5
    assert 'Number of common combinations: 5 (100.0%)' in captured.out


def test_get_combination_overlap_detects_a_single_shared_row():
    """Test that one row copied from the real data is detected."""
    # Setup
    real_data, _ = _get_demographic_data()
    synthetic_data = pd.concat(
        [
            real_data.iloc[[2]],
            pd.DataFrame({
                'date_of_birth': ['1993-04-04'],
                'zipcode': [30301],
                'gender': ['F'],
            }),
        ],
        ignore_index=True,
    )

    # Run
    result = get_combination_overlap(
        real_data={'customer': real_data},
        synthetic_data={'customer': synthetic_data},
        table_name='customer',
        column_names=['date_of_birth', 'zipcode', 'gender'],
        verbose=False,
    )

    # Assert
    assert result == 1


def test_get_pii_overlap_with_anonymized_column(capsys):
    """Test that a fully anonymized PII column reports no overlap."""
    # Setup
    real_data = pd.DataFrame({'ssn': ['111-11-1111', '222-22-2222', '333-33-3333']})
    synthetic_data = pd.DataFrame({'ssn': ['999-99-9999', '888-88-8888']})

    # Run
    result = get_pii_overlap(
        real_data={'customer': real_data},
        synthetic_data={'customer': synthetic_data},
        table_name='customer',
        pii_column_name='ssn',
    )

    # Assert
    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == f'Number of common data points: 0 (0.0%)\n{NO_PII_OVERLAP_MESSAGE}\n'
