import pandas as pd
import pytest

from sdv.evaluation.utils import (
    get_combination_overlap,
    get_pii_overlap,
    print_referential_integrity,
)
from sdv.metadata import Metadata
from sdv.single_table.copulas import GaussianCopulaSynthesizer

NO_PII_OVERLAP_MESSAGE = '✅ The synthetic data does not contain any PII values from the real data'


@pytest.fixture()
def data_metadata_parent_child():
    """A parent-child dataset with a primary to foreign key relationship between 1 column."""
    data = {
        'parent': pd.DataFrame({'parent_id': [0, 1, 2, 3], 'col': [1, 2, 3, 4]}),
        'child': pd.DataFrame({'child_id': ['A', 'B', 'C', 'D'], 'parent_id': [0, 1, 1, 4]}),
    }
    metadata = Metadata().load_from_dict({
        'tables': {
            'parent': {
                'columns': {
                    'parent_id': {'sdtype': 'id'},
                    'col': {'sdtype': 'numerical'},
                },
                'primary_key': 'parent_id',
            },
            'child': {
                'columns': {
                    'child_id': {'sdtype': 'id'},
                    'parent_id': {'sdtype': 'id'},
                },
                'primary_key': 'child_id',
            },
        },
        'relationships': [
            {
                'parent_table_name': 'parent',
                'parent_primary_key': 'parent_id',
                'child_table_name': 'child',
                'child_foreign_key': 'parent_id',
            }
        ],
    })
    return data, metadata


@pytest.fixture()
def data_metadata_parent_child_composite_keys():
    """A parent-child dataset whose primary and foreign keys span two columns."""
    data = {
        'parent': pd.DataFrame({
            'A': [1, 1, 2, 3],
            'B': ['X', 'Y', 'X', 'Z'],
        }),
        'child': pd.DataFrame({
            'A': [1, 1, 2, 3, 1, 1, 2, 3, 4, 2],
            'B': ['X', 'Y', 'X', 'Z', 'X', 'Z', 'Y', 'X', 'W', 'Z'],
        }),
    }
    metadata = Metadata().load_from_dict({
        'tables': {
            'parent': {
                'columns': {
                    'A': {'sdtype': 'id'},
                    'B': {'sdtype': 'id'},
                },
                'primary_key': ['A', 'B'],
            },
            'child': {
                'columns': {
                    'A': {'sdtype': 'id'},
                    'B': {'sdtype': 'id'},
                },
            },
        },
        'relationships': [
            {
                'parent_table_name': 'parent',
                'parent_primary_key': ['A', 'B'],
                'child_table_name': 'child',
                'child_foreign_key': ['A', 'B'],
            },
        ],
    })
    return data, metadata


def test_print_referential_integrity(data_metadata_parent_child, capsys):
    """Test ``print_referential_integrity`` with a simple parent-child dataset."""
    # Setup
    synthetic_data, metadata = data_metadata_parent_child

    # Run
    print_referential_integrity(metadata, synthetic_data, 'child', 'parent_id', num_rows=4)

    # Assert
    captured = capsys.readouterr().out
    assert 'Picking random child row: A' in captured
    assert 'Picking random child row: B' in captured
    assert 'Picking random child row: C' in captured
    assert 'Picking random child row: D' in captured
    assert captured.count('✅ Found parent row! parent_id: 0') == 1
    assert captured.count('✅ Found parent row! parent_id: 1') == 2
    assert captured.count('❌ Unable to find the linked parent row') == 1


def test_print_referential_integrity_composite_keys(
    data_metadata_parent_child_composite_keys, capsys
):
    """Test ``print_referential_integrity`` with a parent-child dataset with composite keys.

    The child table has no primary key, so no key value is printed in the heading. Two of its
    ten rows reference a combination that is missing from the parent table.
    """
    # Setup
    synthetic_data, metadata = data_metadata_parent_child_composite_keys

    # Run
    print_referential_integrity(metadata, synthetic_data, 'child', ('A', 'B'), num_rows=10)

    # Assert
    captured = capsys.readouterr().out
    assert captured.count('Picking random child row\n') == 10
    assert captured.count('✅ Found parent row! A: 1, B: X') == 2
    assert captured.count('✅ Found parent row! A: 1, B: Y') == 1
    assert captured.count('✅ Found parent row! A: 2, B: X') == 1
    assert captured.count('✅ Found parent row! A: 3, B: Z') == 1
    assert captured.count('❌ Unable to find the linked parent row') == 5


def test_print_referential_integrity_with_null_foreign_key(data_metadata_parent_child, capsys):
    """Test that a null foreign key is reported as expected rather than as a broken link."""
    # Setup
    synthetic_data, metadata = data_metadata_parent_child
    synthetic_data['child'].loc[0, 'parent_id'] = None

    # Run
    print_referential_integrity(metadata, synthetic_data, 'child', 'parent_id', num_rows=4)

    # Assert
    captured = capsys.readouterr().out
    assert captured.count('✅ Foreign key is null; no linked parent row expected') == 1


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

    return {'customer': real_data}, metadata


def test_get_combination_overlap_end_to_end():
    """Test the overlap of a synthesizer's output against the real data."""
    # Setup
    real_data, metadata = _get_demographic_data()
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(real_data)
    synthetic_data = synthesizer.sample('customer', 10)
    column_names = ['date_of_birth', 'zipcode', 'gender']

    # Run
    result = get_combination_overlap(
        real_data=real_data,
        synthetic_data=synthetic_data,
        table_name='customer',
        column_names=column_names,
        verbose=False,
    )

    # Assert
    real_data = real_data['customer']
    synthetic_data = synthetic_data['customer']
    real_combinations = set(real_data[column_names].itertuples(index=False, name=None))
    synthetic_combinations = set(synthetic_data[column_names].itertuples(index=False, name=None))
    assert isinstance(result, int)
    assert result == len(real_combinations & synthetic_combinations)


def test_get_combination_overlap_with_identical_data(capsys):
    """Test that data copied from the real data overlaps completely."""
    # Setup
    real_data, _ = _get_demographic_data()
    column_names = ['date_of_birth', 'zipcode', 'gender']

    # Run
    result = get_combination_overlap(
        real_data=real_data,
        synthetic_data=real_data.copy(),
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
            real_data['customer'].iloc[[2]],
            pd.DataFrame({
                'date_of_birth': ['1993-04-04'],
                'zipcode': [30301],
                'gender': ['F'],
            }),
        ],
        ignore_index=True,
    )
    synthetic_data = {'customer': synthetic_data}

    # Run
    result = get_combination_overlap(
        real_data=real_data,
        synthetic_data=synthetic_data,
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
