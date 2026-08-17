import re

import pandas as pd
import pytest

from sdv.evaluation.utils import print_referential_integrity
from sdv.metadata import Metadata


def _get_metadata(child_primary_key='child_id'):
    """Return metadata for a parent table and a child table linked by ``parent_id``."""
    child_columns = {'child_id': {'sdtype': 'id'}, 'parent_id': {'sdtype': 'id'}}
    child_table = {'columns': child_columns}
    if child_primary_key:
        child_table['primary_key'] = child_primary_key

    return Metadata().load_from_dict({
        'tables': {
            'parent': {
                'columns': {'parent_id': {'sdtype': 'id'}},
                'primary_key': 'parent_id',
            },
            'child': child_table,
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


def _get_data():
    """Return a parent table and a child table whose last row has a broken reference."""
    return {
        'parent': pd.DataFrame({'parent_id': [0, 1]}),
        'child': pd.DataFrame({'child_id': ['A', 'B'], 'parent_id': [0, 9]}),
    }


def test_print_referential_integrity_reports_found_and_missing_rows(capsys):
    """Test that a valid reference prints a match and a broken one prints a failure."""
    # Setup
    metadata = _get_metadata()
    synthetic_data = _get_data()

    # Run
    print_referential_integrity(metadata, synthetic_data, 'child', 'parent_id', num_rows=2)

    # Assert
    captured = capsys.readouterr().out
    assert 'Picking random child row: A\n✅ Found parent row! parent_id: 0\n' in captured
    assert 'Picking random child row: B\n❌ Unable to find the linked parent row\n' in captured


def test_print_referential_integrity_with_null_foreign_key(capsys):
    """Test that a null foreign key is not reported as a broken reference."""
    # Setup
    metadata = _get_metadata()
    synthetic_data = _get_data()
    synthetic_data['child']['parent_id'] = [None, None]

    # Run
    print_referential_integrity(metadata, synthetic_data, 'child', 'parent_id', num_rows=2)

    # Assert
    captured = capsys.readouterr().out
    assert captured.count('✅ Foreign key is null; no linked parent row expected') == 2
    assert '❌' not in captured


def test_print_referential_integrity_without_a_primary_key(capsys):
    """Test that no key value is printed when the child table has no primary key."""
    # Setup
    metadata = _get_metadata(child_primary_key=None)
    synthetic_data = _get_data()

    # Run
    print_referential_integrity(metadata, synthetic_data, 'child', 'parent_id', num_rows=2)

    # Assert
    captured = capsys.readouterr().out
    assert captured.count('Picking random child row\n') == 2


def test_print_referential_integrity_limits_the_number_of_rows(capsys):
    """Test that only ``num_rows`` rows are checked."""
    # Setup
    metadata = _get_metadata()
    synthetic_data = _get_data()

    # Run
    print_referential_integrity(metadata, synthetic_data, 'child', 'parent_id', num_rows=1)

    # Assert
    captured = capsys.readouterr().out
    assert captured.count('Picking random child row') == 1


def test_print_referential_integrity_with_too_few_rows(capsys):
    """Test that ``num_rows`` is lowered to the table size and a warning is raised."""
    # Setup
    metadata = _get_metadata()
    synthetic_data = _get_data()

    # Run
    expected_warning = re.escape(
        "The synthetic data contains '2' rows which is less than num_rows: '5'. "
        "Changing num_rows to '2'."
    )
    with pytest.warns(UserWarning, match=expected_warning):
        print_referential_integrity(metadata, synthetic_data, 'child', 'parent_id', num_rows=5)

    # Assert
    captured = capsys.readouterr().out
    assert captured.count('Picking random child row') == 2


@pytest.mark.parametrize(
    ('num_rows', 'expected_error', 'expected_message'),
    [
        (0, ValueError, "'num_rows' must be an integer greater than 0."),
        (-1, ValueError, "'num_rows' must be an integer greater than 0."),
        (1.5, TypeError, "'num_rows' must be an integer greater than 0."),
        (True, TypeError, "'num_rows' must be an integer greater than 0."),
    ],
    ids=['zero', 'negative', 'float', 'boolean'],
)
def test_print_referential_integrity_with_invalid_num_rows(
    num_rows, expected_error, expected_message
):
    """Test that ``num_rows`` must be an integer greater than 0."""
    # Setup
    metadata = _get_metadata()
    synthetic_data = _get_data()

    # Run and Assert
    with pytest.raises(expected_error, match=re.escape(expected_message)):
        print_referential_integrity(metadata, synthetic_data, 'child', 'parent_id', num_rows)


@pytest.mark.parametrize(
    ('metadata_value', 'table_name', 'foreign_key_name', 'expected_error', 'expected_message'),
    [
        (None, 'child', 'parent_id', TypeError, 'metadata must be of Metadata type.'),
        ('valid', 123, 'parent_id', TypeError, 'table_name must be a string.'),
        (
            'valid',
            'child',
            123,
            TypeError,
            'foreign_key_name must be a string or a tuple of strings.',
        ),
        (
            'valid',
            'missing',
            'parent_id',
            ValueError,
            "table_name: 'missing' not found in metadata.",
        ),
        (
            'valid',
            'child',
            'missing',
            ValueError,
            "foreign_key_name: 'missing' not in Metadata for table_name: 'child'.",
        ),
    ],
    ids=[
        'metadata_not_a_metadata',
        'table_name_not_a_string',
        'foreign_key_not_a_string',
        'table_name_not_in_metadata',
        'foreign_key_not_in_metadata',
    ],
)
def test_print_referential_integrity_with_invalid_input(
    metadata_value, table_name, foreign_key_name, expected_error, expected_message
):
    """Test that invalid metadata, tables and foreign keys raise an error."""
    # Setup
    metadata = _get_metadata() if metadata_value == 'valid' else metadata_value
    synthetic_data = _get_data()

    # Run and Assert
    with pytest.raises(expected_error, match=re.escape(expected_message)):
        print_referential_integrity(metadata, synthetic_data, table_name, foreign_key_name)


def test_print_referential_integrity_without_a_relationship():
    """Test that a foreign key with no matching relationship raises an error."""
    # Setup
    metadata = _get_metadata()
    synthetic_data = _get_data()

    # Run and Assert
    expected_message = re.escape(
        "Unable to find a relationship in metadata given table_name: 'child' and "
        "foreign_key_name: 'child_id'."
    )
    with pytest.raises(ValueError, match=expected_message):
        print_referential_integrity(metadata, synthetic_data, 'child', 'child_id')


def test_print_referential_integrity_with_reordered_composite_key(capsys):
    """Test that a composite foreign key may be given in any order.

    The metadata pairs each foreign key column with a primary key column by position, so the
    values must be looked up using the order the relationship defines, not the order passed in.
    """
    # Setup
    metadata = Metadata().load_from_dict({
        'tables': {
            'parent': {
                'columns': {'P': {'sdtype': 'id'}, 'Q': {'sdtype': 'id'}},
                'primary_key': ['P', 'Q'],
            },
            'child': {'columns': {'A': {'sdtype': 'id'}, 'B': {'sdtype': 'id'}}},
        },
        'relationships': [
            {
                'parent_table_name': 'parent',
                'parent_primary_key': ['P', 'Q'],
                'child_table_name': 'child',
                'child_foreign_key': ['A', 'B'],
            }
        ],
    })
    synthetic_data = {
        'parent': pd.DataFrame({'P': [1], 'Q': ['X']}),
        'child': pd.DataFrame({'A': [1], 'B': ['X']}),
    }

    # Run
    print_referential_integrity(metadata, synthetic_data, 'child', ('B', 'A'), num_rows=1)

    # Assert
    captured = capsys.readouterr().out
    assert '✅ Found parent row! P: 1, Q: X' in captured
