import re

import numpy as np
import pandas as pd
import pytest

from sdv.evaluation.utils import (
    _get_combinations,
    get_combination_overlap,
    get_pii_overlap,
    print_referential_integrity,
)
from sdv.metadata import Metadata

NO_OVERLAP_MESSAGE = (
    '✅ The synthetic data does not contain any of the same combinations from the real data'
)
FEW_OVERLAP_MESSAGE = (
    '⚠️ The synthetic data contains a few of the same combinations as the real data. '
    'This might be due to random chance.'
)
SIGNIFICANT_OVERLAP_MESSAGE = (
    '❌ The synthetic data contains a significant number of the same combinations as the real '
    'data. This might be due to a small number of possible combinations, a large sample of '
    'synthetic data, or a misconfiguration in your synthesizer.'
)
NO_PII_OVERLAP_MESSAGE = '✅ The synthetic data does not contain any PII values from the real data'
FEW_PII_OVERLAP_MESSAGE = (
    '⚠️ The synthetic data contains a few PII values from the real data. '
    'This might be due to random chance.'
)
SIGNIFICANT_PII_OVERLAP_MESSAGE = (
    '❌ The synthetic data contains a significant number of the same PII values of as the real '
    'data. This might be due to a small number of possible PII values, a large sample of '
    'synthetic data, or a misconfiguration in your synthesizer.'
)


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


def _pandas_to_table_dicts(real_table, synthetic_table):
    """Return the real and synthetic data as single-table dictionaries."""
    return {'table': real_table}, {'table': synthetic_table}


def test__get_combinations_returns_unique_combinations():
    """Test that repeated rows are only counted as a single combination."""
    # Setup
    data = pd.DataFrame({'a': ['x', 'x', 'y'], 'b': [1, 1, 2]})

    # Run
    result = _get_combinations(data)

    # Assert
    assert result == {('x', 1), ('y', 2)}


@pytest.mark.parametrize(
    ('real_values', 'synthetic_values', 'expected_result', 'expected_summary'),
    [
        (['x', 'y'], ['q', 'r'], 0, ('0 (0.0%)', NO_OVERLAP_MESSAGE)),
        (list(range(51)), list(range(50, 100)), 1, ('1 (1.0%)', FEW_OVERLAP_MESSAGE)),
        (list(range(51)), list(range(49, 100)), 2, ('2 (2.0%)', FEW_OVERLAP_MESSAGE)),
        (['x', 'y', 'z'], ['x', 'q'], 1, ('1 (25.0%)', SIGNIFICANT_OVERLAP_MESSAGE)),
    ],
    ids=['none', 'few', 'two_percent_boundary', 'significant'],
)
def test_get_combination_overlap_reports_the_overlap(
    capsys, real_values, synthetic_values, expected_result, expected_summary
):
    """Test the reported count, percentage and interpretation for each threshold."""
    # Setup
    real_data, synthetic_data = _pandas_to_table_dicts(
        pd.DataFrame({'a': real_values}), pd.DataFrame({'a': synthetic_values})
    )
    counts, message = expected_summary

    # Run
    result = get_combination_overlap(real_data, synthetic_data, 'table', ['a'])

    # Assert
    captured = capsys.readouterr()
    assert result == expected_result
    assert captured.out == f'Number of common combinations: {counts}\n{message}\n'


def test_get_combination_overlap_verbose_false(capsys):
    """Test that nothing is printed when ``verbose`` is False."""
    # Setup
    real_data, synthetic_data = _pandas_to_table_dicts(
        pd.DataFrame({'a': ['x']}), pd.DataFrame({'a': ['x']})
    )

    # Run
    result = get_combination_overlap(real_data, synthetic_data, 'table', ['a'], verbose=False)

    # Assert
    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ''


def test_get_combination_overlap_with_missing_values():
    """Test that rows sharing a pattern of missing values are counted as an overlap."""
    # Setup
    real_data, synthetic_data = _pandas_to_table_dicts(
        pd.DataFrame({'a': ['x', np.nan], 'b': [1, 2]}),
        pd.DataFrame({'a': [np.nan], 'b': [2]}),
    )

    # Run
    result = get_combination_overlap(real_data, synthetic_data, 'table', ['a', 'b'], verbose=False)

    # Assert
    assert result == 1


def test_get_combination_overlap_with_mismatched_datetime_dtypes():
    """Test that a parsed and an unparsed datetime column are counted as an overlap.

    A synthetic value that cannot be parsed is coerced to ``NaT`` instead of raising.
    """
    # Setup
    real_data, synthetic_data = _pandas_to_table_dicts(
        pd.DataFrame({
            'date_of_birth': pd.to_datetime(['2020-01-01', '2020-01-02']),
            'gender': ['M', 'F'],
        }),
        pd.DataFrame({
            'date_of_birth': ['2020-01-01', 'not-a-date'],
            'gender': ['M', 'F'],
        }),
    )

    # Run
    result = get_combination_overlap(
        real_data, synthetic_data, 'table', ['date_of_birth', 'gender'], verbose=False
    )

    # Assert
    assert result == 1


def test_get_combination_overlap_with_mismatched_numeric_dtypes():
    """Test that the same value stored as an int and a float is counted as an overlap."""
    # Setup
    real_data, synthetic_data = _pandas_to_table_dicts(
        pd.DataFrame({'zipcode': [94301, 94302]}), pd.DataFrame({'zipcode': [94301.0, 94999.0]})
    )

    # Run
    result = get_combination_overlap(real_data, synthetic_data, 'table', ['zipcode'], verbose=False)

    # Assert
    assert result == 1


def test_get_combination_overlap_with_mismatched_non_numeric_dtypes():
    """Test that columns that are neither numeric nor datetime are compared as strings."""
    # Setup
    real_data, synthetic_data = _pandas_to_table_dicts(
        pd.DataFrame({'a': ['1', 'b']}), pd.DataFrame({'a': [1, 2]})
    )

    # Run
    result = get_combination_overlap(real_data, synthetic_data, 'table', ['a'], verbose=False)

    # Assert
    assert result == 1


def test_get_combination_overlap_with_empty_tables(capsys):
    """Test that empty tables do not raise a ``ZeroDivisionError``."""
    # Setup
    real_data, synthetic_data = _pandas_to_table_dicts(
        pd.DataFrame({'a': [], 'b': []}), pd.DataFrame({'a': [], 'b': []})
    )

    # Run
    result = get_combination_overlap(real_data, synthetic_data, 'table', ['a', 'b'])

    # Assert
    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == f'Number of common combinations: 0 (0.0%)\n{NO_OVERLAP_MESSAGE}\n'


def test_get_combination_overlap_does_not_modify_the_input_data():
    """Test that the real and synthetic data are not modified in place."""
    # Setup
    real_table = pd.DataFrame({'a': [94301], 'b': ['x']})
    synthetic_table = pd.DataFrame({'a': [94301.0], 'b': ['x']})
    real_data, synthetic_data = _pandas_to_table_dicts(real_table.copy(), synthetic_table.copy())

    # Run
    get_combination_overlap(real_data, synthetic_data, 'table', ['a', 'b'], verbose=False)

    # Assert
    pd.testing.assert_frame_equal(real_data['table'], real_table)
    pd.testing.assert_frame_equal(synthetic_data['table'], synthetic_table)


@pytest.mark.parametrize(
    ('table_name', 'column_names', 'expected_error', 'expected_message'),
    [
        (123, ['a'], TypeError, "'table_name' must be a string, got int."),
        ('table', 'a', TypeError, "'column_names' must be a list of strings."),
        ('table', ['a', 2], TypeError, "'column_names' must be a list of strings."),
        ('table', [], ValueError, "'column_names' must contain at least one column name."),
        ('missing', ['a'], ValueError, "Table 'missing' is not present in 'real_data'."),
        (
            'table',
            ['a', 'b', 'c'],
            ValueError,
            "The columns 'b', 'c' are not present in table 'table' of 'real_data'.",
        ),
    ],
    ids=[
        'table_name_not_a_string',
        'column_names_not_a_list',
        'column_names_not_all_strings',
        'no_columns',
        'missing_table',
        'missing_columns',
    ],
)
def test_get_combination_overlap_with_invalid_input(
    table_name, column_names, expected_error, expected_message
):
    """Test that invalid argument types, tables and columns raise an error."""
    # Setup
    real_data, synthetic_data = _pandas_to_table_dicts(
        pd.DataFrame({'a': ['x']}), pd.DataFrame({'a': ['x']})
    )

    # Run and Assert
    with pytest.raises(expected_error, match=expected_message):
        get_combination_overlap(real_data, synthetic_data, table_name, column_names)


def test_get_combination_overlap_with_missing_table_in_synthetic_data():
    """Test that a table missing from the synthetic data raises an error."""
    # Setup
    real_data = {'table': pd.DataFrame({'a': ['x']})}
    synthetic_data = {'other': pd.DataFrame({'a': ['x']})}

    # Run and Assert
    expected_message = "Table 'table' is not present in 'synthetic_data'."
    with pytest.raises(ValueError, match=expected_message):
        get_combination_overlap(real_data, synthetic_data, 'table', ['a'])


@pytest.mark.parametrize(
    ('real_values', 'synthetic_values', 'expected_result', 'expected_summary'),
    [
        (['a', 'b'], ['y', 'z'], 0, ('0 (0.0%)', NO_PII_OVERLAP_MESSAGE)),
        (list(range(51)), list(range(50, 100)), 1, ('1 (1.0%)', FEW_PII_OVERLAP_MESSAGE)),
        (['a', 'b', 'c'], ['a', 'z'], 1, ('1 (25.0%)', SIGNIFICANT_PII_OVERLAP_MESSAGE)),
    ],
    ids=['none', 'few', 'significant'],
)
def test_get_pii_overlap_reports_the_overlap(
    capsys, real_values, synthetic_values, expected_result, expected_summary
):
    """Test the reported count, percentage and interpretation for each threshold."""
    # Setup
    real_data, synthetic_data = _pandas_to_table_dicts(
        pd.DataFrame({'ssn': real_values}), pd.DataFrame({'ssn': synthetic_values})
    )
    counts, message = expected_summary

    # Run
    result = get_pii_overlap(real_data, synthetic_data, 'table', 'ssn')

    # Assert
    captured = capsys.readouterr()
    assert result == expected_result
    assert captured.out == f'Number of common data points: {counts}\n{message}\n'


def test_get_pii_overlap_with_invalid_column_name():
    """Test that a non-string PII column name raises an error."""
    # Setup
    real_data, synthetic_data = _pandas_to_table_dicts(
        pd.DataFrame({'ssn': ['a']}), pd.DataFrame({'ssn': ['a']})
    )

    # Run and Assert
    expected_message = "'pii_column_name' must be a string, got list."
    with pytest.raises(TypeError, match=expected_message):
        get_pii_overlap(real_data, synthetic_data, 'table', ['ssn'])
