import numpy as np
import pandas as pd
import pytest

from sdv.evaluation.utils import (
    FEW_OVERLAP_MESSAGE,
    FEW_PII_OVERLAP_MESSAGE,
    NO_OVERLAP_MESSAGE,
    NO_PII_OVERLAP_MESSAGE,
    SIGNIFICANT_OVERLAP_MESSAGE,
    SIGNIFICANT_PII_OVERLAP_MESSAGE,
    _get_combinations,
    get_combination_overlap,
    get_pii_overlap,
)


def _wrap(real_table, synthetic_table):
    """Return the real and synthetic data as single-table dictionaries."""
    return {'table': real_table}, {'table': synthetic_table}


class TestGetCombinations:
    def test_returns_unique_combinations(self):
        """Test that repeated rows are only counted as a single combination."""
        # Setup
        data = pd.DataFrame({'a': ['x', 'x', 'y'], 'b': [1, 1, 2]})

        # Run
        result = _get_combinations(data)

        # Assert
        assert result == {('x', 1), ('y', 2)}


class TestGetCombinationOverlap:
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
    def test_reports_the_overlap(
        self, capsys, real_values, synthetic_values, expected_result, expected_summary
    ):
        """Test the reported count, percentage and interpretation for each threshold."""
        # Setup
        real_data, synthetic_data = _wrap(
            pd.DataFrame({'a': real_values}), pd.DataFrame({'a': synthetic_values})
        )
        counts, message = expected_summary

        # Run
        result = get_combination_overlap(real_data, synthetic_data, 'table', ['a'])

        # Assert
        captured = capsys.readouterr()
        assert result == expected_result
        assert captured.out == f'Number of common combinations: {counts}\n{message}\n'

    def test_verbose_false(self, capsys):
        """Test that nothing is printed when ``verbose`` is False."""
        # Setup
        real_data, synthetic_data = _wrap(pd.DataFrame({'a': ['x']}), pd.DataFrame({'a': ['x']}))

        # Run
        result = get_combination_overlap(real_data, synthetic_data, 'table', ['a'], verbose=False)

        # Assert
        captured = capsys.readouterr()
        assert result == 1
        assert captured.out == ''

    def test_with_missing_values(self):
        """Test that rows sharing a pattern of missing values are counted as an overlap."""
        # Setup
        real_data, synthetic_data = _wrap(
            pd.DataFrame({'a': ['x', np.nan], 'b': [1, 2]}),
            pd.DataFrame({'a': [np.nan], 'b': [2]}),
        )

        # Run
        result = get_combination_overlap(
            real_data, synthetic_data, 'table', ['a', 'b'], verbose=False
        )

        # Assert
        assert result == 1

    def test_with_mismatched_datetime_dtypes(self):
        """Test that a parsed and an unparsed datetime column are counted as an overlap.

        A synthetic value that cannot be parsed is coerced to ``NaT`` instead of raising.
        """
        # Setup
        real_data, synthetic_data = _wrap(
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

    def test_with_mismatched_non_numeric_dtypes(self):
        """Test that columns that are neither numeric nor datetime are compared as strings."""
        # Setup
        real_data, synthetic_data = _wrap(
            pd.DataFrame({'a': ['1', 'b']}), pd.DataFrame({'a': [1, 2]})
        )

        # Run
        result = get_combination_overlap(real_data, synthetic_data, 'table', ['a'], verbose=False)

        # Assert
        assert result == 1

    def test_with_empty_tables(self, capsys):
        """Test that empty tables do not raise a ``ZeroDivisionError``."""
        # Setup
        real_data, synthetic_data = _wrap(
            pd.DataFrame({'a': [], 'b': []}), pd.DataFrame({'a': [], 'b': []})
        )

        # Run
        result = get_combination_overlap(real_data, synthetic_data, 'table', ['a', 'b'])

        # Assert
        captured = capsys.readouterr()
        assert result == 0
        assert captured.out == f'Number of common combinations: 0 (0.0%)\n{NO_OVERLAP_MESSAGE}\n'

    def test_does_not_modify_the_input_data(self):
        """Test that the real and synthetic data are not modified in place."""
        # Setup
        real_table = pd.DataFrame({'a': [94301], 'b': ['x']})
        synthetic_table = pd.DataFrame({'a': [94301.0], 'b': ['x']})
        real_data, synthetic_data = _wrap(real_table.copy(), synthetic_table.copy())

        # Run
        get_combination_overlap(real_data, synthetic_data, 'table', ['a', 'b'], verbose=False)

        # Assert
        pd.testing.assert_frame_equal(real_data['table'], real_table)
        pd.testing.assert_frame_equal(synthetic_data['table'], synthetic_table)

    @pytest.mark.parametrize(
        ('table_name', 'column_names', 'expected_message'),
        [
            ('table', [], "'column_names' must contain at least one column name."),
            ('missing', ['a'], "Table 'missing' is not present in 'real_data'."),
            (
                'table',
                ['a', 'b', 'c'],
                "The columns 'b', 'c' are not present in table 'table' of 'real_data'.",
            ),
        ],
        ids=['no_columns', 'missing_table', 'missing_columns'],
    )
    def test_with_invalid_input(self, table_name, column_names, expected_message):
        """Test that invalid tables and columns raise an error."""
        # Setup
        real_data, synthetic_data = _wrap(pd.DataFrame({'a': ['x']}), pd.DataFrame({'a': ['x']}))

        # Run and Assert
        with pytest.raises(ValueError, match=expected_message):
            get_combination_overlap(real_data, synthetic_data, table_name, column_names)

    def test_with_missing_table_in_synthetic_data(self):
        """Test that a table missing from the synthetic data raises an error."""
        # Setup
        real_data = {'table': pd.DataFrame({'a': ['x']})}
        synthetic_data = {'other': pd.DataFrame({'a': ['x']})}

        # Run and Assert
        expected_message = "Table 'table' is not present in 'synthetic_data'."
        with pytest.raises(ValueError, match=expected_message):
            get_combination_overlap(real_data, synthetic_data, 'table', ['a'])


class TestGetPiiOverlap:
    @pytest.mark.parametrize(
        ('real_values', 'synthetic_values', 'expected_result', 'expected_summary'),
        [
            (['a', 'b'], ['y', 'z'], 0, ('0 (0.0%)', NO_PII_OVERLAP_MESSAGE)),
            (list(range(51)), list(range(50, 100)), 1, ('1 (1.0%)', FEW_PII_OVERLAP_MESSAGE)),
            (['a', 'b', 'c'], ['a', 'z'], 1, ('1 (25.0%)', SIGNIFICANT_PII_OVERLAP_MESSAGE)),
        ],
        ids=['none', 'few', 'significant'],
    )
    def test_reports_the_overlap(
        self, capsys, real_values, synthetic_values, expected_result, expected_summary
    ):
        """Test the reported count, percentage and interpretation for each threshold."""
        # Setup
        real_data, synthetic_data = _wrap(
            pd.DataFrame({'ssn': real_values}), pd.DataFrame({'ssn': synthetic_values})
        )
        counts, message = expected_summary

        # Run
        result = get_pii_overlap(real_data, synthetic_data, 'table', 'ssn')

        # Assert
        captured = capsys.readouterr()
        assert result == expected_result
        assert captured.out == f'Number of common data points: {counts}\n{message}\n'
