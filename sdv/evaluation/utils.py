"""Utility methods to compare the real and synthetic data."""

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

MISSING_VALUE_PLACEHOLDER = '__sdv_missing_value__'

NO_OVERLAP_MESSAGE = (
    '✅ The synthetic data does not contain any of the same combinations from the real data'
)
FEW_OVERLAP_MESSAGE = (
    '⚠️ The synthetic data contains a few of the same combinations as the real data. '
    'This might be due to random chance.'
)
SIGNIFICANT_OVERLAP_MESSAGE = (
    '❌ The synthetic data contains a significant number of the same combinations as the '
    'real data. This might be due to a small number of possible combinations, a large sample '
    'of synthetic data, or a misconfiguration in your synthesizer.'
)

NO_PII_OVERLAP_MESSAGE = '✅ The synthetic data does not contain any PII values from the real data'
FEW_PII_OVERLAP_MESSAGE = (
    '⚠️ The synthetic data contains a few PII values from the real data. '
    'This might be due to random chance.'
)
SIGNIFICANT_PII_OVERLAP_MESSAGE = (
    '❌ The synthetic data contains a significant number of the same PII values of as the '
    'real data. This might be due to a small number of possible PII values, a large sample '
    'of synthetic data, or a misconfiguration in your synthesizer.'
)


def _validate_data(real_data, synthetic_data, table_name, column_names):
    """Validate that both datasets contain the table and columns to check."""
    if not column_names:
        raise ValueError("'column_names' must contain at least one column name.")

    for argument_name, data in [('real_data', real_data), ('synthetic_data', synthetic_data)]:
        if table_name not in data:
            raise ValueError(f"Table '{table_name}' is not present in '{argument_name}'.")

        missing = [column for column in column_names if column not in data[table_name].columns]
        if missing:
            missing_columns = "', '".join(missing)
            raise ValueError(
                f"The columns '{missing_columns}' are not present in table '{table_name}' "
                f"of '{argument_name}'."
            )


def _align_dtypes(real_column, synthetic_column):
    """Make sure data types of columns being evaluated match.

    Args:
        real_column (pd.Series):
            The column of real data.
        synthetic_column (pd.Series):
            The column of synthetic data.

    Returns:
        tuple[pd.Series, pd.Series]:
            The real and synthetic column, cast to a comparable dtype.
    """
    if real_column.dtype == synthetic_column.dtype:
        return real_column, synthetic_column

    if is_numeric_dtype(real_column) and is_numeric_dtype(synthetic_column):
        return real_column.astype('float64'), synthetic_column.astype('float64')

    if is_datetime64_any_dtype(real_column) or is_datetime64_any_dtype(synthetic_column):
        return (
            pd.to_datetime(real_column, errors='coerce'),
            pd.to_datetime(synthetic_column, errors='coerce'),
        )

    return real_column.astype(str), synthetic_column.astype(str)


def _get_combinations(data):
    """Get the set of unique combinations of values in the data."""
    combinations = data.astype('object')
    combinations = combinations.where(combinations.notna(), MISSING_VALUE_PLACEHOLDER)

    return set(combinations.itertuples(index=False, name=None))


def _compute_overlap(real_data, synthetic_data, table_name, column_names):
    """Get the number of combinations shared by both datasets and the percentage they represent.

    Args:
        real_data (dict):
            A dictionary mapping a table name to a pandas DataFrame containing real data.
        synthetic_data (dict):
            A dictionary mapping a table name to a pandas DataFrame containing synthetic data.
        table_name (str):
            The name of the table that contains the columns to check.
        column_names (list[str]):
            The column names to combine.

    Returns:
        tuple[int, float]:
            The number of shared combinations and their percentage of all combinations.
    """
    real_values = real_data[table_name][column_names].copy()
    synthetic_values = synthetic_data[table_name][column_names].copy()
    for column_name in column_names:
        real_values[column_name], synthetic_values[column_name] = _align_dtypes(
            real_values[column_name], synthetic_values[column_name]
        )

    real_combinations = _get_combinations(real_values)
    synthetic_combinations = _get_combinations(synthetic_values)

    num_common = len(real_combinations & synthetic_combinations)
    num_total = len(real_combinations | synthetic_combinations)
    percent = round(num_common / num_total * 100, 2) if num_total else 0.0

    return num_common, percent


def get_combination_overlap(real_data, synthetic_data, table_name, column_names, verbose=True):
    """Calculate the overlap of combinations of column values between real and synthetic data.

    Args:
        real_data (dict):
            A dictionary mapping a table name to a pandas DataFrame containing real data.
        synthetic_data (dict):
            A dictionary mapping a table name to a pandas DataFrame containing synthetic data.
        table_name (str):
            The name of the table that contains the columns to check.
        column_names (list[str]):
            A list of strings representing the column names to check. Combinations of these
            columns will be checked.
        verbose (bool):
            Whether to print out the interpretation of the results. Defaults to ``True``.

    Returns:
        int:
            The number of unique combinations that appear in both the real and synthetic data.

    Raises:
        ValueError:
            If the table or any of the columns is missing from the data.
    """
    _validate_data(real_data, synthetic_data, table_name, column_names)
    num_common, percent = _compute_overlap(real_data, synthetic_data, table_name, column_names)

    if verbose:
        print(f'Number of common combinations: {num_common} ({percent}%)')  # noqa: T201
        if num_common == 0:
            print(NO_OVERLAP_MESSAGE)  # noqa: T201
        elif percent <= 2:
            print(FEW_OVERLAP_MESSAGE)  # noqa: T201
        else:
            print(SIGNIFICANT_OVERLAP_MESSAGE)  # noqa: T201

    return num_common


def get_pii_overlap(real_data, synthetic_data, table_name, pii_column_name, verbose=True):
    """Calculate the overlap of PII values between the real and synthetic data.

    Args:
        real_data (dict):
            A dictionary mapping a table name to a pandas DataFrame containing real data.
        synthetic_data (dict):
            A dictionary mapping a table name to a pandas DataFrame containing synthetic data.
        table_name (str):
            The name of the table that contains the PII column to check.
        pii_column_name (str):
            The name of the column that contains PII values to check.
        verbose (bool):
            Whether to print out the interpretation of the results. Defaults to ``True``.

    Returns:
        int:
            The number of unique PII values that appear in both the real and synthetic data.

    Raises:
        ValueError:
            If the table or the column is missing from the data.
    """
    column_names = [pii_column_name]
    _validate_data(real_data, synthetic_data, table_name, column_names)
    num_common, percent = _compute_overlap(real_data, synthetic_data, table_name, column_names)

    if verbose:
        print(f'Number of common data points: {num_common} ({percent}%)')  # noqa: T201
        if num_common == 0:
            print(NO_PII_OVERLAP_MESSAGE)  # noqa: T201
        elif percent <= 2:
            print(FEW_PII_OVERLAP_MESSAGE)  # noqa: T201
        else:
            print(SIGNIFICANT_PII_OVERLAP_MESSAGE)  # noqa: T201

    return num_common
