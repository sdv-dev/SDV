import pandas as pd

from sdv.constraints import create_custom_constraint_class


def is_valid(column_names, data):
    """Validate the constraint."""
    return pd.Series([True for _ in range(len(data[column_names]))])


def transform(column_names, data):
    """Transform the constraint."""
    return data[column_names] ** 2


def reverse_transform(column_names, data):
    """Reverse transform the constraint."""
    return data[column_names] // 2


MyConstraint = create_custom_constraint_class(
    is_valid,
    transform,
    reverse_transform
)
