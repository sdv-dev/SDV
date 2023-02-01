import pandas as pd

from sdv.constraints import create_custom_constraint_class


def is_valid(column_names, data):
    """Validate the constraint."""
    return pd.Series([value[0] > 1 for value in data[column_names].to_numpy()])


def transform(column_names, data):
    """Transform the constraint."""
    data[column_names] = data[column_names] ** 2
    return data


def reverse_transform(column_names, data):
    """Reverse transform the constraint."""
    data[column_names] =  data[column_names] // 2
    return data


MyConstraint = create_custom_constraint_class(
    is_valid,
    transform,
    reverse_transform
)
