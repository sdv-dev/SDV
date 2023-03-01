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
    data[column_names] = data[column_names] // 2
    return data


MyConstraint = create_custom_constraint_class(
    is_valid,
    transform,
    reverse_transform
)


def amenities_is_valid(column_names, data):
    """Validate that if ``has_rewards`` amenities fee is 0."""
    boolean_column = column_names[0]
    numerical_column = column_names[1]
    true_values = (data[boolean_column]) & (data[numerical_column] == 0.0)
    false_values = ~data[boolean_column]

    return (true_values) | (false_values)


def amenities_transform(column_names, data):
    """Transform the data if amenities fee is to be applied."""
    boolean_column = column_names[0]
    numerical_column = column_names[1]
    typical_value = data[numerical_column].median()
    data[numerical_column] = data[numerical_column].mask(
        data[boolean_column],
        typical_value
    )

    return data


def amenities_reverse_transform(column_names, data):
    """Reverse the data if amenities fee is to be applied."""
    boolean_column = column_names[0]
    numerical_column = column_names[1]
    data[numerical_column] = data[numerical_column].mask(data[boolean_column], 0.0)

    return data


IfTrueThenZero = create_custom_constraint_class(
    amenities_is_valid,
    amenities_transform,
    amenities_reverse_transform
)
