import numpy as np


def is_valid(column_names, data, increment, exclusion_column):
    column_name=column_names[0]

    is_divisible = (data[column_name] % increment == 0)
    is_excluded = (data[exclusion_column] > 0)

    return (is_divisible | is_excluded)


def transform(column_names, data, increment, exclusion_column):
    column_name = column_names[0]
    data[column_name] = data[column_name] / increment
    return data


def reverse_transform(column_names, transformed_data, increment, exclusion_column):
    column_name = column_names[0]

    is_included = (transformed_data[exclusion_column] == 0)
    rounded_data = transformed_data[is_included][column_name].round()
    transformed_data.at[is_included, column_name] = rounded_data

    transformed_data[column_name] *= increment
    return transformed_data
