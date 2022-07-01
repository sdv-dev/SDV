import numpy as np


def is_valid(column_names, data, increment, exclusion_column):
    column_name=column_names[0]

    is_divisible = (data[column_name] % increment == 0)
    is_excluded = (data[exclusion_column] > 0)

    return np.array(is_divisible | is_excluded)


def transform(column_names, data, increment, exclusion_column):
    column_name = column_names[0]
    data[column_name] = data[column_name] / increment
    return data


def reverse_transform(column_names, transformed_data, increment, exclusion_column):
    column_name = column_names[0]

    included = transformed_data[column_name].loc[(transformed_data[exclusion_column] == 0)]
    included = included.round()

    transformed_data[column_name] = transformed_data[column_name].multiply(increment).round(2)
    return transformed_data
