import pandas as pd

from sdv.cag import ProgrammableConstraint, SingleTableProgrammableConstraint


class MyConstraint(ProgrammableConstraint):
    def __init__(self, column_names, table_name):
        self.column_names = column_names
        self.table_name = table_name

    def fit(self, data, metadata):
        return

    def transform(self, data):
        data[self.table_name][self.column_names] = data[self.table_name][self.column_names] ** 2
        return data

    def reverse_transform(self, transformed_data):
        table_data = transformed_data[self.table_name]
        table_data[self.column_names] = table_data[self.column_names] // 2
        transformed_data[self.table_name] = table_data
        return transformed_data

    def get_updated_metadata(self, metadata):
        return metadata

    def is_valid(self, synthetic_data):
        is_valid = {
            table_name: pd.Series([True] * len(table_data))
            for table_name, table_data in synthetic_data.items()
        }
        table_data = synthetic_data[self.table_name]
        is_valid_table = pd.Series([
            value[0] > 1 for value in table_data[self.column_names].to_numpy()
        ])
        is_valid[self.table_name] = is_valid_table

        return is_valid


class MySingleTableConstraint(SingleTableProgrammableConstraint):
    def __init__(self, column_names):
        self.column_names = column_names

    def fit(self, data, metadata):
        return

    def transform(self, data):
        data[self.column_names] = data[self.column_names] ** 2
        return data

    def reverse_transform(self, transformed_data):
        transformed_data[self.column_names] = transformed_data[self.column_names] // 2
        return transformed_data

    def get_updated_metadata(self, metadata):
        return metadata

    def is_valid(self, synthetic_data):
        return pd.Series([value[0] > 1 for value in synthetic_data[self.column_names].to_numpy()])


class IfTrueThenZero(ProgrammableConstraint):
    def __init__(self, column_names):
        self.column_names = column_names

    def fit(self, data, metadata):
        return

    def transform(self, data):
        data[self.column_names] = data[self.column_names] ** 2
        return data

    def reverse_transform(self, transformed_data):
        transformed_data[self.column_names] = transformed_data[self.column_names] // 2
        return transformed_data

    def get_updated_metadata(self, metadata):
        return metadata

    def is_valid(self, synthetic_data):
        return pd.Series([value[0] > 1 for value in synthetic_data[self.column_names].to_numpy()])


class SingleTableIfTrueThenZero(SingleTableProgrammableConstraint):
    def __init__(self, column_names):
        self.column_names = column_names

    def fit(self, data, metadata):
        return

    def transform(self, data):
        """Transform the data if amenities fee is to be applied."""
        boolean_column = self.column_names[0]
        numerical_column = self.column_names[1]
        typical_value = data[numerical_column].median()
        data[numerical_column] = data[numerical_column].mask(data[boolean_column], typical_value)

        return data

    def reverse_transform(self, transformed_data):
        """Reverse the data if amenities fee is to be applied."""
        boolean_column = self.column_names[0]
        numerical_column = self.column_names[1]
        transformed_data[numerical_column] = transformed_data[numerical_column].mask(
            transformed_data[boolean_column], 0.0
        )

        return transformed_data

    def get_updated_metadata(self, metadata):
        return metadata

    def is_valid(self, synthetic_data):
        """Validate that if ``has_rewards`` amenities fee is 0."""
        boolean_column = self.column_names[0]
        numerical_column = self.column_names[1]
        true_values = (synthetic_data[boolean_column]) & (synthetic_data[numerical_column] == 0.0)
        false_values = ~synthetic_data[boolean_column]

        return (true_values) | (false_values)
