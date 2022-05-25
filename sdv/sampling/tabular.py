"""SDV Condition class for sampling."""


class Condition():
    """Condition class.

    This class represents a condition that is used for sampling.

    Attributes:
        column_values (dict):
            A dictionary representing the desired conditions. A mapping of
            the column name to column value, which will be satisfied in this
            condition.
        num_rows (int):
            The number of rows to generate for this condition. Defaults to 1.
    """

    column_values = {}
    num_rows = 1

    def __init__(self, column_values, num_rows=1):
        self.column_values = column_values
        self.num_rows = num_rows

    def get_column_values(self):
        """Get the column value mappings in this condition."""
        return self.column_values.copy()

    def get_num_rows(self):
        """Get the desired number of rows for this condition."""
        return self.num_rows
