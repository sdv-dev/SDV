"""Constraint Exceptions."""


class MissingConstraintColumnError(Exception):
    """Error to use when constraint is provided a table with missing columns."""

    def __init__(self, missing_columns):
        self.missing_columns = missing_columns


class MultipleConstraintsErrors(Exception):
    """Error used to represent a list of constraint errors."""
