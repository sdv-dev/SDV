"""Constraint Exceptions."""


class MissingConstraintColumnError(Exception):
    """Error used when constraint is provided a table with missing columns."""

    def __init__(self, missing_columns):
        self.missing_columns = missing_columns


class MultipleConstraintsErrors(Exception):
    """Error used to represent a list of constraint errors."""


class InvalidFunctionError(Exception):
    """Error used when an invalid function is utilized."""


class FunctionError(Exception):
    """Error used when an a function produces an unexpected error."""
