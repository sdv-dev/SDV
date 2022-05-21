"""Constraint Exceptions."""


class MissingConstraintColumnError(Exception):
    """Error to use when constraint is provided a table with missing columns."""


class MultipleConstraintsErrors(Exception):
    """Error used to represent a list of constraint errors."""
