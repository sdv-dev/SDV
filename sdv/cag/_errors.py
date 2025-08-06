"""Constraint Exceptions."""


class ConstraintNotMetError(Exception):
    """Error to raise when a constraint is not met."""


class ConstraintNotMetWarning(Warning):
    """Warning to raise when a constraint is not met but the user can continue."""
