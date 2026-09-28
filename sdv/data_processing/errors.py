"""Data processing exceptions."""


class NotFittedError(Exception):
    """Error to raise when ``DataProcessor`` is used before fitting."""


class InvalidConstraintsError(Exception):
    """Error to raise when constraints are not valid."""

    def __init__(self, errors):
        errors = errors if isinstance(errors, list) else [errors]
        self.errors = errors

    def __str__(self):
        return 'The provided constraint is invalid:\n' + '\n\n'.join(map(str, self.errors))
