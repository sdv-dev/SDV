"""Data processing exceptions."""


class NotFittedError(Exception):
    """Error to raise when ``DataProcessor`` is used before fitting."""
