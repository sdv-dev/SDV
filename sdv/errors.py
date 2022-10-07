"""SDV Exceptions."""


class NotFittedError(Exception):
    """Error to raise when sample is called and the model is not fitted."""


class ConstraintsNotMetError(ValueError):
    """Exception raised when the given data is not valid for the constraints."""

    def __init__(self, message=''):
        self.message = message
        super().__init__(self.message)


class InvalidPreprocessingError(Exception):
    """Error raised during an invalid preprocessing step."""
