"""SDV Exceptions."""

import logging
import traceback

LOGGER = logging.getLogger(__name__)


def log_exc_stacktrace(logger, error):
    """Log the stack trace of an exception.

    Args:
        logger (logging.Logger):
            A logger object to use for the logging.
        error (Exception):
            The error to log.
    """
    message = ''.join(traceback.format_exception(type(error), error, error.__traceback__))
    logger.debug(message)


class NotFittedError(Exception):
    """Error to raise when sample is called and the model is not fitted."""


class ConstraintsNotMetError(ValueError):
    """Exception raised when the given data is not valid for the constraints."""

    def __init__(self, message=''):
        self.message = message
        super().__init__(self.message)


class SynthesizerInputError(Exception):
    """Error to raise when a bad input is provided to a ``Synthesizer``."""


class SamplingError(Exception):
    """Error to raise when sampling gets a bad input or can't be used."""


class NonParametricError(Exception):
    """Exception to indicate that a model is not parametric."""


class InvalidDataError(Exception):
    """Error to raise when data is not valid."""

    def __init__(self, errors):
        self.errors = errors

    def __str__(self):
        return (
            'The provided data does not match the metadata:\n' +
            '\n\n'.join(map(str, self.errors))
        )


class VisualizationUnavailableError(Exception):
    """Exception to indicate that a visualization is unavailable."""
