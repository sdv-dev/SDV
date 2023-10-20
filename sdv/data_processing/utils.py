"""Utility functions for data processing."""

import importlib
import sys
import traceback


def load_module_from_path(path):
    """Return the module from a given ``PosixPath``.

    Args:
        path (pathlib.Path):
            A ``PosixPath`` object from where the module should be imported from.

    Returns:
        module:
            The in memory module for the given file.
    """
    assert path.exists(), 'The expected file was not found.'
    module_path = path.parent
    module_name = path.name.split('.')[0]
    module_path = f'{module_path.name}.{module_name}'
    spec = importlib.util.spec_from_file_location(module_path, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def log_exception(logger):
    """Log the traceback of a caught exception.

    Args:
        logger (logging.Logger):
            A logger object to use for the logging.
    """
    exc_type, exc_value, exc_traceback = sys.exc_info()
    message = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    logger.debug(message)
