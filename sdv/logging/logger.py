"""SDV Logger."""

import csv
import logging
import os
from functools import lru_cache, wraps
from io import StringIO

from sdv.logging.utils import get_sdv_logger_config


class CSVFormatter(logging.Formatter):
    """Logging formatter to convert to CSV."""

    def __init__(self, filename=None):
        super().__init__()
        self.output = StringIO()
        headers = [
            'LEVEL',
            'EVENT',
            'TIMESTAMP',
            'SYNTHESIZER CLASS NAME',
            'SYNTHESIZER ID',
            'TOTAL NUMBER OF TABLES',
            'TOTAL NUMBER OF ROWS',
            'TOTAL NUMBER OF COLUMNS',
        ]
        self.writer = csv.DictWriter(self.output, fieldnames=headers)
        if filename:
            file_exists = os.path.exists(filename)
            if not file_exists:
                self.writer.writeheader()

    def format(self, record):  # noqa: A003
        """Format the record and write to CSV."""
        row = record.msg.copy()
        row['LEVEL'] = record.levelname
        self.writer.writerow(row)
        data = self.output.getvalue()
        self.output.truncate(0)
        self.output.seek(0)
        return data.strip()


def safely_return_logger(func):
    """Decorator to safely return a logger from a function that may raise a `PermissionError`.

    If the decorated function raises a `PermissionError` (commonly due to file-based
    logging to an unwritable path), this decorator catches the error and returns a
    fallback logger configured with a NullHandler instead. This prevents the application
    from crashing due to logging setup failures.

    The fallback logger is named after the current module (__name__) and will silently
    discard all logs.

    Args:
        func (Callable):
            A function that returns a logger and may raise PermissionError.

    Returns:
        Callable:
            A wrapped function that returns either the intended logger or a
            fallback logger with a NullHandler.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except PermissionError:
            fallback_logger = logging.getLogger(__name__)
            fallback_logger.addHandler(logging.NullHandler())
            fallback_logger.warning('Falling back to NullHandler logger due to PermissionError.')
            return fallback_logger

        except Exception:
            return logging.getLogger(__name__)

    return wrapper


@lru_cache()
@safely_return_logger
def get_sdv_logger(logger_name):
    """Get a logger instance with the specified name and configuration.

    This function retrieves or creates a logger instance with the specified name
    and applies configuration settings based on the logger's name and the logging
    configuration.

    Args:
        logger_name (str):
            The name of the logger to retrieve or create.

    Returns:
        logging.Logger:
            A logger instance configured according to the logging configuration
            and the specific settings for the given logger name.
    """
    logger_conf = get_sdv_logger_config()
    logger = logging.getLogger(logger_name)
    if logger_conf.get('log_registry') is None:
        # Return a logger without any extra settings and avoid writing into files or other streams
        return logger

    if logger_conf.get('log_registry') == 'local':
        for handler in logger.handlers:
            # Remove handlers that could exist previously
            logger.removeHandler(handler)

        if logger_name in logger_conf.get('loggers'):
            formatter = None
            config = logger_conf.get('loggers').get(logger_name)
            log_level = getattr(logging, config.get('level', 'INFO'))
            if config.get('formatter'):
                if config.get('formatter') == 'sdv.logging.logger.CSVFormatter':
                    filename = config.get('handlers').get('filename')
                    formatter = CSVFormatter(filename=filename)
            elif config.get('format'):
                formatter = logging.Formatter(config.get('format'))

            logger.setLevel(log_level)
            logger.propagate = config.get('propagate', False)
            handler = config.get('handlers')
            handlers = handler.get('class')
            handlers = [handlers] if isinstance(handlers, str) else handlers
            for handler_class in handlers:
                if handler_class == 'logging.FileHandler':
                    logfile = handler.get('filename')
                    file_handler = logging.FileHandler(logfile)
                    file_handler.setLevel(log_level)
                    file_handler.setFormatter(formatter)
                    logger.addHandler(file_handler)
                elif handler_class in ('logging.consoleHandler', 'logging.StreamHandler'):
                    ch = logging.StreamHandler()
                    ch.setLevel(log_level)
                    ch.setFormatter(formatter)
                    logger.addHandler(ch)

        return logger
