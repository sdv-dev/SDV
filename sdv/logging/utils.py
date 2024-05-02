"""Utilities for configuring logging within the SDV library."""

import contextlib
import logging
from pathlib import Path

import platformdirs
import yaml


def get_sdv_logger_config():
    """Return a dictionary with the logging configuration."""
    store_path = Path(platformdirs.user_data_dir('sdv', 'sdv-dev'))
    store_path.mkdir(parents=True, exist_ok=True)
    config_path = Path(__file__).parent / 'sdv_logger_config.yml'

    if (store_path / 'sdv_logger_config.yml').exists():
        config_path = store_path / 'sdv_logger_config.yml'

    with open(config_path, 'r') as f:
        logger_conf = yaml.safe_load(f)

    for logger in logger_conf.get('loggers', {}).values():
        handler = logger.get('handlers', {})
        if handler.get('filename') == 'sdv_logs.log':
            handler['filename'] = store_path / handler['filename']

    return logger_conf


@contextlib.contextmanager
def disable_single_table_logger():
    """Temporarily disables logging for the single table synthesizers.

    This context manager temporarily removes all handlers associated with
    the ``SingleTableSynthesizer`` logger, disabling logging for that module
    within the current context. After the context exits, the
    removed handlers are restored to the logger.
    """
    # Logging without ``SingleTableSynthesizer``
    single_table_logger = logging.getLogger('SingleTableSynthesizer')
    handlers = single_table_logger.handlers
    single_table_logger.handlers = []
    try:
        yield
    finally:
        for handler in handlers:
            single_table_logger.addHandler(handler)
