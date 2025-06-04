"""SDV versions."""

from importlib.metadata import version

community = version('sdv')
enterprise = None

__all__ = ('community', 'enterprise')
