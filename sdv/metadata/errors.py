"""Metadata Exceptions."""


class MetadataError(Exception):
    """Error to raise when Metadata is not valid."""


class MetadataNotFittedError(MetadataError):
    """Error to raise when Metadata is used before fitting."""
