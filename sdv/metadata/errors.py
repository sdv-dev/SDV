"""Metadata Exceptions."""


class InvalidMetadataError(Exception):
    """Error to raise when Metadata is not valid."""


class MetadataNotFittedError(InvalidMetadataError):
    """Error to raise when Metadata is used before fitting."""
