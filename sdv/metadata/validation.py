"""Column relationship validation functions."""
import rdt
from rdt.errors import TransformerInputError

from sdv.metadata.errors import InvalidMetadataError


def _check_import_address_transformers():
    """Check that the address transformers can be imported."""
    error_message = (
        'You must have SDV Enterprise with the address add-on to use the address features'
    )
    if not hasattr(rdt.transformers, 'address'):
        raise ImportError(error_message)

    has_randomlocationgenerator = hasattr(rdt.transformers.address, 'RandomLocationGenerator')
    has_regionalanonymizer = hasattr(rdt.transformers.address, 'RegionalAnonymizer')
    if not has_randomlocationgenerator or not has_regionalanonymizer:
        raise ImportError(error_message)


def validate_address_sdtypes(columns_to_sdtypes):
    """Validate sdtypes for address column relationship.

    Args:
        columns_to_sdtypes (dict):
            Dictionary mapping column names to sdtypes.

    Raises:
        ``InvalidMetadataError`` if column sdtypes are invalid for the relationship.
    """
    _check_import_address_transformers()
    try:
        rdt.transformers.address.RandomLocationGenerator._validate_sdtypes(columns_to_sdtypes)
    except TransformerInputError as error:
        raise InvalidMetadataError(str(error))


def _check_import_gps_transformers():
    """Check that the gps transformers can be imported."""
    error_message = (
        'You must have SDV Enterprise with the gps add-on to use the gps features'
    )
    if not hasattr(rdt.transformers, 'gps'):
        raise ImportError(error_message)

    has_randomlocationgenerator = hasattr(rdt.transformers.gps, 'RandomLocationGenerator')
    has_metroareaanonymizer = hasattr(rdt.transformers.gps, 'MetroAreaAnonymizer')
    has_gpsnoiser = hasattr(rdt.transformers.gps, 'GPSNoiser')
    if not has_randomlocationgenerator or not has_metroareaanonymizer or not has_gpsnoiser:
        raise ImportError(error_message)


def validate_gps_sdtypes(columns_to_sdtypes):
    """Validate sdtypes for gps column relationship.

    Args:
        columns_to_sdtypes (dict):
            Dictionary mapping column names to sdtypes.

    Raises:
        ``InvalidMetadataError`` if column sdtypes are invalid for the relationship.
    """
    _check_import_gps_transformers()
    try:
        rdt.transformers.gps.RandomLocationGenerator._validate_sdtypes(columns_to_sdtypes)
    except TransformerInputError as error:
        raise InvalidMetadataError(str(error))
