"""Column relationship validation functions."""
from sdv.metadata.errors import InvalidMetadataError


def validate_address_sdtypes(column_metadata, column_names):
    """Validate sdtypes for address column relationship.

    Args:
        column_metadata (dict):
            Column metadata for the table.
        column_names (list[str]):
            List of the column names involved in this relationship.

    Raises:
        - ``InvalidMetadataError`` if column sdtypes are invalid for the relationship.
    """
    valid_sdtypes = (
        'country_code', 'administrative_unit', 'city', 'postcode', 'street_address',
        'secondary_address', 'state', 'state_abbr'
    )
    bad_columns = []
    for column_name in column_names:
        if column_name not in column_metadata:
            continue
        if column_metadata[column_name].get('sdtype') not in valid_sdtypes:
            bad_columns.append(column_name)

    if bad_columns:
        raise InvalidMetadataError(
            f'Columns {bad_columns} have unsupported sdtypes for column relationship '
            "type 'address'."
        )
