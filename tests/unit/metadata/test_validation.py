"""Test Single Table Metadata."""

import re

import pytest

from sdv.metadata.errors import InvalidMetadataError
from sdv.metadata.validation import validate_address_sdtypes


def test_validate_address_sdtypes():
    """Test address sdtype validation."""
    # Setup
    column_metadata = {
        'col_1': {'sdtype': 'id'},
        'col_2': {'sdtype': 'numerical'},
        'col_3': {'sdtype': 'state'}
    }

    # Run and Assert
    err_msg = re.escape(
        "Columns ['col_1', 'col_2'] have unsupported sdtypes for column "
        "relationship type 'address'"
    )
    with pytest.raises(InvalidMetadataError, match=err_msg):
        validate_address_sdtypes(column_metadata, ['col_1', 'col_2', 'col_3', 'col_4'])
