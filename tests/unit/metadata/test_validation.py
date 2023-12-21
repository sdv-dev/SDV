"""Test Single Table Metadata."""
import re
from unittest.mock import patch

import pytest
from rdt.errors import TransformerInputError

from sdv.metadata.errors import InvalidMetadataError
from sdv.metadata.validation import validate_address_sdtypes


@patch('sdv.metadata.validation._check_import_address_transformers')
@patch('rdt.transformers')
def test_validate_address_sdtypes(mock_transformers, mock_check_import):
    """Test address sdtype validation."""
    # Setup
    columns_to_sdtypes = {
        'col_1': {'sdtype': 'id'},
        'col_2': {'sdtype': 'numerical'},
        'col_3': {'sdtype': 'state'}
    }
    mock_validate_sdtypes = mock_transformers.address.RandomLocationGenerator._validate_sdtypes

    # Run
    validate_address_sdtypes(columns_to_sdtypes)

    # Asserts
    mock_check_import.assert_called_once()
    mock_validate_sdtypes.assert_called_once_with(columns_to_sdtypes)


@patch('sdv.metadata.validation._check_import_address_transformers')
@patch('rdt.transformers')
def test_validate_address_sdtypes_error(mock_transformers, mock_check_import):
    """Test address sdtype validation."""
    # Setup
    columns_to_sdtypes = {
        'col_1': {'sdtype': 'id'},
        'col_2': {'sdtype': 'numerical'},
        'col_3': {'sdtype': 'state'}
    }
    mock_validate_sdtypes = mock_transformers.address.RandomLocationGenerator._validate_sdtypes
    mock_validate_sdtypes.side_effect = TransformerInputError('Error')

    # Run and Assert
    expected_message = re.escape('Error')
    with pytest.raises(InvalidMetadataError, match=expected_message):
        validate_address_sdtypes(columns_to_sdtypes)
