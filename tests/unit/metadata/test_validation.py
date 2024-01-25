"""Test Single Table Metadata."""
import re
from unittest.mock import Mock, patch

import pytest
from rdt.errors import TransformerInputError

from sdv.metadata.errors import InvalidMetadataError
from sdv.metadata.validation import (
    _check_import_address_transformers, _check_import_gps_transformers, validate_address_sdtypes,
    validate_gps_sdtypes)


def test__check_import_address_transformers_without_address_module():
    """Test ``_check_import_address_transformers`` when address module doesn't exist."""
    # Run and Assert
    expected_message = (
        'You must have SDV Enterprise with the address add-on to use the address features'
    )
    with pytest.raises(ImportError, match=expected_message):
        _check_import_address_transformers()


@patch('rdt.transformers')
def test__check_import_address_transformers_without_premium_features(mock_transformers):
    """Test ``_check_import_address_transformers`` when the user doesn't have the transformers."""
    # Setup
    mock_address = Mock()
    del mock_address.RandomLocationGenerator
    del mock_address.RegionalAnonymizer
    mock_transformers.address = mock_address

    # Run and Assert
    expected_message = (
        'You must have SDV Enterprise with the address add-on to use the address features'
    )
    with pytest.raises(ImportError, match=expected_message):
        _check_import_address_transformers()


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


def test__check_import_gps_transformers_without_gps_module():
    """Test ``_check_import_gps_transformers`` when gps module doesn't exist."""
    # Run and Assert
    expected_message = (
        'You must have SDV Enterprise with the gps add-on to use the gps features'
    )
    with pytest.raises(ImportError, match=expected_message):
        _check_import_gps_transformers()


@patch('rdt.transformers')
def test__check_import_gps_transformers_without_premium_features(mock_transformers):
    """Test ``_check_import_gps_transformers`` when the user doesn't have the transformers."""
    # Setup
    mock_gps = Mock()
    del mock_gps.RandomLocationGenerator
    del mock_gps.MetroAreaAnonymizer
    del mock_gps.GPSNoiser
    mock_transformers.gps = mock_gps

    # Run and Assert
    expected_message = (
        'You must have SDV Enterprise with the gps add-on to use the gps features'
    )
    with pytest.raises(ImportError, match=expected_message):
        _check_import_gps_transformers()


@patch('sdv.metadata.validation._check_import_gps_transformers')
@patch('rdt.transformers')
def test_validate_gps_sdtypes(mock_transformers, mock_check_import):
    """Test gps sdtype validation."""
    # Setup
    columns_to_sdtypes = {
        'col_1': {'sdtype': 'id'},
        'col_2': {'sdtype': 'numerical'},
        'col_3': {'sdtype': 'city'}
    }
    mock_validate_sdtypes = mock_transformers.gps.RandomLocationGenerator._validate_sdtypes

    # Run
    validate_gps_sdtypes(columns_to_sdtypes)

    # Asserts
    mock_check_import.assert_called_once()
    mock_validate_sdtypes.assert_called_once_with(columns_to_sdtypes)


@patch('sdv.metadata.validation._check_import_gps_transformers')
@patch('rdt.transformers')
def test_validate_gps_sdtypes_error(mock_transformers, mock_check_import):
    """Test gps sdtype validation."""
    # Setup
    columns_to_sdtypes = {
        'col_1': {'sdtype': 'id'},
        'col_2': {'sdtype': 'numerical'},
        'col_3': {'sdtype': 'city'}
    }
    mock_validate_sdtypes = mock_transformers.gps.RandomLocationGenerator._validate_sdtypes
    mock_validate_sdtypes.side_effect = TransformerInputError('Error')

    # Run
    expected_message = re.escape('Error')
    with pytest.raises(InvalidMetadataError, match=expected_message):
        validate_gps_sdtypes(columns_to_sdtypes)

    # Asserts
    mock_check_import.assert_called_once()
