import re
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import io
import logging
import os
from zipfile import ZipFile
import urllib.request
import requests
import boto3
from botocore.errorfactory import ClientError
from botocore import UNSIGNED
from botocore.client import Config
from sdv.datasets.demo import download_demo
from sdv.datasets.errors import InvalidArgumentError
from sdv.metadata.single_table import SingleTableMetadata
from sdv.single_table.copulagan import CopulaGANSynthesizer
import pytest


def test_download_demo_invalid_modality():
    """Test it crashes when an invalid modality is passed."""
    # Run and Assert
    err_msg = re.escape("'modality' must be in ['single_table', 'multi_table', 'sequential'].")
    with pytest.raises(InvalidArgumentError, match=err_msg):
        download_demo('invalid_modality', 'dataset_name')

def test_download_demo_folder_doesnt_exist(tmpdir):
    """Test it crashes when folder ``output_folder_name`` already exist."""
    # Run and Assert
    err_msg = re.escape(
        f"Folder '{tmpdir}' already exists. Please specify a different name "
        "or use 'load_csvs' to load from an existing folder."
    )
    with pytest.raises(InvalidArgumentError, match=err_msg):
        download_demo('single_table', 'dataset_name', tmpdir)

def test_download_demo_dataset_doesnt_exist():
    """Test it crashes when ``dataset_name`` doesn't exist."""
    # Run and Assert
    err_msg = re.escape(
        f"Invalid dataset name 'invalid_dataset'. "
        "Use 'list_available_demos' to get a list of demo datasets."
    )
    with pytest.raises(InvalidArgumentError, match=err_msg):
        download_demo('single_table', 'invalid_dataset')

def test_download_demo_dataset_of_incorrect_modality():
    """Test it crashes when ``modality`` doesn't match the ``dataset_name``."""
    # Run and Assert
    err_msg = re.escape(
        f"Dataset name 'credit' is a 'single_table' dataset. "
        f"Use 'load_single_table_demo' to load this dataset."
    )
    with pytest.raises(InvalidArgumentError, match=err_msg):
        download_demo('multi_table', 'credit')

def test_download_demo_doesnt_raise():
    download_demo('single_table', 'credit')

def test_download_demo_single_table():
    """Test it can download a single table dataset."""
    # Run
    metadata, table = download_demo('single_table', 'credit')

    # Assert
    expected_table = pd.DataFrame({'0': [0, 0], '1': [0, 0]})
    pd.testing.assert_frame_equal(table.head(2), expected_table)

    expected_metadata_dict = {
        "columns": {
            "0": {
                "sdtype": "numerical", 
                "computer_representation": "Int64"
            }, 
            "1": {
                "sdtype": "numerical", 
                "computer_representation": "Int64"
            }
        }, 
        "SCHEMA_VERSION": "SINGLE_TABLE_V1"
    }
    assert metadata.to_dict() == expected_metadata_dict

