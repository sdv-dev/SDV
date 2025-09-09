"""Methods to load datasets."""

import io
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import boto3
import numpy as np
import pandas as pd
import yaml
from botocore import UNSIGNED
from botocore.client import Config

from sdv.errors import DemoResourceNotFoundError
from sdv.metadata.metadata import Metadata

LOGGER = logging.getLogger(__name__)
BUCKET = 'sdv-datasets-public'
BUCKET_URL = 'https://sdv-datasets-public.s3.amazonaws.com'
SIGNATURE_VERSION = UNSIGNED
METADATA_FILENAME = 'metadata.json'


def _validate_modalities(modality):
    possible_modalities = ['single_table', 'multi_table', 'sequential']
    if modality not in possible_modalities:
        raise ValueError(f"'modality' must be in {possible_modalities}.")


def _validate_output_folder(output_folder_name):
    if output_folder_name and os.path.exists(output_folder_name):
        raise ValueError(
            f"Folder '{output_folder_name}' already exists. Please specify a different name "
            "or use 'load_csvs' to load from an existing folder."
        )


def _create_s3_client():
    """Create and return an S3 client with unsigned requests."""
    return boto3.client('s3', config=Config(signature_version=SIGNATURE_VERSION))


def _get_data_from_bucket(object_key):
    s3 = _create_s3_client()
    response = s3.get_object(Bucket=BUCKET, Key=object_key)
    return response['Body'].read()


def _list_objects(prefix):
    """List all objects under a given prefix using pagination.

    Args:
        prefix (str):
            The S3 prefix to list.

    Returns:
        list[dict]:
            A list of object summaries.
    """
    client = _create_s3_client()
    contents = []
    paginator = client.get_paginator('list_objects_v2')
    for resp in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        contents.extend(resp.get('Contents') or [])

    if not contents:
        raise DemoResourceNotFoundError(f"No objects found under '{prefix}' in bucket '{BUCKET}'.")

    return contents


def _find_data_zip_key(contents, dataset_prefix):
    """Find the 'data.zip' object key under dataset prefix, case-insensitive.

    Args:
        contents (list[dict]):
            List of objects from S3.
        dataset_prefix (str):
            Prefix like 'single_table/dataset/'.

    Returns:
        str:
            The key to the data zip if found.
    """
    prefix_lower = dataset_prefix.lower()
    for entry in contents:
        key = entry.get('Key') or ''
        key_lower = key.lower()
        if key_lower.startswith(prefix_lower) and key_lower.endswith('/data.zip'):
            return key

    raise DemoResourceNotFoundError("Could not find 'data.zip' for the requested dataset.")


def _get_first_v1_metadata_bytes(contents, dataset_prefix):
    """Find and return bytes of the first V1 metadata JSON under `dataset_prefix`.

    Scans S3 listing `contents` and, for any JSON file directly under the dataset prefix,
    downloads and returns its bytes if it contains METADATA_SPEC_VERSION == 'V1'.

    Returns:
        bytes:
            The bytes of the first V1 metadata JSON.
    """
    prefix_lower = dataset_prefix.lower()
    for entry in contents:
        key = entry.get('Key') or ''
        key_lower = key.lower()
        if not (key_lower.startswith(prefix_lower) and key_lower.endswith('.json')):
            continue

        try:
            raw = _get_data_from_bucket(key)
            metadict = json.loads(raw)
            if isinstance(metadict, dict) and metadict.get('METADATA_SPEC_VERSION') == 'V1':
                return raw

        except Exception:
            continue

    raise DemoResourceNotFoundError(
        'Could not find a valid metadata JSON with METADATA_SPEC_VERSION "V1".'
    )


def _download(modality, dataset_name):
    """Download dataset resources from a bucket.

    Returns:
        tuple:
            (BytesIO(zip_bytes), metadata_bytes)
    """
    dataset_prefix = f'{modality}/{dataset_name}/'
    LOGGER.info(
        f"Downloading dataset '{dataset_name}' for modality '{modality}' from "
        f'{BUCKET_URL}/{dataset_prefix}'
    )
    contents = _list_objects(dataset_prefix)

    zip_key = _find_data_zip_key(contents, dataset_prefix)
    zip_bytes = _get_data_from_bucket(zip_key)
    metadata_bytes = _get_first_v1_metadata_bytes(contents, dataset_prefix)

    return io.BytesIO(zip_bytes), metadata_bytes


def _extract_data(bytes_io, output_folder_name):
    with ZipFile(bytes_io) as zf:
        if output_folder_name:
            os.makedirs(output_folder_name, exist_ok=True)
            zf.extractall(output_folder_name)
            metadata_v0_filepath = os.path.join(output_folder_name, 'metadata_v0.json')
            if os.path.isfile(metadata_v0_filepath):
                os.remove(metadata_v0_filepath)

            metadata_v1_filepath = os.path.join(output_folder_name, 'metadata_v1.json')
            if os.path.isfile(metadata_v1_filepath):
                os.rename(
                    metadata_v1_filepath,
                    os.path.join(output_folder_name, METADATA_FILENAME),
                )

        else:
            in_memory_directory = {}
            for name in zf.namelist():
                in_memory_directory[name] = zf.read(name)

            return in_memory_directory


def _get_data(modality, output_folder_name, in_memory_directory):
    data = {}
    if output_folder_name:
        for filename in os.listdir(output_folder_name):
            if filename.endswith('.csv'):
                table_name = Path(filename).stem
                data_path = os.path.join(output_folder_name, filename)
                data[table_name] = pd.read_csv(data_path)

    else:
        for filename, file_ in in_memory_directory.items():
            if filename.endswith('.csv'):
                table_name = Path(filename).stem
                data[table_name] = pd.read_csv(io.StringIO(file_.decode()), low_memory=False)

    if modality != 'multi_table':
        data = data.popitem()[1]

    return data


def download_demo(modality, dataset_name, output_folder_name=None):
    """Download a demo dataset.

    Args:
        modality (str):
            The modality of the dataset: ``'single_table'``, ``'multi_table'``, ``'sequential'``.
        dataset_name (str):
            Name of the dataset to be downloaded from the sdv-datasets-public S3 bucket.
        output_folder_name (str or None):
            The name of the local folder where the metadata and data should be stored.
            If ``None`` the data is not saved locally and is loaded as a Python object.
            Defaults to ``None``.

    Returns:
        tuple (data, metadata):
            If ``data`` is single table or sequential, it is a DataFrame.
            If ``data`` is multi table, it is a dictionary mapping table name to DataFrame.
            ``metadata`` is of class ``Metadata`` which can represent single table or multi table.

    Raises:
        Error:
            * If the ``dataset_name`` exists in the bucket but under a different modality.
            * If the ``dataset_name`` doesn't exist in the bucket.
            * If there is already a folder named ``output_folder_name``.
    """
    _validate_modalities(modality)
    _validate_output_folder(output_folder_name)
    data_io, metadata_bytes = _download(modality, dataset_name)
    in_memory_directory = _extract_data(data_io, output_folder_name)
    data = _get_data(modality, output_folder_name, in_memory_directory)

    try:
        metadict = json.loads(metadata_bytes)
        metadata = Metadata().load_from_dict(metadict, dataset_name)
    except Exception as e:
        raise DemoResourceNotFoundError('Failed to parse metadata JSON for the dataset.') from e

    return data, metadata


def _iter_metainfo_yaml_entries(contents, modality):
    """Yield (dataset_name, yaml_key) for metainfo.yaml files under a modality.

    This matches keys like '<modality>/<dataset>/metainfo.yaml'.
    """
    modality_lower = (modality or '').lower()
    for entry in contents or []:
        key = entry.get('Key') or ''
        parts = key.split('/')
        if len(parts) < 3:
            continue
        if parts[0].lower() != modality_lower:
            continue
        filename = parts[-1]
        if filename.lower() != 'metainfo.yaml':
            continue
        dataset_name = parts[1]
        if not dataset_name:
            continue

        yield dataset_name, key


def get_available_demos(modality):
    """Get demo datasets available for a ``modality``.

    Args:
        modality (str):
            The modality of the dataset: ``'single_table'``, ``'multi_table'``, ``'sequential'``.

    Returns:
        pandas.DataFrame:
            A table with three columns:
                ``dataset_name``: The name of the dataset.
                ``size_MB``: The unzipped folder size in MB.
                ``num_tables``: The number of tables in the dataset.
    """
    _validate_modalities(modality)
    contents = _list_objects(f'{modality}/')
    tables_info = defaultdict(list)
    for dataset_name, yaml_key in _iter_metainfo_yaml_entries(contents, modality):
        try:
            raw = _get_data_from_bucket(yaml_key)
            info = yaml.safe_load(raw) or {}
            name = info.get('dataset-name') or dataset_name

            size_mb_val = info.get('dataset-size-mb')
            try:
                size_mb = float(size_mb_val) if size_mb_val is not None else np.nan
            except Exception:
                size_mb = np.nan

            num_tables_val = info.get('num-tables')
            try:
                num_tables = int(num_tables_val)
            except Exception:
                try:
                    num_tables = int(float(num_tables_val))
                except Exception:
                    num_tables = np.nan

            tables_info['dataset_name'].append(name)
            tables_info['size_MB'].append(size_mb)
            tables_info['num_tables'].append(num_tables)

        except Exception:
            continue

    df = pd.DataFrame(tables_info)
    if not df.empty:
        df['num_tables'] = pd.to_numeric(df['num_tables'], errors='coerce')
        df['size_MB'] = pd.to_numeric(df['size_MB'], errors='coerce')

    return df
