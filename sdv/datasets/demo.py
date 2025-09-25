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

from sdv.errors import DemoResourceNotFoundError, DemoResourceNotFoundWarning
from sdv.metadata.metadata import Metadata

LOGGER = logging.getLogger(__name__)
BUCKET = 'sdv-datasets-public'
BUCKET_URL = f'https://{BUCKET}.s3.amazonaws.com'
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
        contents.extend(resp.get('Contents', []))

    if not contents:
        raise DemoResourceNotFoundError(f"No objects found under '{prefix}' in bucket '{BUCKET}'.")

    return contents


def _search_contents_keys(contents, match_fn):
    """Return list of keys from ``contents`` that satisfy ``match_fn``.

    Args:
        contents (list[dict]):
            S3 list_objects-like contents entries.
        match_fn (callable):
            Function that receives a key (str) and returns True if it matches.

    Returns:
        list[str]:
            Keys in their original order that matched the predicate.
    """
    matches = []
    for entry in contents or []:
        key = entry.get('Key', '')
        try:
            if match_fn(key):
                matches.append(key)
        except Exception:
            continue

    return matches


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

    def is_data_zip(key):
        return key.lower() == f'{prefix_lower}data.zip'

    matches = _search_contents_keys(contents, is_data_zip)
    if matches:
        return matches[0]

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

    def is_direct_json_under_prefix(key):
        key_lower = key.lower()
        return (
            key_lower.startswith(prefix_lower)
            and key_lower.endswith('.json')
            and 'metadata' in key_lower
            and key_lower.count('/') == prefix_lower.count('/')
        )

    candidate_keys = _search_contents_keys(contents, is_direct_json_under_prefix)

    for key in candidate_keys:
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

        else:
            in_memory_directory = {}
            for name in zf.namelist():
                in_memory_directory[name] = zf.read(name)

            return in_memory_directory


def _get_data(modality, output_folder_name, in_memory_directory):
    data = {}
    if output_folder_name:
        for root, _dirs, files in os.walk(output_folder_name):
            for filename in files:
                if filename.endswith('.csv'):
                    table_name = Path(filename).stem
                    data_path = os.path.join(root, filename)
                    data[table_name] = pd.read_csv(data_path)

    else:
        for filename, file_ in in_memory_directory.items():
            if filename.endswith('.csv'):
                table_name = Path(filename).stem
                data[table_name] = pd.read_csv(io.StringIO(file_.decode()), low_memory=False)

    if not data:
        raise DemoResourceNotFoundError(
            'Demo data could not be downloaded because no csv files were found in data.zip'
        )

    if modality != 'multi_table':
        data = data.popitem()[1]

    return data


def _get_metadata(metadata_bytes, dataset_name, output_folder_name=None):
    """Parse metadata bytes and optionally persist to ``output_folder_name``.

    Args:
        metadata_bytes (bytes):
            Raw bytes of the metadata JSON file.
        dataset_name (str):
            The dataset name used when loading into ``Metadata``.
        output_folder_name (str or None):
            Optional folder path where to write ``metadata.json``.

    Returns:
        Metadata:
            Parsed metadata object.
    """
    try:
        metadict = json.loads(metadata_bytes)
        metadata = Metadata().load_from_dict(metadict, dataset_name)
    except Exception as e:
        raise DemoResourceNotFoundError('Failed to parse metadata JSON for the dataset.') from e

    if output_folder_name:
        try:
            metadata_path = os.path.join(str(output_folder_name), METADATA_FILENAME)
            with open(metadata_path, 'wb') as f:
                f.write(metadata_bytes)

        except Exception:
            warnings.warn(
                (
                    f'Error saving {METADATA_FILENAME} for dataset {dataset_name} into '
                    f'{output_folder_name}.',
                ),
                DemoResourceNotFoundWarning,
            )

    return metadata


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
    metadata = _get_metadata(metadata_bytes, dataset_name, output_folder_name)

    return data, metadata


def _iter_metainfo_yaml_entries(contents, modality):
    """Yield (dataset_name, yaml_key) for metainfo.yaml files under a modality.

    This matches keys like '<modality>/<dataset>/metainfo.yaml'.
    """
    modality_lower = (modality or '').lower()

    def is_metainfo_yaml(key):
        parts = key.split('/')
        if len(parts) != 3:
            return False
        if parts[0].lower() != modality_lower:
            return False
        if parts[-1].lower() != 'metainfo.yaml':
            return False
        return bool(parts[1])

    for key in _search_contents_keys(contents, is_metainfo_yaml):
        dataset_name = key.split('/')[1]
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

            size_mb_val = info.get('dataset-size-mb')
            try:
                size_mb = float(size_mb_val) if size_mb_val is not None else np.nan
            except (ValueError, TypeError):
                LOGGER.info(
                    f'Invalid dataset-size-mb {size_mb_val} for dataset '
                    f'{dataset_name}; defaulting to NaN.'
                )
                size_mb = np.nan

            num_tables_val = info.get('num-tables', np.nan)
            if isinstance(num_tables_val, str):
                try:
                    num_tables_val = float(num_tables_val)
                except (ValueError, TypeError):
                    LOGGER.info(
                        f'Could not cast num_tables_val {num_tables_val} to float for '
                        f'dataset {dataset_name}; defaulting to NaN.'
                    )
                    num_tables_val = np.nan

            try:
                num_tables = int(num_tables_val) if not pd.isna(num_tables_val) else np.nan
            except (ValueError, TypeError):
                LOGGER.info(
                    f'Invalid num-tables {num_tables_val} for '
                    f'dataset {dataset_name} when parsing as int.'
                )
                num_tables = np.nan

            tables_info['dataset_name'].append(dataset_name)
            tables_info['size_MB'].append(size_mb)
            tables_info['num_tables'].append(num_tables)

        except Exception:
            continue

    return pd.DataFrame(tables_info)


def _find_text_key(contents, dataset_prefix, filename):
    """Find a text file key (README.txt or SOURCE.txt).

    Performs a case-insensitive search for ``filename`` directly under ``dataset_prefix``.

    Args:
        contents (list[dict]):
            List of objects from S3.
        dataset_prefix (str):
            Prefix like 'single_table/dataset/'.
        filename (str):
            The filename to look for (e.g., 'README.txt').

    Returns:
        str or None:
            The key if found, otherwise ``None``.
    """
    expected_lower = f'{dataset_prefix}{filename}'.lower()
    for entry in contents:
        key = entry.get('Key') or ''
        if key.lower() == expected_lower:
            return key

    return None


def _get_text_file_content(modality, dataset_name, filename, output_filepath=None):
    """Fetch text file content under the dataset prefix.

    Args:
        modality (str):
            The modality of the dataset: ``'single_table'``, ``'multi_table'``, ``'sequential'``.
        dataset_name (str):
            The name of the dataset.
        filename (str):
            The filename to fetch (``'README.txt'`` or ``'SOURCE.txt'``).
        output_filepath (str or None):
            If provided, save the file contents at this path.

    Returns:
        str or None:
            The decoded text contents if the file exists, otherwise ``None``.
    """
    _validate_modalities(modality)
    if output_filepath is not None and not str(output_filepath).endswith('.txt'):
        fname = (filename or '').lower()
        file_type = 'README' if 'readme' in fname else 'source'
        raise ValueError(
            f'The {file_type} can only be saved as a txt file. '
            "Please provide a filepath ending in '.txt'"
        )

    dataset_prefix = f'{modality}/{dataset_name}/'
    contents = _list_objects(dataset_prefix)

    key = _find_text_key(contents, dataset_prefix, filename)
    if not key:
        if filename.upper() == 'README.TXT':
            msg = 'No README information is available for this dataset.'
        elif filename.upper() == 'SOURCE.TXT':
            msg = 'No source information is available for this dataset.'
        else:
            msg = f'No {filename} information is available for this dataset.'

        if output_filepath:
            msg = f'{msg} The requested file ({output_filepath}) will not be created.'

        warnings.warn(msg, DemoResourceNotFoundWarning)
        return None

    try:
        raw = _get_data_from_bucket(key)
    except Exception:
        LOGGER.info(f'Error fetching {filename} for dataset {dataset_name}.')
        return None

    text = raw.decode('utf-8', errors='replace')
    if output_filepath:
        if os.path.exists(str(output_filepath)):
            raise ValueError(
                f"A file named '{output_filepath}' already exists. "
                'Please specify a different filepath.'
            )
        try:
            parent = os.path.dirname(str(output_filepath))
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write(text)

        except Exception:
            LOGGER.info(f'Error saving {filename} for dataset {dataset_name}.')
            pass

    return text


def get_source(modality, dataset_name, output_filepath=None):
    """Get dataset source/citation text.

    Args:
        modality (str):
            The modality of the dataset: ``'single_table'``, ``'multi_table'``, ``'sequential'``.
        dataset_name (str):
            The name of the dataset to get the source information for.
        output_filepath (str or None):
            Optional path where to save the file.

    Returns:
        str or None:
            The contents of the source file if it exists; otherwise ``None``.
    """
    return _get_text_file_content(modality, dataset_name, 'SOURCE.txt', output_filepath)


def get_readme(modality, dataset_name, output_filepath=None):
    """Get dataset README text.

    Args:
        modality (str):
            The modality of the dataset: ``'single_table'``, ``'multi_table'``, ``'sequential'``.
        dataset_name (str):
            The name of the dataset to get the README for.
        output_filepath (str or None):
            Optional path where to save the file.

    Returns:
        str or None:
            The contents of the README file if it exists; otherwise ``None``.
    """
    return _get_text_file_content(modality, dataset_name, 'README.txt', output_filepath)
