"""Methods to load datasets."""

import io
import json
import logging
import os
import warnings
from collections import defaultdict
from functools import wraps
from pathlib import Path
from zipfile import ZipFile

import boto3
import botocore
import numpy as np
import pandas as pd
import yaml
from botocore import UNSIGNED
from botocore.client import Config

from sdv.errors import DemoResourceNotFoundError, DemoResourceNotFoundWarning
from sdv.metadata.metadata import Metadata

LOGGER = logging.getLogger(__name__)
PUBLIC_BUCKET = 'sdv-datasets-public'
SIGNATURE_VERSION = UNSIGNED
METADATA_FILENAME = 'metadata.json'
FALLBACK_ENCODING = 'latin-1'


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


def _create_s3_client(bucket, credentials=None):
    """Create and return an S3 client with unsigned requests."""
    if bucket != PUBLIC_BUCKET:
        raise ValueError('Private buckets are only supported in SDV Enterprise.')
    if credentials is not None:
        raise ValueError(
            'DataCebo credentials for private buckets are only supported in SDV Enterprise.'
        )

    return boto3.client('s3', config=Config(signature_version=SIGNATURE_VERSION))


def _get_data_from_bucket(object_key, bucket, client):
    response = client.get_object(Bucket=bucket, Key=object_key)
    return response['Body'].read()


def _get_dataset_name_from_prefix(prefix):
    return prefix.split('/')[1]


def _list_objects(prefix, bucket, client):
    """List all objects under a given prefix using pagination.

    Args:
        prefix (str):
            The S3 prefix to list.
        bucket (str):
            The name of the bucket to list objects of.
        client (botocore.client.S3):
            S3 client.

    Returns:
        list[dict]:
            A list of object summaries.
    """
    contents = []
    paginator = client.get_paginator('list_objects_v2')
    for resp in paginator.paginate(Bucket=bucket, Prefix=prefix):
        contents.extend(resp.get('Contents', []))

    if not contents:
        prefix_parts = prefix.split('/')
        modality = prefix_parts[0]
        dataset_name = _get_dataset_name_from_prefix(prefix)
        if dataset_name:
            raise DemoResourceNotFoundError(
                f"Could not download dataset '{dataset_name}' from bucket '{bucket}'. "
                'Make sure the bucket name is correct. If the bucket is private '
                'make sure to provide your credentials.'
            )
        else:
            raise DemoResourceNotFoundError(
                f"Could not list datasets in modality '{modality}' from bucket '{bucket}'. "
                'Make sure the bucket name is correct. If the bucket is private '
                'make sure to provide your credentials.'
            )

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


def _find_data_zip_key(contents, dataset_prefix, bucket):
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

    dataset_name = _get_dataset_name_from_prefix(dataset_prefix)
    raise DemoResourceNotFoundError(
        f"Could not download dataset '{dataset_name}' from bucket '{bucket}'. "
        "The dataset is missing 'data.zip' file."
    )


def _get_first_v1_metadata_bytes(contents, dataset_prefix, bucket, client):
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
            raw = _get_data_from_bucket(key, bucket=bucket, client=client)
            metadict = json.loads(raw)
            if isinstance(metadict, dict) and metadict.get('METADATA_SPEC_VERSION') == 'V1':
                return raw

        except Exception:
            continue

    dataset_name = _get_dataset_name_from_prefix(dataset_prefix)
    raise DemoResourceNotFoundError(
        f"Could not download dataset '{dataset_name}' from bucket '{bucket}'. "
        'The dataset is missing a valid metadata.'
    )


def _download_text_file_error_message(
    modality,
    dataset_name,
    output_filepath=None,
    bucket=PUBLIC_BUCKET,
    filename=None,
    **kwargs,
):
    return (
        f"Could not retrieve '{filename}' for dataset '{dataset_name}' "
        f"from bucket '{bucket}'. "
        'Make sure the bucket name is correct. If the bucket is private '
        'make sure to provide your credentials.'
    )


def _download_error_message(
    modality,
    dataset_name,
    output_folder_name=None,
    s3_bucket_name=PUBLIC_BUCKET,
    credentials=None,
    **kwargs,
):
    return (
        f"Could not download dataset '{dataset_name}' from bucket '{s3_bucket_name}'. "
        'Make sure the bucket name is correct. If the bucket is private '
        'make sure to provide your credentials.'
    )


def _list_modality_error_message(modality, s3_bucket_name, **kwargs):
    return (
        f"Could not list datasets in modality '{modality}' from bucket '{s3_bucket_name}'. "
        'Make sure the bucket name is correct. If the bucket is private '
        'make sure to provide your credentials.'
    )


def handle_aws_client_errors(error_message_builder):
    """Decorate a function to translate AWS client errors into more descriptive errors.

    This decorator catches ``botocore.exceptions.ClientError`` raised by the wrapped
    function and re-raises it as a ``DemoResourceNotFoundError`` with a custom error
    message. The error message is generated dynamically using the provided
    ``error_message_builder`` function.

    Args:
        error_message_builder (Callable):
            A callable that receives the same ``*args`` and ``**kwargs`` as the wrapped
            function and returns an error message.

    Returns:
        func:
            A wrapped function.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                function_result = func(*args, **kwargs)
            except botocore.exceptions.ClientError as error:
                message = error_message_builder(*args, **kwargs)
                raise DemoResourceNotFoundError(message) from error

            return function_result

        return wrapper

    return decorator


def _download(modality, dataset_name, bucket, credentials=None):
    """Download dataset resources from a bucket.

    Returns:
        tuple:
            (BytesIO(zip_bytes), metadata_bytes)
    """
    client = _create_s3_client(bucket=bucket, credentials=credentials)
    dataset_prefix = f'{modality}/{dataset_name}/'
    bucket_url = f'https://{bucket}.s3.amazonaws.com'
    LOGGER.info(
        f"Downloading dataset '{dataset_name}' for modality '{modality}' from "
        f'{bucket_url}/{dataset_prefix}'
    )
    contents = _list_objects(dataset_prefix, bucket=bucket, client=client)
    zip_key = _find_data_zip_key(contents, dataset_prefix, bucket)
    zip_bytes = _get_data_from_bucket(zip_key, bucket=bucket, client=client)
    metadata_bytes = _get_first_v1_metadata_bytes(
        contents, dataset_prefix, bucket=bucket, client=client
    )

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


def _get_data_with_output_folder(output_folder_name):
    """Load CSV tables from an extracted folder on disk.

    Returns a tuple of (data_dict, skipped_files).
    Non-CSV files are ignored.
    """
    data = {}
    skipped_files = []
    for root, _dirs, files in os.walk(output_folder_name):
        for filename in files:
            if not filename.lower().endswith('.csv'):
                skipped_files.append(filename)
                continue

            table_name = Path(filename).stem
            data_path = os.path.join(root, filename)
            try:
                data[table_name] = pd.read_csv(data_path)
            except UnicodeDecodeError:
                data[table_name] = pd.read_csv(data_path, encoding=FALLBACK_ENCODING)
            except Exception as e:
                rel = os.path.relpath(data_path, output_folder_name)
                skipped_files.append(f'{rel}: {e}')

    return data, skipped_files


def _get_data_without_output_folder(in_memory_directory):
    """Load CSV tables directly from in-memory zip contents.

    Returns a tuple of (data_dict, skipped_files).
    Non-CSV entries are ignored.
    """
    data = {}
    skipped_files = []
    for filename, file_ in in_memory_directory.items():
        if not filename.lower().endswith('.csv'):
            skipped_files.append(filename)
            continue

        table_name = Path(filename).stem
        try:
            data[table_name] = pd.read_csv(io.BytesIO(file_), low_memory=False)
        except UnicodeDecodeError:
            data[table_name] = pd.read_csv(
                io.BytesIO(file_), low_memory=False, encoding=FALLBACK_ENCODING
            )
        except Exception as e:
            skipped_files.append(f'{filename}: {e}')

    return data, skipped_files


def _get_data(modality, output_folder_name, in_memory_directory, bucket, dataset_name):
    if output_folder_name:
        data, skipped_files = _get_data_with_output_folder(output_folder_name)
    else:
        data, skipped_files = _get_data_without_output_folder(in_memory_directory)

    if skipped_files:
        warnings.warn('Skipped files: ' + ', '.join(sorted(skipped_files)))

    if not data:
        raise DemoResourceNotFoundError(
            f"Could not download dataset '{dataset_name}' from bucket '{bucket}'. "
            'The dataset is missing `csv` file/s.'
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
        raise DemoResourceNotFoundError(
            f"Could not parse the metadata for dataset '{dataset_name}'. "
            'The dataset is missing a valid metadata file.'
        ) from e

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


@handle_aws_client_errors(_download_error_message)
def download_demo(
    modality, dataset_name, output_folder_name=None, s3_bucket_name=PUBLIC_BUCKET, credentials=None
):
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
        s3_bucket_name (str):
            The name of the bucket to download from. Only 'sdv-datasets-public' is supported in
            SDV Community. SDV Enterprise is required for other buckets.
        credentials (dict):
            Dictionary containing DataCebo license key and username. It takes the form:
            {
                'username': 'example@datacebo.com',
                'license_key': '<MY_LICENSE_KEY>'
            }

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

    data_io, metadata_bytes = _download(modality, dataset_name, s3_bucket_name, credentials)
    in_memory_directory = _extract_data(data_io, output_folder_name)
    data = _get_data(
        modality,
        output_folder_name,
        in_memory_directory,
        s3_bucket_name,
        dataset_name,
    )
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


def _get_info_from_yaml_key(yaml_key, bucket, client):
    """Load and parse YAML metadata from an S3 key."""
    raw = _get_data_from_bucket(yaml_key, bucket=bucket, client=client)
    return yaml.safe_load(raw) or {}


def _parse_size_mb(size_mb_val, dataset_name):
    """Parse the size (MB) value into a float or NaN with logging on failures."""
    try:
        return float(size_mb_val) if size_mb_val is not None else np.nan
    except (ValueError, TypeError):
        LOGGER.info(
            f'Invalid dataset-size-mb {size_mb_val} for dataset {dataset_name}; defaulting to NaN.'
        )
        return np.nan


def _parse_num_tables(num_tables_val, dataset_name):
    """Parse the num-tables value into an int or NaN with logging on failures."""
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
        return int(num_tables_val) if not pd.isna(num_tables_val) else np.nan
    except (ValueError, TypeError):
        LOGGER.info(
            f'Invalid num-tables {num_tables_val} for dataset {dataset_name} when parsing as int.'
        )
        return np.nan


@handle_aws_client_errors(_list_modality_error_message)
def get_available_demos(modality, s3_bucket_name=PUBLIC_BUCKET, credentials=None):
    """Get demo datasets available for a ``modality``.

    Args:
        modality (str):
            The modality of the dataset: ``'single_table'``, ``'multi_table'``, ``'sequential'``.
        s3_bucket_name (str):
            The name of the bucket to download from. Only 'sdv-datasets-public' is supported in
            SDV Community. SDV Enterprise is required for other buckets.
        credentials (dict):
            Dictionary containing DataCebo license key and username. It takes the form:
            {
                'username': 'example@datacebo.com',
                'license_key': '<MY_LICENSE_KEY>'
            }

    Returns:
        pandas.DataFrame:
            A table with three columns:
                ``dataset_name``: The name of the dataset.
                ``size_MB``: The unzipped folder size in MB.
                ``num_tables``: The number of tables in the dataset.
    """
    _validate_modalities(modality)
    s3_client = _create_s3_client(bucket=s3_bucket_name, credentials=credentials)
    contents = _list_objects(f'{modality}/', bucket=s3_bucket_name, client=s3_client)
    tables_info = defaultdict(list)
    for dataset_name, yaml_key in _iter_metainfo_yaml_entries(contents, modality):
        try:
            info = _get_info_from_yaml_key(yaml_key, bucket=s3_bucket_name, client=s3_client)

            size_mb = _parse_size_mb(info.get('dataset-size-mb'), dataset_name)
            num_tables = _parse_num_tables(info.get('num-tables', np.nan), dataset_name)

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


def _validate_text_file_content(modality, output_filepath, filename):
    """Validation for the text file content method."""
    _validate_modalities(modality)
    if output_filepath is not None and not str(output_filepath).endswith('.txt'):
        fname = (filename or '').lower()
        file_type = 'README' if 'readme' in fname else 'source'
        raise ValueError(
            f'The {file_type} can only be saved as a txt file. '
            "Please provide a filepath ending in '.txt'"
        )


def _raise_warnings(filename, output_filepath):
    """Warn about missing text resources for a dataset."""
    if (filename or '').upper() == 'README.TXT':
        msg = 'No README information is available for this dataset.'
    elif (filename or '').upper() == 'SOURCE.TXT':
        msg = 'No source information is available for this dataset.'
    else:
        msg = f'No {filename} information is available for this dataset.'

    if output_filepath:
        msg = f'{msg} The requested file ({output_filepath}) will not be created.'

    warnings.warn(msg, DemoResourceNotFoundWarning)


def _save_document(text, output_filepath, filename, dataset_name):
    """Persist ``text`` to ``output_filepath`` if provided."""
    if not output_filepath:
        return

    if os.path.exists(str(output_filepath)):
        raise ValueError(
            f"A file named '{output_filepath}' already exists. Please specify a different filepath."
        )

    try:
        parent = os.path.dirname(str(output_filepath))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(text)
    except Exception:
        LOGGER.info(f'Error saving {filename} for dataset {dataset_name}.')


@handle_aws_client_errors(_download_text_file_error_message)
def _get_text_file_content(
    modality, dataset_name, filename, output_filepath=None, bucket=PUBLIC_BUCKET, credentials=None
):
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
        bucket (str):
            The name of the bucket to download from. Only 'sdv-datasets-public' is supported in
            SDV Community. SDV Enterprise is required for other buckets.
        credentials (dict):
            Dictionary containing DataCebo license key and username. It takes the form:
            {
                'username': 'example@datacebo.com',
                'license_key': '<MY_LICENSE_KEY>'
            }

    Returns:
        str or None:
            The decoded text contents if the file exists, otherwise ``None``.
    """
    _validate_text_file_content(modality, output_filepath, filename)

    dataset_prefix = f'{modality}/{dataset_name}/'
    s3_client = _create_s3_client(bucket=bucket, credentials=credentials)
    contents = _list_objects(dataset_prefix, bucket=bucket, client=s3_client)
    key = _find_text_key(contents, dataset_prefix, filename)
    if not key:
        _raise_warnings(filename, output_filepath)
        return None

    try:
        raw = _get_data_from_bucket(key, bucket=bucket, client=s3_client)
    except Exception:
        LOGGER.info(f'Error fetching {filename} for dataset {dataset_name}.')
        return None

    text = raw.decode('utf-8', errors='replace')
    _save_document(text, output_filepath, filename, dataset_name)

    return text


@handle_aws_client_errors(_download_text_file_error_message)
def get_source(
    modality, dataset_name, output_filepath=None, s3_bucket_name=PUBLIC_BUCKET, credentials=None
):
    """Get dataset source/citation text.

    Args:
        modality (str):
            The modality of the dataset: ``'single_table'``, ``'multi_table'``, ``'sequential'``.
        dataset_name (str):
            The name of the dataset to get the source information for.
        output_filepath (str or None):
            Optional path where to save the file.
        s3_bucket_name (str):
            The name of the bucket to download from. Only 'sdv-datasets-public' is supported in
            SDV Community. SDV Enterprise is required for other buckets.
        credentials (dict):
            Dictionary containing DataCebo license key and username. It takes the form:
            {
                'username': 'example@datacebo.com',
                'license_key': '<MY_LICENSE_KEY>'
            }

    Returns:
        str or None:
            The contents of the source file if it exists; otherwise ``None``.
    """
    return _get_text_file_content(
        modality=modality,
        dataset_name=dataset_name,
        filename='SOURCE.txt',
        output_filepath=output_filepath,
        bucket=s3_bucket_name,
        credentials=credentials,
    )


def get_readme(
    modality, dataset_name, output_filepath=None, s3_bucket_name=PUBLIC_BUCKET, credentials=None
):
    """Get dataset README text.

    Args:
        modality (str):
            The modality of the dataset: ``'single_table'``, ``'multi_table'``, ``'sequential'``.
        dataset_name (str):
            The name of the dataset to get the README for.
        output_filepath (str or None):
            Optional path where to save the file.
        s3_bucket_name (str):
            The name of the bucket to download from. Only 'sdv-datasets-public' is supported in
            SDV Community. SDV Enterprise is required for other buckets.
        credentials (dict):
            Dictionary containing DataCebo license key and username. It takes the form:
            {
                'username': 'example@datacebo.com',
                'license_key': '<MY_LICENSE_KEY>'
            }

    Returns:
        str or None:
            The contents of the README file if it exists; otherwise ``None``.
    """
    return _get_text_file_content(
        modality=modality,
        dataset_name=dataset_name,
        filename='README.txt',
        output_filepath=output_filepath,
        bucket=s3_bucket_name,
        credentials=credentials,
    )
