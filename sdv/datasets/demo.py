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
from botocore.exceptions import ClientError

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
    session = boto3.Session()
    return session.client('s3', config=Config(signature_version=SIGNATURE_VERSION))


def _get_data_from_bucket(object_key):
    s3 = _create_s3_client()
    response = s3.get_object(Bucket=BUCKET, Key=object_key)
    return response['Body'].read()


def _list_objects(prefix):
    s3 = _create_s3_client()
    contents = []
    continuation_token = None
    while True:
        kwargs = {'Bucket': BUCKET, 'Prefix': prefix}
        if continuation_token:
            kwargs['ContinuationToken'] = continuation_token
        response = s3.list_objects_v2(**kwargs)
        contents.extend(response.get('Contents', []) or [])
        if response.get('IsTruncated'):
            continuation_token = response.get('NextContinuationToken')
        else:
            break
    return contents


def _find_data_zip_key(contents, dataset_prefix):
    for item in contents:
        key = item.get('Key', '')
        if not key.startswith(dataset_prefix):
            continue
        if key.lower().endswith('data.zip'):
            return key
    return None


def _collect_json_keys(contents, dataset_prefix):
    json_keys = []
    for item in contents:
        key = item.get('Key', '')
        if not key.startswith(dataset_prefix):
            continue
        if key.lower().endswith('.json'):
            json_keys.append(key)
    return json_keys


def _select_metadata_v1_bytes(json_keys):
    for key in json_keys:
        try:
            candidate = _get_data_from_bucket(key)
            try:
                metadict = json.loads(candidate)
            except Exception:
                metadict = json.loads(candidate.decode('utf-8'))
            if isinstance(metadict, dict) and metadict.get('METADATA_SPEC_VERSION') == 'V1':
                return candidate
        except Exception:
            continue
    return None


def _iter_metainfo_yaml_entries(contents, modality):
    for item in contents:
        key = item.get('Key') or ''
        if not key.lower().endswith('metainfo.yaml'):
            continue
        parts = key.split('/')
        if len(parts) < 3 or parts[0] != modality:
            continue
        yield parts[1], key


def _download(modality, dataset_name):
    dataset_prefix = f'{modality}/{dataset_name}/'
    dataset_url = f'{BUCKET_URL}/{dataset_prefix}'
    LOGGER.info(f'Downloading dataset {dataset_name} from {dataset_url}')

    try:
        contents = _list_objects(dataset_prefix)
    except ClientError as exc:
        raise DemoResourceNotFoundError(
            f"Failed to list resources for dataset '{dataset_name}' in '{modality}'."
        ) from exc
    if not contents:
        raise DemoResourceNotFoundError(
            f"Dataset '{dataset_name}' under modality '{modality}' was not found."
        )

    data_zip_key = _find_data_zip_key(contents, dataset_prefix)
    json_keys = _collect_json_keys(contents, dataset_prefix)

    if not data_zip_key:
        raise DemoResourceNotFoundError(
            f"Could not find 'data.zip' for dataset '{dataset_name}' in modality '{modality}'."
        )

    metadata_bytes = _select_metadata_v1_bytes(json_keys)

    if metadata_bytes is None:
        raise DemoResourceNotFoundError(
            f"Could not locate a metadata JSON with METADATA_SPEC_VERSION 'V1' for dataset "
            f"'{dataset_name}' in modality '{modality}'."
        )

    try:
        data_zip_bytes = _get_data_from_bucket(data_zip_key)
    except ClientError as exc:
        raise DemoResourceNotFoundError(
            f"Failed to download 'data.zip' for dataset '{dataset_name}'."
        ) from exc

    return io.BytesIO(data_zip_bytes), metadata_bytes


def _extract_data(bytes_io, output_folder_name):
    with ZipFile(bytes_io) as zf:
        if output_folder_name:
            os.makedirs(output_folder_name, exist_ok=True)
            zf.extractall(output_folder_name)
            metadata_v0_filepath = os.path.join(output_folder_name, 'metadata_v0.json')
            if os.path.isfile(metadata_v0_filepath):
                os.remove(metadata_v0_filepath)
            try:
                os.rename(
                    os.path.join(output_folder_name, 'metadata_v1.json'),
                    os.path.join(output_folder_name, METADATA_FILENAME),
                )
            except FileNotFoundError:
                # No metadata inside the zip under new structure; ignore.
                pass
            except OSError:
                # Any other rename issue should not break extraction.
                pass

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


def _get_metadata(output_folder_name, in_memory_directory, dataset_name):
    metadata = Metadata()
    metadict = None
    if output_folder_name:
        try:
            json_files = [
                fn for fn in os.listdir(output_folder_name) if fn.lower().endswith('.json')
            ]
        except Exception:
            json_files = []

        for filename in json_files:
            path = os.path.join(output_folder_name, filename)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    candidate = json.load(f)
                if isinstance(candidate, dict) and candidate.get('METADATA_SPEC_VERSION') == 'V1':
                    metadict = candidate
                    break
            except Exception:
                continue

        if metadict is None:
            raise DemoResourceNotFoundError(
                f"Metadata JSON with METADATA_SPEC_VERSION 'V1' not found for {dataset_name}."
            )
        metadata = metadata.load_from_dict(metadict, dataset_name)

    else:
        json_keys = [
            key for key in (in_memory_directory or {}).keys() if key.lower().endswith('.json')
        ]
        for key in json_keys:
            try:
                raw = in_memory_directory[key]
                metadict_candidate = json.loads(raw if isinstance(raw, str) else raw.decode())
                if (
                    isinstance(metadict_candidate, dict)
                    and metadict_candidate.get('METADATA_SPEC_VERSION') == 'V1'
                ):
                    metadict = metadict_candidate
                    break
            except Exception:
                continue

        if metadict is None:
            raise DemoResourceNotFoundError(
                f"Metadata JSON with METADATA_SPEC_VERSION 'V1' not found for {dataset_name}."
            )
        metadata = metadata.load_from_dict(metadict, dataset_name)

    return metadata


def download_demo(modality, dataset_name, output_folder_name=None):
    """Download a demo dataset.

    Args:
        modality (str):
            The modality of the dataset: ``'single_table'``, ``'multi_table'``, ``'sequential'``.
        dataset_name (str):
            Name of the dataset to be downloaded from the sdv-datasets S3 bucket.
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
    bytes_io, metadata_bytes = _download(modality, dataset_name)
    in_memory_directory = _extract_data(bytes_io, output_folder_name)

    if output_folder_name:
        try:
            metadata_path = os.path.join(output_folder_name, METADATA_FILENAME)
            with open(metadata_path, 'wb') as f:
                f.write(metadata_bytes)
        except Exception as exc:
            raise DemoResourceNotFoundError(
                f"Failed to save metadata for dataset '{dataset_name}'."
            ) from exc
    else:
        if in_memory_directory is None:
            in_memory_directory = {}
        in_memory_directory[METADATA_FILENAME] = metadata_bytes

    data = _get_data(modality, output_folder_name, in_memory_directory)
    metadata = _get_metadata(output_folder_name, in_memory_directory, dataset_name)

    return data, metadata


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

    Raises:
        Error:
            * If ``modality`` is not ``'single_table'``, ``'multi_table'`` or ``'sequential'``.
    """
    _validate_modalities(modality)
    tables_info = defaultdict(list)
    prefix = f'{modality}/'
    try:
        contents = _list_objects(prefix)
    except ClientError:
        contents = []

    for dataset_name, key in _iter_metainfo_yaml_entries(contents, modality):
        # Read YAML safely and extract fields
        try:
            yaml_bytes = _get_data_from_bucket(key)
            meta = yaml.safe_load(yaml_bytes) or {}
        except Exception:
            continue

        size_mb_val = np.nan
        num_tables_val = np.nan
        try:
            size_mb_val = round(float(meta.get('dataset-size-mb', np.nan)), 2)
        except Exception:
            size_mb_val = np.nan
        try:
            num_tables_val = int(meta.get('num-tables', np.nan))
        except Exception:
            num_tables_val = np.nan

        tables_info['dataset_name'].append(dataset_name)
        tables_info['size_MB'].append(size_mb_val)
        tables_info['num_tables'].append(num_tables_val)

    df = pd.DataFrame(tables_info)
    if df.empty:
        df = pd.DataFrame({
            'dataset_name': pd.Series(dtype=str),
            'size_MB': pd.Series(dtype=float),
            'num_tables': pd.Series(dtype=float),
        })
    else:
        df['num_tables'] = pd.to_numeric(df['num_tables'], errors='coerce')
        df['size_MB'] = pd.to_numeric(df['size_MB'], errors='coerce')
    return df
