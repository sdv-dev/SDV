"""Methods to load datasets."""

import io
import json
import logging
import os
import urllib.request
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import boto3
import numpy as np
import pandas as pd
from botocore import UNSIGNED
from botocore.client import Config

from sdv.metadata.multi_table import MultiTableMetadata
from sdv.metadata.single_table import SingleTableMetadata

LOGGER = logging.getLogger(__name__)
BUCKET = 'sdv-demo-datasets'
BUCKET_URL = 'https://sdv-demo-datasets.s3.amazonaws.com'
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


def _download(modality, dataset_name):
    dataset_url = f'{BUCKET_URL}/{modality.upper()}/{dataset_name}.zip'
    LOGGER.info(f'Downloading dataset {dataset_name} from {dataset_url}')
    try:
        response = urllib.request.urlopen(dataset_url)
    except urllib.error.HTTPError:
        raise ValueError(
            f"Invalid dataset name '{dataset_name}'. "
            'Make sure you have the correct modality for the dataset name or '
            "use 'get_available_demos' to get a list of demo datasets."
        )

    return io.BytesIO(response.read())


def _extract_data(bytes_io, output_folder_name):
    with ZipFile(bytes_io) as zf:
        if output_folder_name:
            os.makedirs(output_folder_name, exist_ok=True)
            zf.extractall(output_folder_name)
            os.remove(os.path.join(output_folder_name, 'metadata_v0.json'))
            os.rename(
                os.path.join(output_folder_name, 'metadata_v1.json'),
                os.path.join(output_folder_name, METADATA_FILENAME)
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
                data[table_name] = pd.read_csv(io.StringIO(file_.decode()))

    if modality != 'multi_table':
        data = data.popitem()[1]

    return data


def _get_metadata(modality, output_folder_name, in_memory_directory):
    metadata = MultiTableMetadata() if modality == 'multi_table' else SingleTableMetadata()
    if output_folder_name:
        metadata_path = os.path.join(output_folder_name, METADATA_FILENAME)
        metadata = metadata.load_from_json(metadata_path)

    else:
        metadict = json.loads(in_memory_directory['metadata_v1.json'])
        metadata = metadata.load_from_dict(metadict)

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
            If ``metadata`` is single table or sequential, it is a ``SingleTableMetadata`` object.
            If ``metadata`` is multi table, it is a ``MultiTableMetadata`` object.

    Raises:
        Error:
            * If the ``dataset_name`` exists in the bucket but under a different modality.
            * If the ``dataset_name`` doesn't exist in the bucket.
            * If there is already a folder named ``output_folder_name``.
            * If ``modality`` is not ``'single_table'``, ``'multi_table'`` or ``'sequential'``.
    """
    _validate_modalities(modality)
    _validate_output_folder(output_folder_name)
    bytes_io = _download(modality, dataset_name)
    in_memory_directory = _extract_data(bytes_io, output_folder_name)
    data = _get_data(modality, output_folder_name, in_memory_directory)
    metadata = _get_metadata(modality, output_folder_name, in_memory_directory)

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
    client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    tables_info = defaultdict(list)
    for item in client.list_objects(Bucket=BUCKET)['Contents']:
        dataset_modality, dataset = item['Key'].split('/', 1)
        if dataset_modality == modality.upper():
            tables_info['dataset_name'].append(dataset.replace('.zip', ''))
            headers = client.head_object(Bucket=BUCKET, Key=item['Key'])['Metadata']
            size_mb = headers.get('size-mb', np.nan)
            tables_info['size_MB'].append(round(float(size_mb), 2))
            tables_info['num_tables'].append(headers.get('num-tables', np.nan))

    return pd.DataFrame(tables_info)
