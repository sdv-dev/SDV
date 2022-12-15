"""Methods to load datasets."""

import io
import logging
import os
import shutil
import tempfile
import urllib.request
from zipfile import ZipFile

import pandas as pd
import requests

from sdv.datasets.errors import InvalidArgumentError
from sdv.metadata.multi_table import MultiTableMetadata
from sdv.metadata.single_table import SingleTableMetadata

LOGGER = logging.getLogger(__name__)
BUCKET_URL = 'https://sdv-demo-datasets.s3.amazonaws.com'


def _get_dataset_url(modality, dataset_name):
    return os.path.join(BUCKET_URL, modality.upper(), dataset_name + '.zip')


def _validate_args_download_demo(modality, dataset_name, output_folder_name):
    possible_modalities = ['single_table', 'multi_table', 'sequential']
    if modality not in possible_modalities:
        raise InvalidArgumentError(f"'modality' must be in {possible_modalities}.")

    if output_folder_name and os.path.exists(output_folder_name):
        raise InvalidArgumentError(
            f"Folder '{output_folder_name}' already exists. Please specify a different name "
            "or use 'load_csvs' to load from an existing folder."
        )

    # If the dataset exists in the wrong modality, raise that error.
    # If the dataset doesn't exist at all, raise different error.
    dataset_url = _get_dataset_url(modality, dataset_name)
    response = requests.head(dataset_url)
    if response.status_code != 200:
        other_modalities = set(possible_modalities) - {modality}
        for other_modality in other_modalities:
            dataset_url = _get_dataset_url(other_modality, dataset_name)
            response = requests.head(dataset_url)
            if response.status_code == 200:
                raise InvalidArgumentError(
                    f"Dataset name '{dataset_name}' is a '{other_modality}' dataset. "
                    f"Use 'load_{other_modality}_demo' to load this dataset."
                )

        raise InvalidArgumentError(
            f"Invalid dataset name '{dataset_name}'. "
            "Use 'list_available_demos' to get a list of demo datasets."
        )


def _download(modality, dataset_name, output_folder_name):
    dataset_url = _get_dataset_url(modality, dataset_name)

    LOGGER.info(f'Downloading dataset {dataset_name} from {dataset_url}')
    response = urllib.request.urlopen(dataset_url)
    bytes_io = io.BytesIO(response.read())

    LOGGER.info(f'Extracting dataset into {output_folder_name}')
    with ZipFile(bytes_io) as zf:
        os.makedirs(output_folder_name, exist_ok=True)
        zf.extractall(output_folder_name)
        os.remove(os.path.join(output_folder_name, 'metadata_v0.json'))
        os.rename(
            os.path.join(output_folder_name, 'metadata_v1.json'),
            os.path.join(output_folder_name, 'metadata.json')
        )


def _get_data(modality, output_folder_name):
    data = {}
    for filename in os.listdir(output_folder_name):
        if filename != 'metadata.json':
            data_path = os.path.join(output_folder_name, filename)
            data[filename] = pd.read_csv(data_path)

    if modality != 'multi_table':
        data = data.popitem()[1]

    return data


def _get_metadata(modality, output_folder_name):
    metadata_path = os.path.join(output_folder_name, 'metadata.json')
    metadata = MultiTableMetadata() if modality == 'multi_table' else SingleTableMetadata()
    metadata = metadata.load_from_json(metadata_path)

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
    _validate_args_download_demo(modality, dataset_name, output_folder_name)

    use_temp_dir = False
    if output_folder_name is None:
        use_temp_dir = True
        output_folder_name = tempfile.mkdtemp()

    _download(modality, dataset_name, output_folder_name)
    data = _get_data(modality, output_folder_name)
    metadata = _get_metadata(modality, output_folder_name)

    if use_temp_dir:
        shutil.rmtree(output_folder_name)

    return data, metadata
