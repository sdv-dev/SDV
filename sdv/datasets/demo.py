"""Methods to load datasets."""

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
from sdv.datasets.errors import InvalidArgumentError
from sdv.metadata.dataset import Metadata


LOGGER = logging.getLogger(__name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
BUCKET_URL = 'https://sdv-demo-datasets.s3.amazonaws.com'
BUCKET_NAME = 'sdv-demo-datasets'


def boto_solution(dataset_name):
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    try:
        metadata = s3.head_object(Bucket=BUCKET_NAME, Key=dataset_name + '.zip')
        print(metadata)
    except ClientError as error:
        if error.response['ResponseMetadata']['HTTPStatusCode'] == 404:
            raise ValueError(
                f"Invalid dataset name '{dataset_name}'. Use 'list_available_demos' "
                'to get a list of demo datasets.'
            )
        print(error)






def _validate_args_download_demo(modality, dataset_name, output_folder_name):
    possible_modalities = ['single_table', 'multi_table', 'sequential']
    if modality not in possible_modalities:
        raise InvalidArgumentError(f"'modality' must be in {possible_modalities}.")
    
    if output_folder_name and os.path.exists(output_folder_name):
        raise InvalidArgumentError(
            f"Folder '{output_folder_name}' already exists. Please specify a different name "
            "or use 'load_csvs' to load from an existing folder."
        )

    url = f'{BUCKET_URL}/{modality.upper()}/{dataset_name}.zip'
    res = requests.head(url)
    if res.status_code != 200:
        other_modalities = set(possible_modalities) - set([modality])
        for other_modality in other_modalities:
            url = f'{BUCKET_URL}/{other_modality.upper()}/{dataset_name}.zip'
            res = requests.head(url)
            if res.status_code == 200:
                raise InvalidArgumentError(
                    f"Dataset name '{dataset_name}' is a '{other_modality}' dataset. "
                    f"Use 'load_{other_modality}_demo' to load this dataset."
                )

        raise InvalidArgumentError(
            f"Invalid dataset name '{dataset_name}'. "
            "Use 'list_available_demos' to get a list of demo datasets."
        )

def _download(modality, dataset_name, output_folder_name):
    url = f'{BUCKET_URL}/{modality.upper()}/{dataset_name}.zip'

    LOGGER.info(f'Downloading dataset {dataset_name} from {url}')
    response = urllib.request.urlopen(url)
    bytes_io = io.BytesIO(response.read())

    LOGGER.info(f'Extracting dataset into {output_folder_name}')
    with ZipFile(bytes_io) as zf:
        zf.extractall(output_folder_name)

def _get_dataset_path(dataset_name, modality, output_folder_name):
    os.makedirs(output_folder_name)
    _download(dataset_name, modality, output_folder_name)
    return os.path.join(output_folder_name, dataset_name)

def _dtypes64(table):
    for name, column in table.items():
        if column.dtype == np.int32:
            table[name] = column.astype('int64')
        elif column.dtype == np.float32:
            table[name] = column.astype('float64')

    return table

def _load_demo_dataset(modality, dataset_name, output_folder_name):
    dataset_path = _get_dataset_path(modality, dataset_name, output_folder_name)
    meta = Metadata(metadata=os.path.join(dataset_path, 'metadata.json'))
    tables = {
        name: _dtypes64(table)
        for name, table in meta.load_tables().items()
    }
    return meta, tables


def download_demo(modality, dataset_name, output_folder_name=None):
    """Download demo datasets.

    Args:
        modality (str):
            The modality of the dataset: ``'single_table'``, ``'multi_table'``, ``'sequential'``.
        dataset_name (str):
            Name of the dataset to be downloaded from the sdv-datasets S3 bucket.
        output_folder_name (str):
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
            * If ``modality`` is not one of ``'single_table'``, ``'multi_table'``, ``'sequential'``.
    """
    _validate_args_download_demo(modality, dataset_name, output_folder_name)
    return _load_demo_dataset(modality, dataset_name, output_folder_name)

