"""Functions to load demo datasets."""

import io
import logging
import os
import urllib.request
from zipfile import ZipFile

import numpy as np
import pandas as pd
from faker import Faker

from sdv.metadata import Metadata, Table

LOGGER = logging.getLogger(__name__)


DEMO_METADATA = {
    'tables': {
        'users': {
            'primary_key': 'user_id',
            'fields': {
                'user_id': {
                    'type': 'id',
                    'subtype': 'integer'
                },
                'country': {
                    'type': 'categorical'
                },
                'gender': {
                    'type': 'categorical'
                },
                'age': {
                    'type': 'numerical',
                    'subtype': 'integer'
                }
            }
        },
        'sessions': {
            'primary_key': 'session_id',
            'fields': {
                'session_id': {
                    'type': 'id',
                    'subtype': 'integer'
                },
                'user_id': {
                    'ref': {
                        'field': 'user_id',
                        'table': 'users'
                    },
                    'type': 'id',
                    'subtype': 'integer'
                },
                'device': {
                    'type': 'categorical'
                },
                'os': {
                    'type': 'categorical'
                }
            }
        },
        'transactions': {
            'primary_key': 'transaction_id',
            'fields': {
                'transaction_id': {
                    'type': 'id',
                    'subtype': 'integer'
                },
                'session_id': {
                    'ref': {
                        'field': 'session_id',
                        'table': 'sessions'
                    },
                    'type': 'id',
                    'subtype': 'integer'
                },
                'timestamp': {
                    'type': 'datetime',
                    'format': '%Y-%m-%d'
                },
                'amount': {
                    'type': 'numerical',
                    'subtype': 'float'
                },
                'approved': {
                    'type': 'boolean'
                }
            }
        }
    }
}


DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
DATA_URL = 'https://sdv-datasets.s3.amazonaws.com/{}.zip'
DATASETS_URL = 'https://sdv-datasets.s3.amazonaws.com/datasets.csv'


def _dtypes64(table):
    for name, column in table.items():
        if column.dtype == np.int32:
            table[name] = column.astype('int64')
        elif column.dtype == np.float32:
            table[name] = column.astype('float64')

    return table


def _download(dataset_name, data_path):
    url = DATA_URL.format(dataset_name)

    LOGGER.info('Downloading dataset {} from {}'.format(dataset_name, url))
    response = urllib.request.urlopen(url)
    bytes_io = io.BytesIO(response.read())

    LOGGER.info('Extracting dataset into {}'.format(data_path))
    with ZipFile(bytes_io) as zf:
        zf.extractall(data_path)


def _get_dataset_path(dataset_name, data_path):
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if not os.path.exists(os.path.join(data_path, dataset_name)):
        _download(dataset_name, data_path)

    return os.path.join(data_path, dataset_name)


def _load_relational_dummy():
    users = pd.DataFrame({
        'user_id': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'country': ['US', 'UK', 'ES', 'UK', 'US', 'DE', 'BG', 'ES', 'FR', 'UK'],
        'gender': ['M', 'F', None, 'M', 'F', 'M', 'F', None, 'F', None],
        'age': [34, 23, 44, 22, 54, 57, 45, 41, 23, 30]
    })
    sessions = pd.DataFrame({
        'session_id': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'user_id': [0, 1, 1, 2, 4, 5, 6, 6, 6, 8],
        'device': ['mobile', 'tablet', 'tablet', 'mobile', 'mobile',
                   'mobile', 'mobile', 'tablet', 'mobile', 'tablet'],
        'os': ['android', 'ios', 'android', 'android', 'ios',
               'android', 'ios', 'ios', 'ios', 'ios']
    })
    transactions = pd.DataFrame({
        'transaction_id': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'session_id': [0, 0, 1, 3, 5, 5, 7, 8, 9, 9],
        'timestamp': ['2019-01-01T12:34:32', '2019-01-01T12:42:21', '2019-01-07T17:23:11',
                      '2019-01-10T11:08:57', '2019-01-10T21:54:08', '2019-01-11T11:21:20',
                      '2019-01-22T14:44:10', '2019-01-23T10:14:09', '2019-01-27T16:09:17',
                      '2019-01-29T12:10:48'],
        'amount': [100.0, 55.3, 79.5, 112.1, 110.0, 76.3, 89.5, 132.1, 68.0, 99.9],
        'approved': [True, True, True, False, False, True, True, False, True, True],
    })
    transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])

    tables = {
        'users': _dtypes64(users),
        'sessions': _dtypes64(sessions),
        'transactions': _dtypes64(transactions),
    }

    return Metadata(DEMO_METADATA), tables


def _load_demo_dataset(dataset_name, data_path):
    dataset_path = _get_dataset_path(dataset_name, data_path)
    meta = Metadata(metadata=os.path.join(dataset_path, 'metadata.json'))
    tables = {
        name: _dtypes64(table)
        for name, table in meta.load_tables().items()
    }
    return meta, tables


def load_demo(dataset_name=None, data_path=DATA_PATH, metadata=False):
    """Load relational demo data.

    If a dataset name is given, it is downloaded from the sdv-datasets S3 bucket.
    Otherwise, a toy dataset with three simple tables is loaded:

        * users: user data including country, gender and age.
        * sessions: sessions data with a foreign key to user.
        * transactions: transactions data with a foreign key to sessions.

    If ``metadata`` is ``True``, the output will be a tuple with a ``Metadata``
    instance for the dataset and a ``tables`` dict that contains the tables loaded
    as ``pandas.DataFrames``.
    If ``metadata`` is ``False``, only the ``tables`` are returned.

    Args:
        dataset_name (str):
            Dataset name to be downloaded, if ``None`` use default demo data. Defaults to ``None``.
        data_path (str):
            Data path to save the dataset files, only used if dataset_name is provided.
            Defaults to ``DATA_PATH``.
        metadata (bool):
            If ``True`` return Metadata object. Defaults to ``False``.

    Returns:
        dict or tuple:
            If ``metadata`` is ``False`` return a ``dict`` with the tables data.
            If ``metadata`` is ``True`` return a ``tuple`` with Metadata and tables data.
    """
    if dataset_name:
        meta, tables = _load_demo_dataset(dataset_name, data_path)
    else:
        meta, tables = _load_relational_dummy()

    if metadata:
        return meta, tables

    return tables


def _load_tabular_dummy():
    """Load a dummy tabular demo dataframe."""
    age = np.random.randint(30, 50, 12)
    age_when_joined = age - np.random.randint(0, 10, 12)
    faker = Faker()
    names = [faker.name() for _ in range(12)]
    adresses = [faker.address() for _ in range(12)]

    return pd.DataFrame({
        'company': ['Pear', 'Pear', 'Glasses', 'Glasses', 'Cheerper', 'Cheerper'] * 2,
        'department': ['Sales', 'Design', 'AI', 'Search Engine', 'BigData', 'Support'] * 2,
        'name': names,
        'address': adresses,
        'age': age,
        'age_when_joined': age_when_joined,
        'years_in_the_company': age - age_when_joined
    })


def load_tabular_demo(dataset_name=None, table_name=None, data_path=DATA_PATH, metadata=False):
    """Load a tabular demo.

    If a dataset name is given, it is downloaded from the sdv-datasets S3 bucket.
    Otherwise, a toy dataset with a single table that contains data from a short fake
    collection of employees.

    If ``metadata`` is ``True``, the output will be a tuple with a ``Metadata``
    instance for the dataset and a ``pandas.DataFrame`` with the data from the table.
    If ``metadata`` is ``False``, only the ``pandas.DataFrame`` is returned.

    Args:
        dataset_name (str):
            Dataset name to be downloaded, if ``None`` use default demo data. Defaults to ``None``.
        table_name (str):
            If a table name is given, return this table from the indicated dataset.
            Otherwise, return the first one.
        data_path (str):
            Data path to save the dataset files, only used if dataset_name is provided.
            Defaults to ``DATA_PATH``.
        metadata (bool):
            If ``True`` also return a Table object. Defaults to ``False``.

    Returns:
        pandas.DataFrame or tuple:
            If ``metadata`` is ``False`` return a ``pandas.DataFrame`` with the tables data.
            If ``metadata`` is ``True`` return a ``tuple`` with a Table and the data.
    """
    if dataset_name:
        meta, tables = _load_demo_dataset(dataset_name, data_path)

        if table_name is None:
            table_name = meta.get_tables()[0]

        table = _dtypes64(tables[table_name])

        if metadata:
            return Table.from_dict(meta.get_table_meta(table_name)), table

        return table

    table = _dtypes64(_load_tabular_dummy())
    if metadata:
        table_meta = Table.from_dict({
            'fields': {
                'company': {'type': 'categorical'},
                'department': {'type': 'categorical'},
                'name': {'type': 'categorical'},
                'address': {'type': 'categorical'},
                'age': {'type': 'numerical', 'subtype': 'integer'},
                'age_when_joined': {'type': 'numerical', 'subtype': 'integer'},
                'years_in_the_company': {'type': 'numerical', 'subtype': 'integer'}
            },
            'constraints': [
                {
                    'constraint': 'UniqueCombinations',
                    'columns': ['company', 'department'],
                },
                {
                    'constraint': 'GreaterThan',
                    'low': 'age_when_joined',
                    'high': 'age'
                }
            ],
            'model_kwargs': {}
        })
        return table_meta, table

    return table


def load_timeseries_demo(dataset_name=None, table_name=None, metadata=False):
    """Load a timeseries demo.

    If a dataset name is given, it is downloaded from the sdv-datasets S3 bucket.
    Otherwise, a the NASDAQ100_2019 dataset is loaded.

    If ``metadata`` is ``True``, the output will be a tuple with a ``Metadata``
    instance for the dataset and a ``pandas.DataFrame`` with the data from the table.
    If ``metadata`` is ``False``, only the ``pandas.DataFrame`` is returned.

    Args:
        dataset_name (str):
            Dataset name to be downloaded, if ``None`` use default dataset. Defaults to ``None``.
        table_name (str):
            If a table name is given, return this table from the indicated dataset.
            Otherwise, return the first one.
        data_path (str):
            Data path to save the dataset files, only used if dataset_name is provided.
            Defaults to ``DATA_PATH``.
        metadata (bool):
            If ``True`` also return a Table object. Defaults to ``False``.

    Returns:
        pandas.DataFrame or tuple:
            If ``metadata`` is ``False`` return a ``pandas.DataFrame`` with the tables data.
            If ``metadata`` is ``True`` return a ``tuple`` with a Table and the data.
    """
    dataset_name = dataset_name or 'nasdaq100_2019'
    return load_tabular_demo(dataset_name, table_name, data_path=DATA_PATH, metadata=metadata)


def get_available_demos():
    """Get available demos and information about them.

    Returns:
        pandas.DataFrame:
            Table with the available demos.
    """
    return pd.read_csv(DATASETS_URL)
