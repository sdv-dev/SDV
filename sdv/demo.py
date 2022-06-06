"""Functions to load demo datasets."""

import io
import logging
import os
import urllib.request
from datetime import datetime, timedelta
from zipfile import ZipFile

import numpy as np
import pandas as pd
import scipy as sp
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
                },
                'minutes': {
                    'type': 'numerical',
                    'subtype': 'integer'
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
                    'format': '%Y-%m-%dT%H:%M'
                },
                'amount': {
                    'type': 'numerical',
                    'subtype': 'float'
                },
                'cancelled': {
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
        'user_id': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'device': ['mobile', 'tablet', 'tablet', 'mobile', 'mobile',
                   'mobile', 'mobile', 'tablet', 'mobile', 'tablet'],
        'os': ['android', 'ios', 'android', 'android', 'ios',
               'android', 'ios', 'ios', 'ios', 'ios'],
        'minutes': [23, 12, 8, 13, 9, 32, 7, 21, 29, 34],
    })
    transactions = pd.DataFrame({
        'transaction_id': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'session_id': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'timestamp': ['2019-01-01T12:34:32', '2019-01-01T12:42:21', '2019-01-07T17:23:11',
                      '2019-01-10T11:08:57', '2019-01-10T21:54:08', '2019-01-11T11:21:20',
                      '2019-01-22T14:44:10', '2019-01-23T10:14:09', '2019-01-27T16:09:17',
                      '2019-01-29T12:10:48'],
        'amount': [100.0, 55.3, 79.5, 112.1, 110.0, 76.3, 89.5, 132.1, 68.0, 99.9],
        'cancelled': [False, False, False, True, True, False, False, True, False, False],
    })
    transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])

    tables = {
        'users': _dtypes64(users),
        'sessions': _dtypes64(sessions),
        'transactions': _dtypes64(transactions),
    }

    return Metadata(DEMO_METADATA), tables


def sample_relational_demo(size=30):
    """Sample demo data with the indicate number of rows in the parent table."""
    # Users
    faker = Faker()
    countries = [faker.country_code() for _ in range(5)]
    country = np.random.choice(countries, size=size)
    gender = np.random.choice(['F', 'M', None], p=[0.5, 0.4, 0.1], size=size)
    age = (
        sp.stats.truncnorm.rvs(-1.2, 1.5, loc=30, scale=10, size=size).astype(int)
        + 3 * (gender == 'M')
        + 3 * (country == countries[0]).astype(int)
    )
    num_sessions = (
        sp.stats.gamma.rvs(1, loc=0, scale=2, size=size)
        * (0.8 + 0.2 * (gender == 'F'))
    ).round().astype(int)

    users = pd.DataFrame({
        'country': country,
        'gender': gender,
        'age': age,
        'num_sessions': num_sessions
    })
    users.index.name = 'user_id'

    # Sessions
    sessions = pd.DataFrame()
    for user_id, user in users.iterrows():
        device_weights = [0.1, 0.4, 0.5] if user.gender == 'M' else [0.3, 0.4, 0.3]
        devices = np.random.choice(
            ['mobile', 'tablet', 'pc'],
            size=user.num_sessions,
            p=device_weights
        )
        os = []
        pc_weights = [0.6, 0.3, 0.1] if user.age > 30 else [0.2, 0.4, 0.4]
        pc_os = np.random.choice(['windows', 'macos', 'linux'], p=pc_weights)
        phone_weights = [0.7, 0.3] if user.age > 30 else [0.9, 0.1]
        phone_os = np.random.choice(['android', 'ios'], p=phone_weights)
        for device in devices:
            os.append(pc_os if device == 'pc' else phone_os)

        minutes = (
            sp.stats.truncnorm.rvs(-3, 3, loc=30, scale=10, size=user.num_sessions)
            * (1 + 0.1 * (user.gender == 'M'))
            * (1 + user.age / 100)
            * (1 + 0.1 * (devices == 'pc'))
        )
        num_transactions = (minutes / 10) * (0.5 + (user.gender == 'F'))

        sessions = sessions.append(pd.DataFrame({
            'user_id': np.full(user.num_sessions, int(user_id)),
            'device': devices,
            'os': os,
            'minutes': minutes.round().astype(int),
            'num_transactions': num_transactions.round().astype(int),
        }), ignore_index=True)

    sessions.index.name = 'session_id'
    del users['num_sessions']

    # Transactions
    transactions = pd.DataFrame()
    for session_id, session in sessions.iterrows():
        size = session.num_transactions
        if size:
            amount_base = sp.stats.truncnorm.rvs(-2, 4, loc=100, scale=50, size=size)
            is_apple = session['os'] in ('ios', 'macos')
            amount_modif = np.random.random(size) * 100 * is_apple
            amount = amount_base / np.random.randint(1, size + 1) + amount_modif

            seconds = np.random.randint(3600 * 24 * 365)
            start = datetime(2019, 1, 1) + timedelta(seconds=seconds)

            timestamp = sorted([
                start + timedelta(seconds=int(seconds))
                for seconds in np.random.randint(60 * session.minutes, size=size)
            ])
            cancelled = np.random.random(size=size) < (1 / (size * 2))
            transactions = transactions.append(pd.DataFrame({
                'session_id': np.full(session.num_transactions, int(session_id)),
                'timestamp': timestamp,
                'amount': amount.round(2),
                'cancelled': cancelled,
            }), ignore_index=True)

    transactions.index.name = 'transaction_id'
    del sessions['num_transactions']

    tables = {
        'users': _dtypes64(users.reset_index()),
        'sessions': _dtypes64(sessions.reset_index()),
        'transactions': _dtypes64(transactions.reset_index()),
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
    age_when_joined = age - np.random.randint(1, 10, 12)
    years_exp = np.random.randint(1, 6, 12)
    contractor = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0] * 2

    is_contractor = np.array(contractor).astype(bool)
    salary = np.random.randint(60, 320, 12) * 500.
    bonus = np.random.randint(10, 50, 12) * 500.
    salary[is_contractor] = np.random.uniform(30000, 160000, 4).round(2)
    bonus[is_contractor] = np.random.uniform(5000, 25000, 4).round(2)

    return pd.DataFrame({
        'company': ['Pear', 'Pear', 'Glasses', 'Glasses', 'Cheerper', 'Cheerper'] * 2,
        'department': ['Sales', 'Design', 'AI', 'Search Engine', 'BigData', 'Support'] * 2,
        'employee_id': [1, 5, 1, 7, 6, 11, 28, 75, 33, 56, 42, 80],
        'age': age,
        'age_when_joined': age_when_joined,
        'years_in_the_company': age - age_when_joined,
        'salary': salary,
        'annual_bonus': bonus,
        'prior_years_experience': years_exp,
        'full_time': [1.0, 0.0, 1.0, 0.0, 0.0, 0.0] * 2,
        'part_time': [0.0, 0.0, 0.0, 0.0, 1.0, 1.0] * 2,
        'contractor': contractor
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
                'employee_id': {'type': 'numerical', 'subtype': 'integer'},
                'age': {'type': 'numerical', 'subtype': 'integer'},
                'age_when_joined': {'type': 'numerical', 'subtype': 'integer'},
                'years_in_the_company': {'type': 'numerical', 'subtype': 'integer'},
                'salary': {'type': 'numerical', 'subtype': 'float'},
                'annual_bonus': {'type': 'numerical', 'subtype': 'float'},
                'prior_years_experience': {'type': 'numerical', 'subtype': 'integer'}
            },
            'constraints': [
                {
                    'constraint': 'FixedCombinations',
                    'column_names': ['company', 'department'],
                },
                {
                    'constraint': 'Inequality',
                    'low_column_name': 'age_when_joined',
                    'high_column_name': 'age'
                },
                {
                    'constraint': 'ScalarInequality',
                    'value': 30000,
                    'column_name': 'salary'
                },
                {
                    'constraint': 'Positive',
                    'columns': 'prior_years_experience'
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
