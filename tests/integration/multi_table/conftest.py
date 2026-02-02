import random
from copy import deepcopy

import pandas as pd
import pytest

from sdv.datasets.demo import download_demo
from sdv.metadata.metadata import Metadata


@pytest.fixture()
def fake_hotels():
    data, metadata = download_demo('multi_table', 'fake_hotels')
    return deepcopy(data), deepcopy(metadata)


@pytest.fixture()
def data_metadata_1_to_1(fake_hotels):
    data, metadata = fake_hotels
    guests_table = data['guests']
    guests_columns = [
        'guest_email',
        'has_rewards',
        'hotel_id',
        'billing_address',
        'credit_card_number',
    ]
    room_cols = [
        'guest_email',
        'room_type',
        'amenities_fee',
        'checkin_date',
        'checkout_date',
        'room_rate',
    ]
    new_guest_tables = {'guests': guests_columns, 'rooms': room_cols}
    metadata_dict = metadata.to_dict()
    for table, columns in new_guest_tables.items():
        data[table] = guests_table[columns]
        metadata_dict['tables'][table] = {
            'primary_key': 'guest_email',
            'columns': {col: metadata.tables['guests'].columns[col] for col in columns},
        }

    metadata_dict['relationships'] = [
        {
            'parent_table_name': 'hotels',
            'parent_primary_key': 'hotel_id',
            'child_table_name': 'guests',
            'child_foreign_key': 'hotel_id',
        },
        {
            'parent_table_name': 'guests',
            'parent_primary_key': 'guest_email',
            'child_table_name': 'rooms',
            'child_foreign_key': 'guest_email',
        },
    ]
    metadata = Metadata.load_from_dict(metadata_dict)
    return data, metadata


@pytest.fixture()
def data_metadata_1_to_1_or_0():
    data = {
        'users': pd.DataFrame({
            'user_id': range(10),
            'date_joined': [
                '2024-01-01',
                '2024-02-01',
                '2024-03-01',
                '2024-04-01',
                '2024-05-01',
            ]
            * 2,
        }),
        'survey_response': pd.DataFrame({
            'user_id': range(9),
            'age': [11, 22, 33, 44, 55, 66, 77, 88, 99],
        }),
    }
    metadata = Metadata.load_from_dict({
        'tables': {
            'users': {
                'columns': {
                    'user_id': {'sdtype': 'id'},
                    'date_joined': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
                },
                'primary_key': 'user_id',
            },
            'survey_response': {
                'columns': {
                    'user_id': {'sdtype': 'id'},
                    'age': {'sdtype': 'numerical'},
                },
                'primary_key': 'user_id',
            },
        },
        'relationships': [
            {
                'parent_table_name': 'users',
                'parent_primary_key': 'user_id',
                'child_table_name': 'survey_response',
                'child_foreign_key': 'user_id',
            }
        ],
    })
    return data, metadata


@pytest.fixture
def data_metadata_1_to_1_subset_arrow(fake_hotels):
    data, metadata = fake_hotels
    data.pop('hotels')
    metadata.remove_table('hotels')

    original_guests_table = data['guests']
    guest_cols = ['guest_email', 'has_rewards', 'hotel_id']
    guest_pii_cols = ['guest_email', 'billing_address', 'credit_card_number']
    room_cols = [
        'guest_email',
        'hotel_id',
        'room_type',
        'amenities_fee',
        'checkin_date',
        'checkout_date',
        'room_rate',
    ]
    new_guest_tables = {'guests': guest_cols, 'guests_pii': guest_pii_cols, 'rooms': room_cols}
    metadata_dict = metadata.to_dict()
    for table, columns in new_guest_tables.items():
        data[table] = original_guests_table[columns]
        metadata_dict['tables'][table] = {
            'primary_key': 'guest_email',
            'columns': {col: metadata.tables['guests'].columns[col] for col in columns},
        }
    metadata_dict['relationships'] = [
        {
            'parent_table_name': 'guests',
            'parent_primary_key': 'guest_email',
            'child_table_name': 'guests_pii',
            'child_foreign_key': 'guest_email',
        },
        {
            'parent_table_name': 'guests',
            'parent_primary_key': 'guest_email',
            'child_table_name': 'rooms',
            'child_foreign_key': 'guest_email',
        },
    ]
    data['guests_pii'] = data['guests_pii'].head(100)
    data['rooms'] = data['rooms'].head(50)
    metadata = Metadata.load_from_dict(metadata_dict)
    return deepcopy(data), deepcopy(metadata)


@pytest.fixture
def data_metadata_1_to_1_to_1_subset_to_subset(data_metadata_1_to_1_subset_arrow):
    data, metadata = data_metadata_1_to_1_subset_arrow
    metadata_dict = metadata.to_dict()
    metadata_dict['relationships'] = [
        {
            'parent_table_name': 'guests',
            'parent_primary_key': 'guest_email',
            'child_table_name': 'guests_pii',
            'child_foreign_key': 'guest_email',
        },
        {
            'parent_table_name': 'guests_pii',
            'parent_primary_key': 'guest_email',
            'child_table_name': 'rooms',
            'child_foreign_key': 'guest_email',
        },
    ]
    metadata = Metadata.load_from_dict(metadata_dict)
    return data, metadata


@pytest.fixture
def data_metadata_multiple_foreign_keys():
    parent_1_ids = range(0, 10)
    parent_2_ids = range(10, 20)
    parent = pd.DataFrame({
        'parent_id': parent_1_ids,
        'col_categorical': random.choices(['A', 'B', 'C', 'D', 'E'], k=10),
    })
    child = pd.DataFrame({
        'parent_1_id': parent_1_ids,
        'parent_2_id': parent_2_ids,
        'col_numerical': [10.2, 20.3] * 5,
    })
    second_parent = pd.DataFrame({'parent_id': parent_2_ids, 'col_boolean': [True, False] * 5})
    data = {
        'parent': parent,
        'child': child,
        'second_parent': second_parent,
    }
    metadata = Metadata.load_from_dict({
        'tables': {
            'parent': {
                'columns': {
                    'parent_id': {'sdtype': 'id'},
                    'col_categorical': {'sdtype': 'categorical'},
                },
                'primary_key': 'parent_id',
            },
            'child': {
                'columns': {
                    'parent_1_id': {'sdtype': 'id'},
                    'parent_2_id': {'sdtype': 'id'},
                    'col_numerical': {'sdtype': 'numerical'},
                },
                'primary_key': 'parent_1_id',
            },
            'second_parent': {
                'columns': {'parent_id': {'sdtype': 'id'}, 'col_boolean': {'sdtype': 'boolean'}},
                'primary_key': 'parent_id',
            },
        },
        'relationships': [
            {
                'parent_table_name': 'parent',
                'child_table_name': 'child',
                'parent_primary_key': 'parent_id',
                'child_foreign_key': 'parent_1_id',
            },
            {
                'parent_table_name': 'second_parent',
                'child_table_name': 'child',
                'parent_primary_key': 'parent_id',
                'child_foreign_key': 'parent_2_id',
            },
        ],
    })
    assert data['child']['parent_1_id'].equals(data['parent']['parent_id'])
    assert data['child']['parent_2_id'].equals(data['second_parent']['parent_id'])
    metadata.validate()
    metadata.validate_data(data)
    return data, metadata


@pytest.fixture
def data_metadata_multiple_foreign_keys_subset(data_metadata_multiple_foreign_keys):
    _, metadata = data_metadata_multiple_foreign_keys
    parent = pd.DataFrame({'parent_id': [1, 2, 3], 'col_categorical': ['A', 'B', 'C']})
    child = pd.DataFrame({
        'parent_1_id': [1, 2],
        'parent_2_id': [1, 1],
        'col_numerical': [100.5, 101.6],
    })
    second_parent = pd.DataFrame({
        'parent_id': [1, 2],
        'col_boolean': [True, False],
    })
    data = {
        'parent': parent,
        'child': child,
        'second_parent': second_parent,
    }
    assert set(data['child']['parent_1_id']).issubset(set(data['parent']['parent_id']))
    assert set(data['child']['parent_2_id']).issubset(set(data['second_parent']['parent_id']))
    metadata.validate()
    metadata.validate_data(data)
    return data, metadata
