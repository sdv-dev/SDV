import pytest

from sdv.datasets.demo import download_demo
from copy import deepcopy
from sdv.metadata.metadata import Metadata

import pandas as pd

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
def data_metadata_1_to_1_subset_diamond(fake_hotels):
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
def data_metadata_1_to_1_to_1_subset_to_subset(data_metadata_1_to_1_subset_diamond):
    data, metadata = data_metadata_1_to_1_subset_diamond
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
