"""Integration tests for Multi Table Metadata."""

import json
from unittest.mock import patch

from sdv.datasets.demo import download_demo
from sdv.metadata import MultiTableMetadata, SingleTableMetadata
from tests.utils import get_multi_table_metadata


def test_multi_table_metadata():
    """Test ``MultiTableMetadata``."""
    # Create an instance
    instance = MultiTableMetadata()

    # To dict
    result = instance.to_dict()

    # Assert
    assert result == {'tables': {}, 'relationships': [], 'METADATA_SPEC_VERSION': 'MULTI_TABLE_V1'}
    assert instance.tables == {}
    assert instance.relationships == []


@patch('rdt.transformers')
def test_add_column_relationship(mock_rdt_transformers):
    """Test ``add_column_relationship`` method."""

    # Setup
    class RandomLocationGeneratorMock:
        @classmethod
        def _validate_sdtypes(cls, columns_to_sdtypes):
            pass

    mock_rdt_transformers.address.RandomLocationGenerator = RandomLocationGeneratorMock
    _, instance = download_demo('multi_table', 'fake_hotels')
    instance.update_column('hotels', 'city', sdtype='city')
    instance.update_column('hotels', 'state', sdtype='state')

    # Run
    instance.add_column_relationship('hotels', 'address', ['city', 'state'])

    # Assert
    instance.validate()
    assert instance.tables['hotels'].column_relationships == [
        {'type': 'address', 'column_names': ['city', 'state']}
    ]


def test_remove_primary_key():
    # Setup
    metadata = get_multi_table_metadata()

    # Run
    metadata.remove_primary_key('nesreca')

    # Assert
    expected_relationships = [
        {
            'parent_table_name': 'upravna_enota',
            'parent_primary_key': 'id_upravna_enota',
            'child_table_name': 'nesreca',
            'child_foreign_key': 'upravna_enota',
        },
        {
            'parent_table_name': 'upravna_enota',
            'parent_primary_key': 'id_upravna_enota',
            'child_table_name': 'oseba',
            'child_foreign_key': 'upravna_enota',
        },
    ]
    assert metadata.tables['nesreca'].primary_key is None
    assert metadata.relationships == expected_relationships


def test_upgrade_metadata(tmp_path):
    """Test the ``upgrade_metadata`` method."""
    # Setup
    old_metadata = {
        'tables': {
            'nesreca': {
                'fields': {
                    'upravna_enota': {
                        'type': 'id',
                        'subtype': 'integer',
                        'ref': {'table': 'upravna_enota', 'field': 'id_upravna_enota'},
                    },
                    'id_nesreca': {'type': 'id', 'subtype': 'integer'},
                },
                'primary_key': 'id_nesreca',
            },
            'oseba': {
                'fields': {
                    'upravna_enota': {
                        'type': 'id',
                        'subtype': 'integer',
                        'ref': {'table': 'upravna_enota', 'field': 'id_upravna_enota'},
                    },
                    'id_nesreca': {
                        'type': 'id',
                        'subtype': 'integer',
                        'ref': {'table': 'nesreca', 'field': 'id_nesreca'},
                    },
                },
            },
            'upravna_enota': {
                'fields': {'id_upravna_enota': {'type': 'id', 'subtype': 'integer'}},
                'primary_key': 'id_upravna_enota',
            },
        }
    }
    filepath = tmp_path / 'old.json'
    old_metadata_file = open(filepath, 'w')
    json.dump(old_metadata, old_metadata_file)
    old_metadata_file.close()

    # Run
    new_metadata = MultiTableMetadata.upgrade_metadata(filepath=filepath).to_dict()

    # Assert
    expected_metadata = {
        'tables': {
            'nesreca': {
                'primary_key': 'id_nesreca',
                'columns': {
                    'upravna_enota': {'sdtype': 'id', 'regex_format': r'\d{30}'},
                    'id_nesreca': {'sdtype': 'id', 'regex_format': r'\d{30}'},
                },
            },
            'oseba': {
                'columns': {
                    'upravna_enota': {'sdtype': 'id', 'regex_format': r'\d{30}'},
                    'id_nesreca': {'sdtype': 'id', 'regex_format': r'\d{30}'},
                }
            },
            'upravna_enota': {
                'primary_key': 'id_upravna_enota',
                'columns': {'id_upravna_enota': {'sdtype': 'id', 'regex_format': r'\d{30}'}},
            },
        },
        'relationships': [
            {
                'parent_table_name': 'upravna_enota',
                'parent_primary_key': 'id_upravna_enota',
                'child_table_name': 'nesreca',
                'child_foreign_key': 'upravna_enota',
            },
            {
                'parent_table_name': 'upravna_enota',
                'parent_primary_key': 'id_upravna_enota',
                'child_table_name': 'oseba',
                'child_foreign_key': 'upravna_enota',
            },
            {
                'parent_table_name': 'nesreca',
                'parent_primary_key': 'id_nesreca',
                'child_table_name': 'oseba',
                'child_foreign_key': 'id_nesreca',
            },
        ],
        'METADATA_SPEC_VERSION': 'MULTI_TABLE_V1',
    }
    assert new_metadata['METADATA_SPEC_VERSION'] == expected_metadata['METADATA_SPEC_VERSION']
    assert new_metadata['tables'] == expected_metadata['tables']
    for relationship in new_metadata['relationships']:
        assert relationship in expected_metadata['relationships']


def test_detect_from_dataframes():
    """Test the ``detect_from_dataframes`` method."""
    # Setup
    real_data, _ = download_demo(modality='multi_table', dataset_name='fake_hotels')

    metadata = MultiTableMetadata()

    # Run
    metadata.detect_from_dataframes(real_data)

    # Assert
    metadata.update_column(
        table_name='hotels',
        column_name='classification',
        sdtype='categorical',
    )

    expected_metadata = {
        'tables': {
            'hotels': {
                'columns': {
                    'hotel_id': {'sdtype': 'id'},
                    'city': {'sdtype': 'city', 'pii': True},
                    'state': {'sdtype': 'administrative_unit', 'pii': True},
                    'rating': {'sdtype': 'numerical'},
                    'classification': {'sdtype': 'categorical'},
                },
                'primary_key': 'hotel_id',
            },
            'guests': {
                'columns': {
                    'guest_email': {'sdtype': 'email', 'pii': True},
                    'hotel_id': {'sdtype': 'id'},
                    'has_rewards': {'sdtype': 'categorical'},
                    'room_type': {'sdtype': 'categorical'},
                    'amenities_fee': {'sdtype': 'numerical'},
                    'checkin_date': {'sdtype': 'datetime', 'datetime_format': '%d %b %Y'},
                    'checkout_date': {'sdtype': 'datetime', 'datetime_format': '%d %b %Y'},
                    'room_rate': {'sdtype': 'numerical'},
                    'billing_address': {'sdtype': 'unknown', 'pii': True},
                    'credit_card_number': {'sdtype': 'credit_card_number', 'pii': True},
                },
                'primary_key': 'guest_email',
            },
        },
        'relationships': [
            {
                'parent_table_name': 'hotels',
                'child_table_name': 'guests',
                'parent_primary_key': 'hotel_id',
                'child_foreign_key': 'hotel_id',
            }
        ],
        'METADATA_SPEC_VERSION': 'MULTI_TABLE_V1',
    }
    assert metadata.to_dict() == expected_metadata


def test_detect_from_csvs(tmp_path):
    """Test the ``detect_from_csvs`` method."""
    # Setup
    real_data, _ = download_demo(modality='multi_table', dataset_name='fake_hotels')

    metadata = MultiTableMetadata()

    for table_name, dataframe in real_data.items():
        csv_path = tmp_path / f'{table_name}.csv'
        dataframe.to_csv(csv_path, index=False)

    # Run
    metadata.detect_from_csvs(folder_name=tmp_path)

    # Assert
    metadata.update_column(
        table_name='hotels',
        column_name='classification',
        sdtype='categorical',
    )

    expected_metadata = {
        'tables': {
            'hotels': {
                'columns': {
                    'hotel_id': {'sdtype': 'id'},
                    'city': {'sdtype': 'city', 'pii': True},
                    'state': {'sdtype': 'administrative_unit', 'pii': True},
                    'rating': {'sdtype': 'numerical'},
                    'classification': {'sdtype': 'categorical'},
                },
                'primary_key': 'hotel_id',
            },
            'guests': {
                'columns': {
                    'guest_email': {'sdtype': 'email', 'pii': True},
                    'hotel_id': {'sdtype': 'id'},
                    'has_rewards': {'sdtype': 'categorical'},
                    'room_type': {'sdtype': 'categorical'},
                    'amenities_fee': {'sdtype': 'numerical'},
                    'checkin_date': {'sdtype': 'datetime', 'datetime_format': '%d %b %Y'},
                    'checkout_date': {'sdtype': 'datetime', 'datetime_format': '%d %b %Y'},
                    'room_rate': {'sdtype': 'numerical'},
                    'billing_address': {'sdtype': 'unknown', 'pii': True},
                    'credit_card_number': {'sdtype': 'credit_card_number', 'pii': True},
                },
                'primary_key': 'guest_email',
            },
        },
        'relationships': [
            {
                'parent_table_name': 'hotels',
                'child_table_name': 'guests',
                'parent_primary_key': 'hotel_id',
                'child_foreign_key': 'hotel_id',
            }
        ],
        'METADATA_SPEC_VERSION': 'MULTI_TABLE_V1',
    }

    assert metadata.to_dict() == expected_metadata


def test_get_column_names():
    """Test the ``get_column_names`` method."""
    # Setup
    metadata = get_multi_table_metadata()

    # Run
    matches = metadata.get_column_names('nesreca', sdtype='id')

    # Assert
    assert set(matches) == {'upravna_enota', 'id_nesreca'}


def test_get_table_metadata():
    """Test the ``get_table_metadata`` method."""
    # Setup
    metadata = get_multi_table_metadata()
    metadata.add_column('nesreca', 'latitude', sdtype='latitude')
    metadata.add_column('nesreca', 'longitude', sdtype='longitude')
    metadata.add_column_relationship('nesreca', 'gps', ['latitude', 'longitude'])

    # Run
    table_metadata = metadata.get_table_metadata('nesreca')

    # Assert
    assert isinstance(table_metadata, SingleTableMetadata)
    expected_metadata = {
        'primary_key': 'id_nesreca',
        'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1',
        'columns': {
            'upravna_enota': {'sdtype': 'id'},
            'id_nesreca': {'sdtype': 'id'},
            'nesreca_val': {'sdtype': 'numerical'},
            'latitude': {'sdtype': 'latitude', 'pii': True},
            'longitude': {'sdtype': 'longitude', 'pii': True},
        },
        'column_relationships': [{'type': 'gps', 'column_names': ['latitude', 'longitude']}],
    }
    assert table_metadata.to_dict() == expected_metadata


def test_anonymize():
    """Test the ``anonymize`` method."""
    # Setup
    metadata_dict = {
        'tables': {
            'real_table1': {
                'columns': {
                    'table1_primary_key': {'sdtype': 'id', 'regex_format': 'ID_[0-9]{3}'},
                    'table1_column2': {'sdtype': 'categorical'},
                },
                'primary_key': 'table1_primary_key',
            },
            'real_table2': {
                'columns': {
                    'table2_primary_key': {'sdtype': 'email'},
                    'table2_foreign_key': {'sdtype': 'id', 'regex_format': 'ID_[0-9]{3}'},
                },
                'primary_key': 'table2_primary_key',
            },
        },
        'relationships': [
            {
                'parent_table_name': 'real_table1',
                'parent_primary_key': 'table1_primary_key',
                'child_table_name': 'real_table2',
                'child_foreign_key': 'table2_foreign_key',
            }
        ],
    }
    metadata = MultiTableMetadata.load_from_dict(metadata_dict)
    table1_metadata = metadata.tables['real_table1']
    table2_metadata = metadata.tables['real_table2']
    metadata.validate()

    # Run
    anonymized = metadata.anonymize()

    # Assert
    anonymized.validate()

    assert anonymized.tables.keys() == {'table1', 'table2'}
    assert len(anonymized.relationships) == len(metadata.relationships)
    assert anonymized.relationships[0]['parent_table_name'] == 'table1'
    assert anonymized.relationships[0]['child_table_name'] == 'table2'
    assert anonymized.relationships[0]['parent_primary_key'] == 'col1'
    assert anonymized.relationships[0]['child_foreign_key'] == 'col2'

    anon_primary_key_metadata = anonymized.tables['table1'].columns['col1']
    assert anon_primary_key_metadata == table1_metadata.columns['table1_primary_key']

    anon_foreign_key_metadata = anonymized.tables['table2'].columns['col2']
    assert anon_foreign_key_metadata == table2_metadata.columns['table2_foreign_key']

    assert anonymized.tables['table1'].to_dict() == table1_metadata.anonymize().to_dict()
    assert anonymized.tables['table2'].to_dict() == table2_metadata.anonymize().to_dict()
