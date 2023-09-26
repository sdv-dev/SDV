"""Integration tests for Multi Table Metadata."""

import json

from sdv.datasets.demo import download_demo
from sdv.metadata import MultiTableMetadata


def test_multi_table_metadata():
    """Test ``MultiTableMetadata``."""

    # Create an instance
    instance = MultiTableMetadata()

    # To dict
    result = instance.to_dict()

    # Assert
    assert result == {
        'tables': {},
        'relationships': [],
        'METADATA_SPEC_VERSION': 'MULTI_TABLE_V1'
    }
    assert instance.tables == {}
    assert instance.relationships == []


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
                        'ref': {
                            'table': 'upravna_enota',
                            'field': 'id_upravna_enota'
                        }
                    },
                    'id_nesreca': {
                        'type': 'id',
                        'subtype': 'integer'
                    },
                },
                'primary_key': 'id_nesreca'
            },
            'oseba': {
                'fields': {
                    'upravna_enota': {
                        'type': 'id',
                        'subtype': 'integer',
                        'ref': {
                            'table': 'upravna_enota',
                            'field': 'id_upravna_enota'
                        }
                    },
                    'id_nesreca': {
                        'type': 'id',
                        'subtype': 'integer',
                        'ref': {
                            'table': 'nesreca',
                            'field': 'id_nesreca'
                        }
                    },
                },
            },
            'upravna_enota': {
                'fields': {
                    'id_upravna_enota': {
                        'type': 'id',
                        'subtype': 'integer'
                    }
                },
                'primary_key': 'id_upravna_enota'
            }
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
                    'id_nesreca': {'sdtype': 'id', 'regex_format': r'\d{30}'}
                }
            },
            'oseba': {
                'columns': {
                    'upravna_enota': {'sdtype': 'id', 'regex_format': r'\d{30}'},
                    'id_nesreca': {'sdtype': 'id', 'regex_format': r'\d{30}'}
                }
            },
            'upravna_enota': {
                'primary_key': 'id_upravna_enota',
                'columns': {'id_upravna_enota': {'sdtype': 'id', 'regex_format': r'\d{30}'}}
            }
        },
        'relationships': [
            {
                'parent_table_name': 'upravna_enota',
                'parent_primary_key': 'id_upravna_enota',
                'child_table_name': 'nesreca',
                'child_foreign_key': 'upravna_enota'
            },
            {
                'parent_table_name': 'upravna_enota',
                'parent_primary_key': 'id_upravna_enota',
                'child_table_name': 'oseba',
                'child_foreign_key': 'upravna_enota'
            },
            {
                'parent_table_name': 'nesreca',
                'parent_primary_key': 'id_nesreca',
                'child_table_name': 'oseba',
                'child_foreign_key': 'id_nesreca'
            }
        ],
        'METADATA_SPEC_VERSION': 'MULTI_TABLE_V1'
    }
    assert new_metadata['METADATA_SPEC_VERSION'] == expected_metadata['METADATA_SPEC_VERSION']
    assert new_metadata['tables'] == expected_metadata['tables']
    for relationship in new_metadata['relationships']:
        assert relationship in expected_metadata['relationships']


def test_detect_from_dataframes():
    """Test the ``detect_from_dataframes`` method."""
    # Setup
    real_data, _ = download_demo(
        modality='multi_table',
        dataset_name='fake_hotels'
    )

    metadata = MultiTableMetadata()

    # Run
    metadata.detect_from_dataframes(real_data)

    # Assert
    metadata.update_column(
        table_name='hotels',
        column_name='city',
        sdtype='categorical',
    )
    metadata.update_column(
        table_name='hotels',
        column_name='state',
        sdtype='categorical',
    )
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
                    'city': {'sdtype': 'categorical'},
                    'state': {'sdtype': 'categorical'},
                    'rating': {'sdtype': 'numerical'},
                    'classification': {'sdtype': 'categorical'}
                },
                'primary_key': 'hotel_id'
            },
            'guests': {
                'columns': {
                    'guest_email': {'sdtype': 'id'},
                    'hotel_id': {'sdtype': 'id'},
                    'has_rewards': {'sdtype': 'categorical'},
                    'room_type': {'sdtype': 'categorical'},
                    'amenities_fee': {'sdtype': 'numerical'},
                    'checkin_date': {'sdtype': 'datetime', 'datetime_format': '%d %b %Y'},
                    'checkout_date': {'sdtype': 'datetime', 'datetime_format': '%d %b %Y'},
                    'room_rate': {'sdtype': 'numerical'},
                    'billing_address': {'sdtype': 'unknown', 'pii': True},
                    'credit_card_number': {'sdtype': 'unknown', 'pii': True}
                },
                'primary_key': 'guest_email'
            }
        },
        'relationships': [
            {
                'parent_table_name': 'hotels',
                'child_table_name': 'guests',
                'parent_primary_key': 'hotel_id',
                'child_foreign_key': 'hotel_id'
            }
        ],
        'METADATA_SPEC_VERSION': 'MULTI_TABLE_V1'
    }
    assert metadata.to_dict() == expected_metadata


def test_detect_from_csvs(tmp_path):
    """Test the ``detect_from_csvs`` method."""
    # Setup
    real_data, _ = download_demo(
        modality='multi_table',
        dataset_name='fake_hotels'
    )

    metadata = MultiTableMetadata()

    for table_name, dataframe in real_data.items():
        csv_path = tmp_path / f'{table_name}.csv'
        dataframe.to_csv(csv_path, index=False)

    # Run
    metadata.detect_from_csvs(folder_name=tmp_path)

    # Assert
    metadata.update_column(
        table_name='hotels',
        column_name='city',
        sdtype='categorical',
    )
    metadata.update_column(
        table_name='hotels',
        column_name='state',
        sdtype='categorical',
    )
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
                    'city': {'sdtype': 'categorical'},
                    'state': {'sdtype': 'categorical'},
                    'rating': {'sdtype': 'numerical'},
                    'classification': {'sdtype': 'categorical'}
                },
                'primary_key': 'hotel_id'
            },
            'guests': {
                'columns': {
                    'guest_email': {'sdtype': 'id'},
                    'hotel_id': {'sdtype': 'id'},
                    'has_rewards': {'sdtype': 'categorical'},
                    'room_type': {'sdtype': 'categorical'},
                    'amenities_fee': {'sdtype': 'numerical'},
                    'checkin_date': {'sdtype': 'datetime', 'datetime_format': '%d %b %Y'},
                    'checkout_date': {'sdtype': 'datetime', 'datetime_format': '%d %b %Y'},
                    'room_rate': {'sdtype': 'numerical'},
                    'billing_address': {'sdtype': 'unknown', 'pii': True},
                    'credit_card_number': {'sdtype': 'unknown', 'pii': True}
                },
                'primary_key': 'guest_email'
            }
        },
        'relationships': [
            {
                'parent_table_name': 'hotels',
                'child_table_name': 'guests',
                'parent_primary_key': 'hotel_id',
                'child_foreign_key': 'hotel_id'
            }
        ],
        'METADATA_SPEC_VERSION': 'MULTI_TABLE_V1'
    }

    assert metadata.to_dict() == expected_metadata


def test_detect_table_from_csv(tmp_path):
    """Test the ``detect_table_from_csv`` method."""
    # Setup
    real_data, _ = download_demo(
        modality='multi_table',
        dataset_name='fake_hotels'
    )

    metadata = MultiTableMetadata()

    for table_name, dataframe in real_data.items():
        csv_path = tmp_path / f'{table_name}.csv'
        dataframe.to_csv(csv_path, index=False)

    # Run
    metadata.detect_table_from_csv('hotels', tmp_path / 'hotels.csv')

    # Assert
    metadata.update_column(
        table_name='hotels',
        column_name='city',
        sdtype='categorical',
    )
    metadata.update_column(
        table_name='hotels',
        column_name='state',
        sdtype='categorical',
    )
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
                    'city': {'sdtype': 'categorical'},
                    'state': {'sdtype': 'categorical'},
                    'rating': {'sdtype': 'numerical'},
                    'classification': {'sdtype': 'categorical'}
                },
                'primary_key': 'hotel_id'
            }
        },
        'relationships': [],
        'METADATA_SPEC_VERSION': 'MULTI_TABLE_V1'
    }

    assert metadata.to_dict() == expected_metadata
