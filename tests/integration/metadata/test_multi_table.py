"""Integration tests for Multi Table Metadata."""

import json

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
