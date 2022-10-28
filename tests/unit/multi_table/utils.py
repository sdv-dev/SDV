from sdv.metadata.multi_table import MultiTableMetadata


def get_multi_table_metadata():
    dict_metadata = {
        'tables': {
            'nesreca': {
                'primary_key': 'id_nesreca',
                'alternate_keys': ['upravna_enota'],
                'columns': {
                    'upravna_enota': {'sdtype': 'numerical'},
                    'id_nesreca': {'sdtype': 'numerical'}
                }
            },
            'oseba': {
                'alternate_keys': ['upravna_enota', 'id_nesreca'],
                'columns': {
                    'upravna_enota': {'sdtype': 'numerical'},
                    'id_nesreca': {'sdtype': 'numerical'}
                }
            },
            'upravna_enota': {
                'primary_key': 'id_upravna_enota',
                'columns': {'id_upravna_enota': {'sdtype': 'numerical'}}
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
        'SCHEMA_VERSION': 'MULTI_TABLE_V1'
    }

    return MultiTableMetadata._load_from_dict(dict_metadata)
