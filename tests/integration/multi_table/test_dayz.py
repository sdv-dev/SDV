from sdv.metadata import Metadata
from sdv.multi_table.dayz import DayZSynthesizer


def test_validate_parameters():
    """Test validating DayZ parameters."""
    # Setup
    metadata = Metadata.load_from_dict({
        'tables': {
            'parent': {
                'columns': {
                    'id': {'sdtype': 'id'},
                    'numerical': {'sdtype': 'numerical'},
                    'datetime': {'sdtype': 'datetime', 'datetime_format': '%d %b %Y'},
                    'categorical': {'sdtype': 'categorical'},
                    'pii': {'sdtype': 'ssn'},
                    'extra_column': {'sdtype': 'numerical'},
                },
                'primary_key': 'id',
            },
            'child': {
                'columns': {
                    'child_fk': {'sdtype': 'id'},
                    'numerical': {'sdtype': 'numerical'},
                    'pii': {'sdtype': 'ssn'},
                }
            }
        },
        'relationships': [
            {
                'parent_table_name': 'parent',
                'child_table_name': 'child',
                'parent_primary_key': 'id',
                'child_foreign_key': 'child_fk',
            }
        ]
    })

    dayz_parameters = {
        'DAYZ_SPEC_VERSION': 'V1',
        'tables': {
            'parent': {
                'num_rows': 250,
                'columns': {
                    'numerical': {'min_value': 18, 'missing_values_proportion': 0.15},
                    'datetime': {'start_timestamp': '01 Apr 2018', 'end_timestamp': '08 Dec 2024'},
                    'pii': {'missing_values_proportion': 0.4},
                },
            }
        },
        'relationships': [
            {
                'parent_table_name': 'parent',
                'child_table_name': 'child',
                'parent_primary_key': 'id',
                'child_foreign_key': 'child_fk',
                'min_cardinality': 1,
                'max_cardinality': 10,
            }
        ]
    }

    # Run and Assert
    DayZSynthesizer.validate_parameters(metadata, dayz_parameters)
