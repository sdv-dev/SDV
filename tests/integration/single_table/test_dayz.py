from sdv.metadata import Metadata
from sdv.single_table.dayz import DayZSynthesizer


def test_validate_parameters():
    """Test validating DayZ parameters."""
    # Setup
    metadata = Metadata.load_from_dict({
        'tables': {
            'table': {
                'columns': {
                    'id': {'sdtype': 'id'},
                    'numerical': {'sdtype': 'numerical'},
                    'datetime': {'sdtype': 'datetime', 'datetime_format': '%d %b %Y'},
                    'categorical': {'sdtype': 'categorical'},
                    'pii': {'sdtype': 'ssn'},
                    'extra_column': {'sdtype': 'numerical'},
                }
            }
        }
    })

    dayz_parameters = {
        'DAYZ_SPEC_VERSION': 'V1',
        'tables': {
            'table': {
                'num_rows': 250,
                'columns': {
                    'numerical': {'min_value': 18, 'missing_values_proportion': 0.15},
                    'datetime': {'start_timestamp': '01 Apr 2018', 'end_timestamp': '08 Dec 2024'},
                    'pii': {'missing_values_proportion': 0.4},
                },
            }
        },
    }

    # Run and Assert
    DayZSynthesizer.validate_parameters(metadata, dayz_parameters)
