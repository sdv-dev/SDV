"""Integration tests for DayZ parameter detection."""

import pandas as pd
import pytest

from sdv.datasets.demo import download_demo
from sdv.metadata import Metadata
from sdv.multi_table import DayZSynthesizer


class TestDayZSynthesizer:
    def test_create_parameters_end_to_end(self):
        """Test the `create_parameters` method end to end."""
        # Setup
        data, metadata = download_demo(modality='multi_table', dataset_name='fake_hotels')

        # Run
        parameters = DayZSynthesizer.create_parameters(data, metadata)

        # Assert
        expected_results = {
            'DAYZ_SPEC_VERSION': 'V1',
            'tables': {
                'guests': {
                    'num_rows': 658,
                    'columns': {
                        'guest_email': {'missing_values_proportion': 0.0},
                        'hotel_id': {'missing_values_proportion': 0.0},
                        'has_rewards': {
                            'missing_values_proportion': 0.0,
                        },
                        'room_type': {
                            'category_values': ['BASIC', 'DELUXE', 'SUITE'],
                            'missing_values_proportion': 0.0,
                        },
                        'amenities_fee': {
                            'num_decimal_digits': 2,
                            'min_value': 0.0,
                            'max_value': 46.64,
                            'missing_values_proportion': 0.07598784194528875,
                        },
                        'checkin_date': {
                            'start_timestamp': '03 Jan 2020',
                            'end_timestamp': '05 Jan 2021',
                            'missing_values_proportion': 0.0,
                        },
                        'checkout_date': {
                            'start_timestamp': '04 Jan 2020',
                            'end_timestamp': '07 Jan 2021',
                            'missing_values_proportion': 0.04559270516717325,
                        },
                        'room_rate': {
                            'num_decimal_digits': 2,
                            'min_value': 48.33,
                            'max_value': 481.61,
                            'missing_values_proportion': 0.0,
                        },
                        'billing_address': {'missing_values_proportion': 0.0},
                        'credit_card_number': {'missing_values_proportion': 0.0},
                    },
                },
                'hotels': {
                    'num_rows': 10,
                    'columns': {
                        'hotel_id': {'missing_values_proportion': 0.0},
                        'city': {
                            'category_values': [
                                'Boston',
                                'San Francisco',
                                'New York City',
                                'Austin',
                                'Los Angeles',
                            ],
                            'missing_values_proportion': 0.0,
                        },
                        'state': {
                            'category_values': [
                                'Massachusetts',
                                'Massachuesetts',
                                'California',
                                'New York',
                                'Texas',
                            ],
                            'missing_values_proportion': 0.0,
                        },
                        'rating': {
                            'num_decimal_digits': 1,
                            'min_value': 3.7,
                            'max_value': 4.9,
                            'missing_values_proportion': 0.1,
                        },
                        'classification': {
                            'category_values': ['RESORT', 'CHAIN', 'MOTEL'],
                            'missing_values_proportion': 0.0,
                        },
                    },
                },
            },
            'relationships': [
                {
                    'parent_table_name': 'hotels',
                    'child_table_name': 'guests',
                    'parent_primary_key': 'hotel_id',
                    'child_foreign_key': 'hotel_id',
                    'min_cardinality': 15,
                    'max_cardinality': 137,
                },
            ],
        }
        assert parameters == expected_results

    def test_validate_parameters(self):
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
                },
            },
            'relationships': [
                {
                    'parent_table_name': 'parent',
                    'child_table_name': 'child',
                    'parent_primary_key': 'id',
                    'child_foreign_key': 'child_fk',
                }
            ],
        })

        dayz_parameters = {
            'DAYZ_SPEC_VERSION': 'V1',
            'tables': {
                'parent': {
                    'num_rows': 250,
                    'columns': {
                        'numerical': {
                            'min_value': 18,
                            'max_value': 38,
                            'missing_values_proportion': 0.15,
                        },
                        'datetime': {
                            'start_timestamp': '01 Apr 2018',
                            'end_timestamp': '08 Dec 2024',
                        },
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
            ],
        }

        # Run and Assert
        DayZSynthesizer.validate_parameters(metadata, dayz_parameters)

    def test_create_parameters_empty_data(self):
        """Test creating parameters with empty data."""
        # Setup
        data = {}
        metadata = Metadata()

        # Run and Assert
        with pytest.raises(ValueError, match='Data is empty'):
            DayZSynthesizer.create_parameters(data, metadata)

    def test_create_parameters_empty_metadata(self):
        """Test creating parameters with empty metadata."""
        # Setup
        data = {'table': pd.DataFrame({'col1': [1, 2, 3]})}
        metadata = Metadata()

        # Run and Assert
        with pytest.raises(ValueError, match='Metadata is empty'):
            DayZSynthesizer.create_parameters(data, metadata)
