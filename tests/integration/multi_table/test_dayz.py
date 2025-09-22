"""Integration tests for DayZ parameter detection."""

from sdv.datasets.demo import download_demo
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
                            'category_values': [False, True],
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
            'relationships': {
                '["hotels", "guests", "hotel_id", "hotel_id"]': {
                    'min_cardinality': 15,
                    'max_cardinality': 137,
                }
            },
        }
        assert parameters == expected_results
