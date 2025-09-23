"""Integration tests for DayZ parameter detection."""

from sdv.datasets.demo import download_demo
from sdv.single_table import DayZSynthesizer


class TestDayZSynthesizer:
    def test_create_parameters_end_to_end(self):
        """Test the `create_parameters` method end to end."""
        # Setup
        data, metadata = download_demo(modality='single_table', dataset_name='fake_hotel_guests')

        # Run
        parameters = DayZSynthesizer.create_parameters(data, metadata)

        # Assert
        expected_results = {
            'DAYZ_SPEC_VERSION': 'V1',
            'tables': {
                'fake_hotel_guests': {
                    'num_rows': 500,
                    'columns': {
                        'guest_email': {'missing_values_proportion': 0.0},
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
                            'max_value': 48.12,
                            'missing_values_proportion': 0.09,
                        },
                        'checkin_date': {
                            'start_timestamp': '05 Jan 2020',
                            'end_timestamp': '07 Jan 2021',
                            'missing_values_proportion': 0.0,
                        },
                        'checkout_date': {
                            'start_timestamp': '07 Jan 2020',
                            'end_timestamp': '08 Jan 2021',
                            'missing_values_proportion': 0.04,
                        },
                        'room_rate': {
                            'num_decimal_digits': 2,
                            'min_value': 83.8,
                            'max_value': 424.84,
                            'missing_values_proportion': 0.0,
                        },
                        'billing_address': {'missing_values_proportion': 0.0},
                        'credit_card_number': {'missing_values_proportion': 0.0},
                    },
                },
            },
        }
        assert parameters == expected_results
