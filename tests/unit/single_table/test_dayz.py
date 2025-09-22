import re
from unittest.mock import patch

import pandas as pd
import pytest

from sdv.errors import SynthesizerInputError
from sdv.metadata import Metadata
from sdv.single_table.dayz import DayZSynthesizer


class TestDayZSynthesizer:
    def test__init__(self):
        """Test the `__init__` method."""
        # Setup
        metadata = Metadata()
        expected_error = re.escape(
            "Only the 'DayZSynthesizer.create_parameters' is a SDV public feature. "
            'To define and use and use a DayZSynthesizer object you must have SDV-Enterprise.'
        )

        # Run and Assert
        with pytest.raises(SynthesizerInputError, match=expected_error):
            DayZSynthesizer(metadata, locales=['es_ES'])

    @patch('sdv.single_table.dayz.create_parameters')
    def test_create_parameters(self, mock_create):
        """Test the `create_parameters` method."""
        # Setup
        data = pd.DataFrame()
        metadata = Metadata()
        mock_create.return_value = {
            'DAYZ_SPEC_VERSION': 'V1',
            'tables': {
                'guests': {
                    'num_rows': 658,
                    'columns': {
                        'guest_email': {'missing_values_proportion': 0.0},
                        'room_type': {
                            'category_values': ['BASIC', 'DELUXE', 'SUITE'],
                            'missing_values_proportion': 0.0,
                        },
                        'numerical_feature': {
                            'missing_values_proportion': 0.0,
                            'num_decimal_digits': 2,
                            'min_value': 0,
                            'max_value': 100,
                        },
                    },
                },
            },
        }

        # Run
        result = DayZSynthesizer.create_parameters(data, metadata, 'output_filename')

        # Assert
        mock_create.assert_called_once_with(data, metadata, 'output_filename')
        assert result == mock_create.return_value
