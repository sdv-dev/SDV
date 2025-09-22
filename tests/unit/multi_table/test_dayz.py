import re
from unittest.mock import patch

import pandas as pd
import pytest

from sdv.errors import SynthesizerInputError
from sdv.metadata import Metadata
from sdv.multi_table.dayz import DayZSynthesizer


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

    @patch('sdv.multi_table.dayz.create_parameters_multi_table')
    def test_create_parameters(self, mock_create_parameters):
        # Setup
        data = pd.DataFrame()
        metadata = Metadata()
        mock_create_parameters.return_value = {
            'DAYZ_SPEC_VERSION': 'V1',
            'tables': {
                'table_name': {
                    'num_rows': 100,
                    'columns': {
                        'col1': {'missing_values_proportion': 0.1},
                        'col2': {'missing_values_proportion': 0.2},
                    },
                }
            },
            'relationships': {
                ('parent_table', 'child_table', 'parent_pk', 'child_fk'): {
                    'min_cardinality': 0,
                    'max_cardinality': 10,
                }
            },
        }

        # Run
        result = DayZSynthesizer.create_parameters(data, metadata, 'output_filename')

        # Assert
        mock_create_parameters.assert_called_once_with(data, metadata, 'output_filename')
        assert result == mock_create_parameters.return_value
