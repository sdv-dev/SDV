import json
from unittest.mock import Mock, patch

import pandas as pd

from sdv.metadata import Metadata
from sdv.single_table.dayz import (
    DayZSynthesizer,
    create_parameters,
    detect_column_parameters,
    detect_table_parameters,
)


def test_detect_table_parameters():
    """Test the `detect_table_parameters` method."""
    # Setup
    data = pd.DataFrame(index=range(10))

    # Run
    result = detect_table_parameters(data)

    # Assert
    assert result == {'num_rows': 10}


def test_detect_column_parameter():
    """Test the `detect_column_parameters` method."""
    # Setup
    data = pd.DataFrame({
        'num_col': [1.0, 2.5, 3.0, None],
        'cat_col': ['A', 'B', 'A', None],
        'date_col': ['2020-01-01', '2020-01-02', None, None],
    })
    metadata = Metadata.load_from_dict({
        'tables': {
            'table_name': {
                'columns': {
                    'num_col': {'sdtype': 'numerical'},
                    'cat_col': {'sdtype': 'categorical'},
                    'date_col': {'sdtype': 'datetime'},
                }
            }
        }
    })
    # Run
    result = detect_column_parameters(data, metadata, 'table_name')

    # Assert
    assert result == {
        'columns': {
            'num_col': {
                'num_decimal_digits': 1,
                'min_value': 1.0,
                'max_value': 3.0,
                'missing_value_proportion': 0.25,
            },
            'cat_col': {
                'category_values': ['A', 'B'],
                'missing_value_proportion': 0.25,
            },
            'date_col': {
                'start_timestamp': pd.Timestamp('2020-01-01'),
                'end_timestamp': pd.Timestamp('2020-01-02'),
                'missing_value_proportion': 0.5,
            },
        }
    }


@patch('sdv.single_table.dayz.detect_column_parameters')
@patch('sdv.single_table.dayz.detect_table_parameters')
def test_create_parameters(mock_detect_table, mock_detect_column):
    """Test the `create_parameters` method."""
    # Setup
    mock_detect_table.return_value = {'num_rows': 100}
    mock_detect_column.return_value = {
        'columns': {
            'col1': {'missing_value_proportion': 0.1},
            'col2': {'missing_value_proportion': 0.2},
        }
    }

    data = pd.DataFrame()
    metadata = Mock()
    metadata._get_single_table_name.return_value = 'table_name'

    # Run
    result = create_parameters(data, metadata)

    # Assert
    metadata.validate.assert_called_once()
    metadata.validate_data.assert_called_once_with({'table_name': data})
    mock_detect_table.assert_called_once_with(data)
    mock_detect_column.assert_called_once_with(data, metadata, 'table_name')
    assert result == {
        'DAYZ_SPEC_VERSION': 'V1',
        'tables': {
            'table_name': {
                'num_rows': 100,
                'columns': {
                    'col1': {'missing_value_proportion': 0.1},
                    'col2': {'missing_value_proportion': 0.2},
                },
            }
        },
    }


class TestDayZSynthesizer:
    @patch('sdv.single_table.dayz.create_parameters')
    def test_create_parameters(self, mock_create, tmp_path):
        """Test the `create_parameters` method."""
        # Setup
        data = pd.DataFrame()
        metadata = Metadata()
        output_filename = tmp_path / 'output.json'
        mock_create.return_value = {
            'DAYZ_SPEC_VERSION': 'V1',
            'tables': {
                'guests': {
                    'num_rows': 658,
                    'columns': {
                        'guest_email': {'missing_value_proportion': 0.0},
                        'room_type': {
                            'category_values': ['BASIC', 'DELUXE', 'SUITE'],
                            'missing_value_proportion': 0.0,
                        },
                        'numerical_feature': {
                            'missing_value_proportion': 0.0,
                            'num_decimal_digits': 2,
                            'min_value': 0,
                            'max_value': 100,
                        },
                    },
                },
            },
        }

        # Run
        result = DayZSynthesizer.create_parameters(
            data, metadata, output_filename=str(output_filename)
        )

        # Assert
        mock_create.assert_called_once_with(data, metadata)
        assert result == mock_create.return_value
        with open(output_filename, 'r') as f:
            output = json.load(f)

        assert output == mock_create.return_value
