import json
from unittest.mock import Mock, patch

import pandas as pd

from sdv.metadata import Metadata
from sdv.single_table._dayz_utils import (
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
        'pk': [0, 1, 2, 3],
        'num_col': [1.0, 2.5, 3.0, None],
        'cat_col': ['A', 'B', 'A', None],
        'date_col': ['2020-01-01', '2020-01-02', None, None],
        'date_col_2': ['2020 Jan 01', '2020 Jan 02', '2020 Jan 03', None],
        'alt_key': ['id0', 'id1', 'id2', 'id3'],
    })
    metadata = Metadata.load_from_dict({
        'tables': {
            'table_name': {
                'columns': {
                    'pk': {'sdtype': 'id'},
                    'num_col': {'sdtype': 'numerical'},
                    'cat_col': {'sdtype': 'categorical'},
                    'date_col': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
                    'date_col_2': {'sdtype': 'datetime'},
                    'alt_key': {'sdtype': 'ssn'},
                },
                'primary_key': 'pk',
                'alternate_keys': ['alt_key'],
            }
        }
    })
    # Run
    result = detect_column_parameters(data, metadata, 'table_name')

    # Assert
    assert result == {
        'columns': {
            'pk': {'missing_values_proportion': 0.0},
            'num_col': {
                'num_decimal_digits': 1,
                'min_value': 1.0,
                'max_value': 3.0,
                'missing_values_proportion': 0.25,
            },
            'cat_col': {
                'category_values': ['A', 'B'],
                'missing_values_proportion': 0.25,
            },
            'date_col': {
                'start_timestamp': '2020-01-01',
                'end_timestamp': '2020-01-02',
                'missing_values_proportion': 0.5,
            },
            'date_col_2': {
                'start_timestamp': '2020-01-01 00:00:00',
                'end_timestamp': '2020-01-03 00:00:00',
                'missing_values_proportion': 0.25,
            },
            'alt_key': {'missing_values_proportion': 0.0},
        }
    }


@patch('sdv.single_table._dayz_utils.detect_column_parameters')
@patch('sdv.single_table._dayz_utils.detect_table_parameters')
def test_create_parameters(mock_detect_table, mock_detect_column, tmp_path):
    """Test the `create_parameters` method."""
    # Setup
    output_filename = tmp_path / 'output.json'
    mock_detect_table.return_value = {'num_rows': 100}
    mock_detect_column.return_value = {
        'columns': {
            'col1': {'missing_values_proportion': 0.1},
            'col2': {'missing_values_proportion': 0.2},
        }
    }

    data = pd.DataFrame()
    metadata = Mock()
    metadata._get_single_table_name.return_value = 'table_name'

    # Run
    result = create_parameters(data, metadata, output_filename=output_filename)

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
                    'col1': {'missing_values_proportion': 0.1},
                    'col2': {'missing_values_proportion': 0.2},
                },
            }
        },
    }
    with open(output_filename, 'r') as f:
        output = json.load(f)

    assert output == result
