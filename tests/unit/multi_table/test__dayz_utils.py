import json
from unittest.mock import patch

import pandas as pd

from sdv.metadata import Metadata
from sdv.multi_table._dayz_utils import (
    create_parameters_multi_table,
    detect_relationship_parameters,
)


def test_detect_relationship_parameters():
    """Test the `detect_relationship_parameters` method."""
    # Setup
    parent_data = pd.DataFrame({'parent_id': [1, 2, 3, 4, 5]})
    child_data = pd.DataFrame({
        'child_id': [10, 11, 12, 13, 14, 15, 16],
        'parent_id': [1, 1, 2, 2, 2, 3, None],
    })
    data = {'parent': parent_data, 'child': child_data}
    metadata_dict = {
        'tables': {
            'parent': {'columns': {'parent_id': {'sdtype': 'id'}}, 'primary_key': 'parent_id'},
            'child': {
                'columns': {'child_id': {'sdtype': 'id'}, 'parent_id': {'sdtype': 'id'}},
                'primary_key': 'child_id',
            },
        },
        'relationships': [
            {
                'parent_table_name': 'parent',
                'child_table_name': 'child',
                'parent_primary_key': 'parent_id',
                'child_foreign_key': 'parent_id',
            }
        ],
    }
    metadata = Metadata.load_from_dict(metadata_dict)

    # Run
    result = detect_relationship_parameters(data, metadata)

    # Assert
    expected = [
        {
            'parent_table_name': 'parent',
            'child_table_name': 'child',
            'parent_primary_key': 'parent_id',
            'child_foreign_key': 'parent_id',
            'min_cardinality': 0,
            'max_cardinality': 3,
        }
    ]
    assert result == expected


@patch('sdv.multi_table._dayz_utils.detect_relationship_parameters')
@patch('sdv.multi_table._dayz_utils.create_parameters')
def test_create_parameters_multi_table(mock_create_parameters, mock_detect_relationship, tmp_path):
    """Test the `create_parameters_multi_table` method."""
    # Setup
    data = pd.DataFrame()
    metadata = Metadata()
    output_filename = str(tmp_path / 'output.json')
    mock_detect_relationship.return_value = {
        '["parent_table", "child_table", "parent_pk", "child_fk"]': {
            'min_cardinality': 0,
            'max_cardinality': 10,
        }
    }
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
    }

    # Run
    result = create_parameters_multi_table(data, metadata, output_filename)

    # Assert
    mock_create_parameters.assert_called_once_with(data, metadata, None)
    mock_detect_relationship.assert_called_once_with(data, metadata)
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
        'relationships': {
            '["parent_table", "child_table", "parent_pk", "child_fk"]': {
                'min_cardinality': 0,
                'max_cardinality': 10,
            }
        },
    }
    assert result == mock_create_parameters.return_value
    with open(output_filename) as f:
        output = json.load(f)

    assert output == result
