import json
import re
from unittest.mock import call, patch

import pandas as pd
import pytest

from sdv.datasets.demo import download_demo
from sdv.errors import SynthesizerInputError, SynthesizerProcessingError
from sdv.metadata import Metadata
from sdv.multi_table.dayz import (
    DayZSynthesizer,
    _detect_relationship_parameters,
    _validate_cardinality,
    _validate_parameters,
    _validate_relationship_parameters,
    _validate_relationship_structure,
    create_parameters_multi_table,
)


@pytest.fixture
def metadata():
    return Metadata.load_from_dict({
        'tables': {
            'grandparent': {'columns': {'id': {'sdtype': 'id'}}, 'primary_key': 'id'},
            'parent': {
                'columns': {
                    'id': {'sdtype': 'id'},
                    'parent_fk': {'sdtype': 'id'},
                    'numerical': {'sdtype': 'numerical'},
                    'datetime': {'sdtype': 'datetime', 'datetime_format': '%d %b %Y'},
                    'categorical': {'sdtype': 'categorical'},
                    'pii': {'sdtype': 'ssn'},
                    'extra_column': {'sdtype': 'numerical'},
                },
                'primary_key': 'id',
            },
            'child': {'columns': {'child_fk': {'sdtype': 'id'}}},
        },
        'relationships': [
            {
                'parent_table_name': 'parent',
                'child_table_name': 'child',
                'parent_primary_key': 'id',
                'child_foreign_key': 'child_fk',
            },
            {
                'parent_table_name': 'grandparent',
                'child_table_name': 'parent',
                'parent_primary_key': 'id',
                'child_foreign_key': 'parent_fk',
            },
        ],
    })


def test__detect_relationship_parameters():
    """Test the `_detect_relationship_parameters` method."""
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
    result = _detect_relationship_parameters(data, metadata)

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


@patch('sdv.multi_table.dayz._detect_relationship_parameters')
@patch('sdv.multi_table.dayz.create_parameters')
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


def test__validate_relationship_structure():
    """Test validating the relationship parameters structure."""
    # Setup
    bad_relationships_value = {'relationships': None}
    bad_relationship_unknown_key = {'bad_key': None}
    bad_relationship_missing_key = {
        'parent_table_name': 'parent',
        'child_table_name': 'child',
        'child_foreign_key': 'child_fk',
    }

    relationship = {
        'parent_table_name': 'parent',
        'parent_primary_key': 'parent_pk',
        'child_table_name': 'child',
        'child_foreign_key': 'child_fk',
    }
    bad_min_cardinality = {**relationship, 'min_cardinality': -5}
    bad_max_cardinality = {**relationship, 'max_cardinality': 0}
    bad_min_max_cardinality = {**relationship, 'min_cardinality': 5, 'max_cardinality': 3}

    # Run and Assert
    expected_bad_relationships_value_msg = re.escape(
        "The 'relationships' parameter value must be a list of dictionaries."
    )
    with pytest.raises(SynthesizerProcessingError, match=expected_bad_relationships_value_msg):
        _validate_relationship_structure(bad_relationships_value)

    expected_unknown_key_msg = re.escape(
        "Relationship parameter contains unexpected key(s) 'bad_key'."
    )
    with pytest.raises(SynthesizerProcessingError, match=expected_unknown_key_msg):
        _validate_relationship_structure({'relationships': [bad_relationship_unknown_key]})

    expected_missing_key_msg = re.escape(
        "Relationship parameter missing required key(s) 'parent_primary_key'."
    )
    with pytest.raises(SynthesizerProcessingError, match=expected_missing_key_msg):
        _validate_relationship_structure({'relationships': [bad_relationship_missing_key]})

    expected_bad_min_cardinality_msg = re.escape(
        "Invalid 'min_cardinality' parameter (-5). The 'min_cardinality' parameter "
        'must be an integer greater than or equal to zero.'
    )
    with pytest.raises(SynthesizerProcessingError, match=expected_bad_min_cardinality_msg):
        _validate_relationship_structure({'relationships': [bad_min_cardinality]})

    expected_bad_max_cardinality_msg = re.escape(
        "Invalid 'max_cardinality' parameter (0). The 'max_cardinality' parameter "
        'must be an integer greater than zero.'
    )
    with pytest.raises(SynthesizerProcessingError, match=expected_bad_max_cardinality_msg):
        _validate_relationship_structure({'relationships': [bad_max_cardinality]})

    expected_bad_min_max_cardinality_msg = re.escape(
        "Invalid cardinality parameters, the 'min_cardinality' must be less than "
        "or equal to the 'max_cardinality'."
    )
    with pytest.raises(SynthesizerProcessingError, match=expected_bad_min_max_cardinality_msg):
        _validate_relationship_structure({'relationships': [bad_min_max_cardinality]})


def test__validate_cardinality():
    """Test validating relationship cardinality."""
    # Setup
    relationship_parameters = {
        'parent_table_name': 'parent',
        'child_table_name': 'child',
        'parent_primary_key': 'parent_id',
        'child_foreign_key': 'child_fk',
        'min_cardinality': 1,
        'max_cardinality': 5,
    }

    # Run and Assert
    expected_min_cardinality_msg = re.escape(
        f'Invalid cardinality parameters for relationship {relationship_parameters}. '
        'Minimum cardinality requires child table to be at least 100 rows.'
    )
    with pytest.raises(SynthesizerProcessingError, match=expected_min_cardinality_msg):
        _validate_cardinality(relationship_parameters, parent_num_rows=100, child_num_rows=50)

    expected_max_cardinality_msg = re.escape(
        f'Invalid cardinality parameters for relationship {relationship_parameters}. '
        f'Maximum cardinality requires child table to be less than 100 rows.'
    )
    with pytest.raises(SynthesizerProcessingError, match=expected_max_cardinality_msg):
        _validate_cardinality(relationship_parameters, parent_num_rows=20, child_num_rows=200)

    del relationship_parameters['max_cardinality']
    del relationship_parameters['min_cardinality']
    _validate_cardinality(relationship_parameters, parent_num_rows=10, child_num_rows=100)


@patch('sdv.multi_table.dayz._validate_cardinality')
def test__validate_relationship_parameters(mock__validate_cardinality, metadata):
    """Test validating relationship parameters."""
    # Setup
    bad_relationship = {
        'parent_table_name': 'grandparent',
        'child_table_name': 'child',
        'parent_primary_key': 'id',
        'child_foreign_key': 'child_fk',
    }
    bad_relationship_parameters = {
        **bad_relationship,
        'min_cardinality': 5,
    }
    relationship = {
        'parent_table_name': 'parent',
        'child_table_name': 'child',
        'parent_primary_key': 'id',
        'child_foreign_key': 'child_fk',
    }
    relationship_parameters = {
        **relationship,
        'max_cardinality': 5,
    }
    dayz_parameters = {
        'tables': {'parent': {'num_rows': 500}},
        'relationships': [relationship_parameters],
    }

    # Run and Assert
    expected_bad_relationship_msg = re.escape(
        'Invalid relationship parameter: '
        f'relationship {bad_relationship} does not exist in the metadata.'
    )
    with pytest.raises(SynthesizerProcessingError, match=expected_bad_relationship_msg):
        _validate_relationship_parameters(
            metadata, {'relationships': [bad_relationship_parameters]}
        )

    expected_duplicate_error_msg = re.escape(
        'Invalid relationship parameter: '
        f'multiple entries for relationship {relationship} in parameters.'
    )
    with pytest.raises(SynthesizerProcessingError, match=expected_duplicate_error_msg):
        _validate_relationship_parameters(
            metadata, {'relationships': [relationship_parameters, relationship_parameters]}
        )

    _validate_relationship_parameters(metadata, dayz_parameters)

    # Assert
    mock__validate_cardinality.assert_has_calls([
        call(relationship_parameters, None, None),
        call(relationship_parameters, 500, None),
    ])


@patch('sdv.multi_table.dayz._validate_parameter_structure')
@patch('sdv.multi_table.dayz._validate_relationship_structure')
@patch('sdv.multi_table.dayz._validate_tables_parameter')
@patch('sdv.multi_table.dayz._validate_relationship_parameters')
def test__validate_parameters(
    mock__validate_relationship_parameters,
    mock__validate_tables_parameter,
    mock__validate_relationship_structure,
    mock__validate_parameter_structure,
    metadata,
):
    """Test ``_validate_parameters`` validates relationship and parameters."""
    # Setup
    dayz_parameters = {
        'tables': {
            'parent': {
                'columns': {
                    'numerical': {'min_value': 100, 'max_value': 1000},
                    'categorical': {'category_values': ['A', 'B', 'C']},
                },
                'num_rows': 500,
            }
        },
        'relationships': [
            {
                'parent_table_name': 'parent',
                'parent_primary_key': 'id',
                'child_table_name': 'child',
                'child_foreign_key': 'child_fk',
                'max_cardinality': 5,
            }
        ],
    }
    # Run
    _validate_parameters(metadata, dayz_parameters)

    # Assert
    mock__validate_parameter_structure.assert_called_once_with(dayz_parameters)
    mock__validate_relationship_structure.assert_called_once_with(dayz_parameters)
    mock__validate_tables_parameter.assert_called_once_with(metadata, dayz_parameters)
    mock__validate_relationship_parameters.assert_called_once_with(metadata, dayz_parameters)


class TestDayZSynthesizer:
    def test__init__(self):
        """Test the `__init__` method."""
        # Setup
        metadata = Metadata()
        expected_error = re.escape(
            "Only the 'DayZSynthesizer.create_parameters' and the "
            'DayZSynthesizer.validate_parameters methods are an SDV public feature. To '
            'define and use a DayZSynthesizer object you must have SDV-Enterprise.'
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

    @patch('sdv.multi_table.dayz._validate_parameters')
    def test_validate_parameters(self, mock__validate_parameters, metadata):
        """Test the ``validate_parameters`` method."""
        # Setup
        dayz_parameters = {
            'tables': {
                'parent': {
                    'columns': {
                        'numerical': {'min_value': 100, 'max_value': 1000},
                        'categorical': {'category_values': ['A', 'B', 'C']},
                    },
                    'num_rows': 500,
                }
            },
            'relationships': [
                {
                    'parent_table_name': 'parent',
                    'parent_primary_key': 'id',
                    'child_table_name': 'child',
                    'child_foreign_key': 'child_fk',
                    'max_cardinality': 5,
                }
            ],
        }
        # Run
        DayZSynthesizer.validate_parameters(metadata, dayz_parameters)

        # Assert
        mock__validate_parameters.assert_called_once_with(metadata, dayz_parameters)

    def test__validate_relationships_is_list_of_dicts(self, metadata):
        """Test that 'relationships' must be a list of dicts."""
        # Run and Assert
        expected_msg = re.escape(
            "The 'relationships' parameter value must be a list of dictionaries."
        )
        with pytest.raises(SynthesizerProcessingError, match=expected_msg):
            DayZSynthesizer.validate_parameters(metadata, {'relationships': {'a', 'b', 'c'}})

        with pytest.raises(SynthesizerProcessingError, match=expected_msg):
            DayZSynthesizer.validate_parameters(metadata, {'relationships': ['a', 'b', 'c']})

    def test__validate_min_cardinality_allows_zero(self):
        """Test that min_cardinality=0 is allowed and does not raise."""
        # Setup
        data, metadata = download_demo('multi_table', 'financial_v1')
        dayz_parameters = DayZSynthesizer.create_parameters(data, metadata)
        dayz_parameters['relationships'] = [
            {
                'parent_table_name': 'district',
                'parent_primary_key': 'district_id',
                'child_table_name': 'account',
                'child_foreign_key': 'district_id',
                'min_cardinality': 0,
            }
        ]

        # Run
        DayZSynthesizer.validate_parameters(metadata, dayz_parameters)
