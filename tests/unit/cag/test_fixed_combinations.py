"""Unit tests for FixedCombinations CAG pattern."""

import re
from unittest.mock import call, patch

import pandas as pd
import pytest

from sdv.cag import FixedCombinations
from sdv.cag._errors import PatternNotMetError
from sdv.metadata import Metadata


class TestFixedCombinations:
    def test___init__(self):
        """Test the ``__init__`` method."""
        # Setup
        column_names = ['col1', 'col2', 'col3']
        table_name = 'table'

        # Run
        instance = FixedCombinations(column_names, table_name=table_name)

        # Assert
        assert instance.column_names == column_names
        assert instance.table_name == table_name
        assert instance._joint_column == 'col1#col2#col3'
        assert instance._combinations is None

    def test___init___invalid_parameters(self):
        """Test the ``__init__`` method errors with invalid arguments."""
        # Setup
        bad_column_names_type = 'col1'
        bad_column_names = ['col1', 2]
        short_column_names = ['col1']
        bad_table_name = 1

        # Run and assert
        bad_column_names_msg = re.escape('`column_names` must be a list of strings.')
        with pytest.raises(ValueError, match=bad_column_names_msg):
            FixedCombinations(bad_column_names_type)

        with pytest.raises(ValueError, match=bad_column_names_msg):
            FixedCombinations(bad_column_names)

        short_column_names_msg = re.escape(
            'FixedCombinations pattern requires at least two columns.'
        )
        with pytest.raises(ValueError, match=short_column_names_msg):
            FixedCombinations(short_column_names)

        bad_table_name_msg = re.escape('`table_name` must be a string or None.')
        with pytest.raises(ValueError, match=bad_table_name_msg):
            FixedCombinations(column_names=['col1', 'col2'], table_name=bad_table_name)

    @patch('sdv.cag.fixed_combinations._validate_table_and_column_names')
    def test__validate_pattern_with_metadata(self, validate_table_and_col_names_mock):
        """Test validating the pattern with metadata."""
        # Setup
        instance = FixedCombinations(['col1', 'col2'])
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'col1': {'sdtype': 'boolean'},
                        'col2': {'sdtype': 'categorical'},
                        'col3': {'sdtype': 'id'},
                    },
                    'column_relationships': [
                        {'type': 'relationship', 'column_names': ['col1', 'col2']}
                    ],
                }
            }
        })

        # Run
        instance._validate_pattern_with_metadata(metadata)

    @patch('sdv.cag.fixed_combinations._validate_table_and_column_names')
    def test__validate_pattern_with_metadata_bad_col_sdtype(
        self, validate_table_and_col_names_mock
    ):
        """Test validating the pattern with metadata."""
        # Setup
        instance = FixedCombinations(['col1', 'col2'])
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'col1': {'sdtype': 'boolean'},
                        'col2': {'sdtype': 'id'},
                        'col3': {'sdtype': 'id'},
                    }
                }
            }
        })

        # Run and assert
        expected_msg = re.escape(
            "Column 'col2' has an incompatible sdtype ('id'). The column sdtype "
            "must be either 'boolean' or 'categorical'."
        )
        with pytest.raises(PatternNotMetError, match=expected_msg):
            instance._validate_pattern_with_metadata(metadata)

    @patch('sdv.cag.fixed_combinations._validate_table_and_column_names')
    def test__validate_pattern_with_metadata_col_relationship(
        self, validate_table_and_col_names_mock
    ):
        """Test validating the pattern with metadata."""
        # Setup
        instance = FixedCombinations(['col1', 'col2'])
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'col1': {'sdtype': 'boolean'},
                        'col2': {'sdtype': 'categorical'},
                        'col3': {'sdtype': 'id'},
                    },
                    'column_relationships': [
                        {'type': 'relationship', 'column_names': ['col2', 'col3']}
                    ],
                }
            }
        })

        # Run and assert
        expected_msg = re.escape(
            "Cannot apply pattern because columns ['col2'] are part of a column relationship."
        )
        with pytest.raises(PatternNotMetError, match=expected_msg):
            instance._validate_pattern_with_metadata(metadata)

    def test__get_updated_metadata(self):
        """Test the ``_get_updated_metadata`` method."""
        # Setup
        instance = FixedCombinations(['col1', 'col2'])
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'col1': {'sdtype': 'boolean'},
                        'col2': {'sdtype': 'categorical'},
                        'col3': {'sdtype': 'id'},
                        'col4': {'sdtype': 'id'},
                    },
                    'column_relationships': [
                        {'type': 'relationship', 'column_names': ['col2', 'col3']},
                        {'type': 'relationship', 'column_names': ['col3', 'col4']},
                    ],
                }
            }
        })

        # Run
        updated_metadata = instance._get_updated_metadata(metadata)

        # Assert
        expected_metadata_dict = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'col3': {'sdtype': 'id'},
                        'col4': {'sdtype': 'id'},
                        'col1#col2': {'sdtype': 'categorical'},
                    },
                    'column_relationships': [
                        {'type': 'relationship', 'column_names': ['col3', 'col4']}
                    ],
                }
            }
        }).to_dict()
        assert updated_metadata.to_dict() == expected_metadata_dict

    @patch('sdv.cag.fixed_combinations.get_mappable_combination')
    def test__fit(self, get_mappable_combination_mock):
        """Test the ``FixedCombinations._fit`` method."""
        # Setup
        columns = ['b', 'c']
        instance = FixedCombinations(column_names=columns)
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'a': {'sdtype': 'id'},
                        'b': {'sdtype': 'id'},
                        'c': {'sdtype': 'categorical'},
                        'b#c': {'sdtype': 'categorical'},
                    },
                }
            }
        })
        data = {
            'table': pd.DataFrame({
                'a': ['a', 'b', 'c'],
                'b': ['d', 'e', 'f'],
                'c': ['g', 'h', 'i'],
                'b#c': ['1', '2', '3'],
            })
        }

        # Run
        instance._fit(data, metadata)

        # Asserts
        expected_combinations = pd.DataFrame({'b': ['d', 'e', 'f'], 'c': ['g', 'h', 'i']})
        expected_calls = [
            call(combination)
            for combination in instance._combinations.itertuples(index=False, name=None)
        ]
        assert instance._joint_column == 'b#c_'
        pd.testing.assert_frame_equal(instance._combinations, expected_combinations)
        assert get_mappable_combination_mock.call_args_list == expected_calls

    @patch('sdv.cag.fixed_combinations.uuid')
    def test__transform(self, mock_uuid):
        """Test the ``FixedCombinations.transform`` method."""
        # Setup
        mock_uuid.uuid5.side_effect = ['combination1', 'combination2', 'combination3']
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'a': {'sdtype': 'categorical'},
                        'b': {'sdtype': 'categorical'},
                        'c': {'sdtype': 'categorical'},
                        'd': {'sdtype': 'categorical'},
                    }
                }
            }
        })
        data = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': [1, 2, 3],
            'c': ['g', 'h', 'i'],
            'd': [2.4, 1.23, 5.6],
        })
        columns = ['b', 'c', 'd']
        instance = FixedCombinations(column_names=columns)
        instance.fit(data, metadata)

        # Run
        out = instance.transform(data)

        # Assert
        assert instance._combinations_to_uuids is not None
        assert instance._uuids_to_combinations is not None
        expected_out = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b#c#d': ['combination1', 'combination2', 'combination3']
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test__reverse_transform(self):
        """Test the ``FixedCombinations.reverse_transform`` method."""
        # Setup
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'a': {'sdtype': 'categorical'},
                        'b': {'sdtype': 'categorical'},
                        'c': {'sdtype': 'categorical'},
                        'd': {'sdtype': 'categorical'},
                    }
                }
            }
        })
        data = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': [1, 2, 3],
            'c': ['g', 'h', 'i'],
            'd': [2.4, 1.23, 5.6],
        })
        columns = ['b', 'c', 'd']
        instance = FixedCombinations(column_names=columns)
        instance.fit(data, metadata)

        # Run
        transformed_data = instance.transform(data)
        out = instance.reverse_transform(transformed_data)

        # Assert
        assert instance._combinations_to_uuids is not None
        assert instance._uuids_to_combinations is not None
        pd.testing.assert_frame_equal(data, out)

    def test__is_valid(self):
        """Test the ``FixedCombinations.is_valid`` method."""
        # Setup
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'a': {'sdtype': 'categorical'},
                        'b': {'sdtype': 'categorical'},
                        'c': {'sdtype': 'categorical'},
                        'd': {'sdtype': 'categorical'},
                    }
                }
            }
        })
        data = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': ['d', 'e', 'f'],
            'c': ['g', 'h', 'i'],
        })
        invalid_data = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': ['D', 'E', 'F'],
            'c': ['g', 'h', 'i'],
        })

        columns = ['b', 'c']
        instance = FixedCombinations(column_names=columns)
        instance.fit(data, metadata)

        # Run
        valid_out = instance.is_valid(data)
        invalid_out = instance.is_valid(invalid_data)

        # Assert
        expected_valid_out = pd.Series([True, True, True], name='b#c')
        pd.testing.assert_series_equal(expected_valid_out, valid_out)
        pd.testing.assert_series_equal(~expected_valid_out, invalid_out)
