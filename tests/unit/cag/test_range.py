"""Unit tests for Range CAG pattern."""

import operator
import re
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sdv.cag import Range
from sdv.cag._errors import PatternNotMetError
from sdv.metadata import Metadata


class TestRange:
    def test___init___incorrect_column_name(self):
        """Test it raises an error if column_name is not a string."""
        # Run and Assert
        err_msg = '`low_column_name`, `middle_column_name` and `high_column_name` must be strings.'
        with pytest.raises(ValueError, match=err_msg):
            Range(low_column_name=1, middle_column_name='b', high_column_name='c')

        with pytest.raises(ValueError, match=err_msg):
            Range(low_column_name='a', middle_column_name=1, high_column_name='c')

        with pytest.raises(ValueError, match=err_msg):
            Range(low_column_name='a', middle_column_name='b', high_column_name=1)

    def test___init___incorrect_strict_boundaries(self):
        """Test it raises an error if strict_boundaries is not a boolean."""
        # Run and Assert
        err_msg = '`strict_boundaries` must be a boolean.'
        with pytest.raises(ValueError, match=err_msg):
            Range(
                low_column_name='a',
                middle_column_name='b',
                high_column_name='c',
                strict_boundaries=1,
            )

    def test___init___incorrect_table_name(self):
        """Test it raises an error if table_name is not a string."""
        # Run and Assert
        err_msg = '`table_name` must be a string or None.'
        with pytest.raises(ValueError, match=err_msg):
            Range(
                low_column_name='a',
                middle_column_name='b',
                high_column_name='c',
                table_name=1,
            )

    def test___init___(self):
        """Test it initializes correctly."""
        # Run
        instance = Range(low_column_name='a', middle_column_name='b', high_column_name='c')

        # Asserts
        assert instance._low_column_name == 'a'
        assert instance._middle_column_name == 'b'
        assert instance._high_column_name == 'c'
        assert instance._low_diff_column_name == 'a#b'
        assert instance._high_diff_column_name == 'b#c'
        assert instance._nan_column_name == 'a#b#c.nan_component'
        assert instance._operator == operator.lt
        assert instance._dtype is None
        assert instance._is_datetime is None
        assert instance.table_name is None
        assert instance._low_datetime_format is None
        assert instance._middle_datetime_format is None
        assert instance._high_datetime_format is None

    def test___init___strict_boundaries_false(self):
        """Test it initializes correctly when strict_boundaries is False."""
        # Run
        instance = Range(
            low_column_name='a',
            middle_column_name='b',
            high_column_name='c',
            strict_boundaries=False,
        )

        # Assert
        assert instance._operator == operator.le

    @patch('sdv.cag.range._validate_table_and_column_names')
    def test__validate_pattern_with_metadata(self, validate_table_and_col_names_mock):
        """Test validating the pattern with metadata."""
        # Setup
        instance = Range(
            low_column_name='low',
            middle_column_name='middle',
            high_column_name='high',
        )
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'low': {'sdtype': 'numerical'},
                        'middle': {'sdtype': 'numerical'},
                        'high': {'sdtype': 'numerical'},
                        'col': {'sdtype': 'id'},
                    },
                    'column_relationships': [
                        {'type': 'relationship', 'column_names': ['low', 'high']}
                    ],
                }
            }
        })

        # Run
        instance._validate_pattern_with_metadata(metadata)

        # Assert
        validate_table_and_col_names_mock.assert_called_once()

    def test__validate_pattern_with_metadata_incorrect_sdtype(self):
        """Test it when the sdtype is not numerical or datetime."""
        # Setup
        instance = Range(
            low_column_name='low',
            middle_column_name='middle',
            high_column_name='high',
        )
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'low': {'sdtype': 'numerical'},
                        'middle': {'sdtype': 'numerical'},
                        'high': {'sdtype': 'boolean'},
                        'col': {'sdtype': 'id'},
                    },
                    'column_relationships': [
                        {'type': 'relationship', 'column_names': ['low', 'high']}
                    ],
                }
            }
        })

        # Run and Assert
        err_msg = re.escape(
            "Column 'high' has an incompatible sdtype 'boolean'. The column "
            "sdtype must be either 'numerical' or 'datetime'."
        )
        with pytest.raises(PatternNotMetError, match=err_msg):
            instance._validate_pattern_with_metadata(metadata)

    def test__validate_pattern_with_metadata_non_matching_sdtype(self):
        """Test it when the sdtypes are not the same."""
        # Setup
        instance = Range(
            low_column_name='low',
            middle_column_name='middle',
            high_column_name='high',
        )
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'low': {'sdtype': 'numerical'},
                        'middle': {'sdtype': 'datetime'},
                        'high': {'sdtype': 'datetime'},
                        'col': {'sdtype': 'id'},
                    },
                    'column_relationships': [
                        {'type': 'relationship', 'column_names': ['low', 'high']}
                    ],
                }
            }
        })

        # Run and Assert
        err_msg = re.escape(
            "Columns 'low', 'middle' and 'high' must have the same sdtype. "
            "Found 'numerical', 'datetime' and 'datetime'."
        )
        with pytest.raises(PatternNotMetError, match=err_msg):
            instance._validate_pattern_with_metadata(metadata)

    def test__validate_pattern_with_data(self):
        """Test it when the data is not valid."""
        # Setup
        instance = Range(
            low_column_name='low',
            middle_column_name='middle',
            high_column_name='high',
            table_name='table',
        )
        data = {'table': pd.DataFrame({'low': [1, 2, 3], 'middle': [4, -1, 6], 'high': [7, 8, 9]})}
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'low': {'sdtype': 'numerical'},
                        'middle': {'sdtype': 'numerical'},
                        'high': {'sdtype': 'numerical'},
                        'col': {'sdtype': 'id'},
                    },
                    'column_relationships': [
                        {'type': 'relationship', 'column_names': ['low', 'high']}
                    ],
                }
            }
        })

        # Run and Assert
        err_msg = re.escape('The range requirement is not met for row indices: [1]')
        with pytest.raises(PatternNotMetError, match=err_msg):
            instance._validate_pattern_with_data(data, metadata)

    def test__validate_pattern_with_data_multiple_rows(self):
        """Test it when the data is not valid for over 5 rows."""
        # Setup
        instance = Range(
            low_column_name='low',
            middle_column_name='middle',
            high_column_name='high',
            table_name='table',
        )
        data = {
            'table': pd.DataFrame({
                'low': [1, 2, 3, 4, 5, 6, 10, 12, 9, 10],
                'middle': [4, 0, 6, 7, 8, 9, 10, 11, 9, -13],
                'high': [7, 0, 7, 10, -1, 12, 13, 14, 9, 16],
            })
        }
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'low': {'sdtype': 'numerical'},
                        'middle': {'sdtype': 'numerical'},
                        'high': {'sdtype': 'numerical'},
                        'col': {'sdtype': 'id'},
                    },
                    'column_relationships': [
                        {'type': 'relationship', 'column_names': ['low', 'high']}
                    ],
                }
            }
        })

        # Run and Assert
        err_msg = re.escape(
            'The range requirement is not met for row indices: [1, 4, 6, 7, 8, +1 more]'
        )
        with pytest.raises(PatternNotMetError, match=err_msg):
            instance._validate_pattern_with_data(data, metadata)

    def test__validate_pattern_with_data_nans(self):
        """Test it when the data is not valid and contains nans."""
        # Setup
        instance = Range(
            low_column_name='low',
            middle_column_name='middle',
            high_column_name='high',
        )
        data = {
            'table': pd.DataFrame({
                'low': [np.nan, np.nan, 3, 4, None, 6, 8, 0],
                'middle': [np.nan, 2, 2, 4, np.nan, -6, 10, float('nan')],
                'high': [np.nan, 8, 9, 10, 11, 12, 13, 14],
                'col': [7, 8, 9, 10, 11, 12, 13, 14],
            })
        }
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'low': {'sdtype': 'numerical'},
                        'middle': {'sdtype': 'numerical'},
                        'high': {'sdtype': 'numerical'},
                        'col': {'sdtype': 'id'},
                    },
                    'column_relationships': [
                        {'type': 'relationship', 'column_names': ['low', 'high']}
                    ],
                }
            }
        })

        # Run and Assert
        err_msg = re.escape('The range requirement is not met for row indices: [2, 3, 5]')
        with pytest.raises(PatternNotMetError, match=err_msg):
            instance._validate_pattern_with_data(data, metadata)

    def test__validate_pattern_with_data_strict_boundaries_false(self):
        """Test it when the data is not valid and contains nans."""
        # Setup
        instance = Range(
            low_column_name='low',
            middle_column_name='middle',
            high_column_name='high',
            strict_boundaries=False,
        )
        data = {
            'table': pd.DataFrame({
                'low': [1, np.nan, 3, 4, None, 6, 8, 0],
                'middle': [4, 2, 2, 4, np.nan, -6, 10, float('nan')],
                'high': [4, 2, 2, 4, np.nan, -6, 10, float('nan')],
                'col': [7, 8, 9, 10, 11, 12, 13, 14],
            })
        }
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'low': {'sdtype': 'numerical'},
                        'middle': {'sdtype': 'numerical'},
                        'high': {'sdtype': 'numerical'},
                        'col': {'sdtype': 'id'},
                    },
                    'column_relationships': [
                        {'type': 'relationship', 'column_names': ['low', 'high']}
                    ],
                }
            }
        })

        # Run and Assert
        err_msg = re.escape('The range requirement is not met for row indices: [2, 5]')
        with pytest.raises(PatternNotMetError, match=err_msg):
            instance._validate_pattern_with_data(data, metadata)

    def test__validate_pattern_with_data_datetime(self):
        """Test it when the data is not valid and contains datetimes."""
        # Setup
        instance = Range(
            low_column_name='low',
            middle_column_name='middle',
            high_column_name='high',
            strict_boundaries=True,
        )
        data = {
            'table': pd.DataFrame({
                'low': [datetime(2020, 5, 17), datetime(2021, 9, 1), np.nan],
                'middle': [datetime(2020, 5, 17, 12, 0, 0), datetime(2021, 9, 1), np.nan],
                'high': [datetime(2020, 5, 18), datetime(2020, 9, 2), datetime(2020, 9, 2)],
                'col': [7, 8, 9],
            })
        }
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'low': {'sdtype': 'datetime'},
                        'middle': {'sdtype': 'datetime'},
                        'high': {'sdtype': 'datetime'},
                        'col': {'sdtype': 'id'},
                    },
                    'column_relationships': [
                        {'type': 'relationship', 'column_names': ['low', 'high']}
                    ],
                }
            }
        })

        # Run and Assert
        err_msg = re.escape('The range requirement is not met for row indices: [1]')
        with pytest.raises(PatternNotMetError, match=err_msg):
            instance._validate_pattern_with_data(data, metadata)

    def test__get_updated_metadata(self):
        """Test the ``_get_updated_metadata`` method."""
        # Setup
        instance = Range(
            low_column_name='low',
            middle_column_name='middle',
            high_column_name='high',
        )
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'low': {'sdtype': 'numerical'},
                        'middle': {'sdtype': 'numerical'},
                        'high': {'sdtype': 'numerical'},
                        'col': {'sdtype': 'id'},
                    },
                    'column_relationships': [
                        {'type': 'relationship', 'column_names': ['low', 'high']},
                        {'type': 'relationship', 'column_names': ['low', 'col']},
                        {'type': 'relationship', 'column_names': ['high', 'col']},
                        {'type': 'relationship', 'column_names': ['middle', 'col']},
                        {'type': 'relationship', 'column_names': ['middle', 'high']},
                        {'type': 'relationship', 'column_names': ['low', 'middle']},
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
                        'low': {'sdtype': 'numerical'},
                        'col': {'sdtype': 'id'},
                        'low#middle': {'sdtype': 'numerical'},
                        'middle#high': {'sdtype': 'numerical'},
                        'low#middle#high.nan_component': {'sdtype': 'categorical'},
                    },
                    'column_relationships': [
                        {'type': 'relationship', 'column_names': ['low', 'col']},
                    ],
                }
            }
        }).to_dict()
        assert updated_metadata.to_dict() == expected_metadata_dict

    def test__fit(self):
        """Test it learns the correct attributes."""
        # Setup
        table_data = {'table': pd.DataFrame({'a': [1, 2, 4], 'b': [4, 5, 6], 'c': [7, 8, 9]})}
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'a': {'sdtype': 'datetime', 'datetime_format': '%y %m, %d'},
                        'b': {'sdtype': 'datetime', 'datetime_format': '%y %m %d'},
                        'c': {'sdtype': 'datetime', 'datetime_format': '%y %m %d %H:%M:%S'},
                    },
                }
            }
        })
        instance = Range(
            low_column_name='a',
            middle_column_name='b',
            high_column_name='c',
            table_name='table',
        )

        # Run
        instance._fit(table_data, metadata)

        # Assert
        assert instance._is_datetime is True
        assert instance._dtype == pd.Series([1]).dtype  # exact dtype (32 or 64) depends on OS
        assert instance._low_datetime_format == '%y %m, %d'
        assert instance._middle_datetime_format == '%y %m %d'
        assert instance._high_datetime_format == '%y %m %d %H:%M:%S'
        assert instance._low_diff_column_name == 'a#b'
        assert instance._high_diff_column_name == 'b#c'

    @pytest.mark.parametrize('dtype', ['Float64', 'Float32', 'Int64', 'Int32', 'Int16', 'Int8'])
    def test__fit_numerical(self, dtype):
        """Test it for numerical columns."""
        # Setup
        table_data = {
            'table': pd.DataFrame(
                {'a': [1, 2, 4], 'b': [4, 5, 6], 'c': [7, 8, 9]},
                dtype=dtype,
            )
        }
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'a': {'sdtype': 'numerical'},
                        'b': {'sdtype': 'numerical'},
                        'c': {'sdtype': 'numerical'},
                    },
                }
            }
        })
        instance = Range(
            low_column_name='a',
            middle_column_name='b',
            high_column_name='c',
            table_name='table',
        )

        # Run
        instance._fit(table_data, metadata)

        # Assert
        assert instance._dtype == dtype
        assert instance._is_datetime is False
        assert instance._low_datetime_format is None
        assert instance._high_datetime_format is None
        assert instance._low_diff_column_name == 'a#b'
        assert instance._high_diff_column_name == 'b#c'

    def test__fit_datetime(self):
        """Test it for datetime strings."""
        # Setup
        table_data = {
            'table': pd.DataFrame({
                'a': pd.to_datetime(['2020-01-01']),
                'b': pd.to_datetime(['2020-01-02']),
                'c': pd.to_datetime(['2020-01-03']),
            })
        }
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'a': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
                        'b': {'sdtype': 'datetime'},
                        'c': {'sdtype': 'datetime'},
                    },
                }
            }
        })
        instance = Range(
            low_column_name='a',
            middle_column_name='b',
            high_column_name='c',
            table_name='table',
        )

        # Run
        instance._fit(table_data, metadata)

        # Assert
        assert instance._dtype == np.dtype('<M8[ns]')
        assert instance._is_datetime is True
        assert instance._low_datetime_format == '%Y-%m-%d'
        assert instance._middle_datetime_format is None
        assert instance._high_datetime_format is None
        assert instance._low_diff_column_name == 'a#b'
        assert instance._high_diff_column_name == 'b#c'

    def test__transform(self):
        """Test it transforms the data correctly."""
        # Setup
        table_data = {
            'table': pd.DataFrame({
                'a': [1, 2, 3],
                'b': [4, 5, 6],
                'c': [7, 8, 9],
                'col': [7, 8, 9],
            })
        }
        instance = Range(
            low_column_name='a',
            middle_column_name='b',
            high_column_name='c',
            table_name='table',
        )

        # Run
        out = instance._transform(table_data)

        # Assert
        out = out['table']
        expected_out = pd.DataFrame({
            'a': [1, 2, 3],
            'col': [7, 8, 9],
            'a#b': [np.log(4)] * 3,
            'b#c': [np.log(4)] * 3,
            'a#b#c.nan_component': [None] * 3,
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test__transform_with_nans(self):
        """Test it transforms the data correctly when it contains nans."""
        # Setup
        instance = Range(
            low_column_name='a',
            middle_column_name='b',
            high_column_name='c',
            table_name='table',
        )

        table_data_with_nans = {
            'table': pd.DataFrame({
                'a': [1, np.nan, 3, np.nan],
                'b': [np.nan, 2, 4, np.nan],
                'c': [np.nan, 3, 5, np.nan],
            })
        }

        table_data_without_nans = {
            'table': pd.DataFrame({
                'a': [1, 2, 3],
                'b': [2, 3, 4],
                'c': [3, 4, 5],
            })
        }

        # Run
        output_with_nans = instance._transform(table_data_with_nans)
        output_without_nans = instance._transform(table_data_without_nans)

        # Assert
        output_with_nans = output_with_nans['table']
        output_without_nans = output_without_nans['table']
        expected_output_with_nans = pd.DataFrame({
            'a': [1.0, 2.0, 3.0, 2.0],
            'a#b': [np.log(2)] * 4,
            'b#c': [np.log(2)] * 4,
            'a#b#c.nan_component': ['b, c', 'a', 'None', 'a, b, c'],
        })

        expected_output_without_nans = pd.DataFrame({
            'a': [1, 2, 3],
            'a#b': [np.log(2)] * 3,
            'b#c': [np.log(2)] * 3,
            'a#b#c.nan_component': [None] * 3,
        })
        pd.testing.assert_frame_equal(output_with_nans, expected_output_with_nans)
        pd.testing.assert_frame_equal(output_without_nans, expected_output_without_nans)

    def test_transform_existing_column_name(self):
        """Test ``_transform`` method when the ``diff_column_name`` already exists in the table."""
        # Setup
        instance = Range(
            low_column_name='a',
            middle_column_name='b',
            high_column_name='c',
            table_name='table',
        )
        instance._low_diff_column_name = 'a#b_'
        table_data = {
            'table': pd.DataFrame({
                'a': [1, 2, 3],
                'b': [4, 5, 6],
                'c': [7, 8, 9],
                'col': [7, 8, 9],
                'a#b': ['c', 'd', 'e'],
            })
        }

        # Run
        output = instance._transform(table_data)

        # Assert
        output = output['table']
        expected_column_name = ['a', 'col', 'a#b', 'a#b_', 'b#c', 'a#b#c.nan_component']
        assert list(output.columns) == expected_column_name

    def test__transform_datetime(self):
        """Test it transforms the data correctly when it contains datetimes."""
        # Setup
        table_data = {
            'table': pd.DataFrame({
                'a': pd.to_datetime(['2020-01-01T00:00:00', '2020-01-02T00:00:00']),
                'b': pd.to_datetime(['2020-01-01T00:00:01', '2020-01-02T00:00:01']),
                'c': pd.to_datetime(['2020-01-01T00:00:02', '2020-01-02T00:00:02']),
                'col': [7, 8],
            })
        }
        instance = Range(
            low_column_name='a',
            middle_column_name='b',
            high_column_name='c',
            table_name='table',
        )
        instance._is_datetime = True

        # Run
        out = instance._transform(table_data)

        # Assert
        out = out['table']
        expected_out = pd.DataFrame({
            'a': pd.to_datetime(['2020-01-01T00:00:00', '2020-01-02T00:00:00']),
            'col': [7, 8],
            'a#b': [np.log(1_000_000_001), np.log(1_000_000_001)],
            'b#c': [np.log(1_000_000_001), np.log(1_000_000_001)],
            'a#b#c.nan_component': [None, None],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test__transform_datetime_dtype_object(self):
        """Test it transforms the data correctly when the dtype is object."""
        # Setup
        table_data = {
            'table': pd.DataFrame({
                'a': ['2020-01-01T00:00:00', '2020-01-02T00:00:00'],
                'b': ['2020-01-01T00:00:01', '2020-01-02T00:00:01'],
                'c': ['2020-01-01T00:00:02', '2020-01-02T00:00:02'],
                'col': [1, 2],
            })
        }
        instance = Range(
            low_column_name='a',
            middle_column_name='b',
            high_column_name='c',
            table_name='table',
        )
        instance._is_datetime = True
        instance._dtype = 'O'

        # Run
        out = instance._transform(table_data)

        # Assert
        out = out['table']
        expected_out = pd.DataFrame({
            'a': ['2020-01-01T00:00:00', '2020-01-02T00:00:00'],
            'col': [1, 2],
            'a#b': [np.log(1_000_000_001), np.log(1_000_000_001)],
            'b#c': [np.log(1_000_000_001), np.log(1_000_000_001)],
            'a#b#c.nan_component': [None, None],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform(self):
        """Test it reverses the transformation correctly."""
        # Setup
        transformed = {
            'table': pd.DataFrame({
                'a': [1, 2, 3],
                'col': [7, 8, 9],
                'a#b': [np.log(4)] * 3,
                'b#c': [np.log(4)] * 3,
                'a#b#c.nan_component': [None] * 3,
            })
        }
        instance = Range(
            low_column_name='a',
            middle_column_name='b',
            high_column_name='c',
            table_name='table',
        )
        instance._dtype = pd.Series([1]).dtype  # exact dtype (32 or 64) depends on OS
        instance._original_data_columns = {'table': ['a', 'b', 'c', 'col']}
        instance._dtypes = {
            'table': {
                'a': pd.Series([1]).dtype,
                'b': pd.Series([1]).dtype,
                'c': pd.Series([1]).dtype,
                'col': pd.Series([1]).dtype,
            }
        }

        # Run
        out = instance.reverse_transform(transformed)

        # Assert
        out = out['table']
        expected_out = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
            'col': [7, 8, 9],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform_floats(self):
        """Test it reverses the transformation correctly when the dtype is float."""
        # Setup
        transformed = {
            'table': pd.DataFrame({
                'a': [1.1, 2.2, 3.3],
                'col': [7, 8, 9],
                'a#b': [np.log(4)] * 3,
                'b#c': [np.log(4)] * 3,
                'a#b#c.nan_component': [None] * 3,
            })
        }
        instance = Range(
            low_column_name='a',
            middle_column_name='b',
            high_column_name='c',
            table_name='table',
        )
        instance._dtype = np.dtype('float')
        instance._original_data_columns = {'table': ['a', 'b', 'c', 'col']}
        instance._dtypes = {
            'table': {
                'a': np.dtype('float'),
                'b': np.dtype('float'),
                'c': np.dtype('float'),
                'col': pd.Series([1]).dtype,
            }
        }

        # Run
        out = instance.reverse_transform(transformed)

        # Assert
        out = out['table']
        expected_out = pd.DataFrame({
            'a': [1.1, 2.2, 3.3],
            'b': [4.1, 5.2, 6.3],
            'c': [7.1, 8.2, 9.3],
            'col': [7, 8, 9],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform_datetime(self):
        """Test it reverses the transformation correctly when the dtype is datetime."""
        # Setup
        transformed = {
            'table': pd.DataFrame({
                'a': pd.to_datetime(['2020-01-01T00:00:00', '2020-01-02T00:00:00']),
                'col': [1, 2],
                'a#b': [np.log(1_000_000_001), np.log(1_000_000_001)],
                'b#c': [np.log(1_000_000_001), np.log(1_000_000_001)],
                'a#b#c.nan_component': [None] * 2,
            })
        }
        instance = Range(
            low_column_name='a',
            middle_column_name='b',
            high_column_name='c',
            table_name='table',
        )
        instance._dtype = np.dtype('<M8[ns]')
        instance._is_datetime = True
        instance._original_data_columns = {'table': ['a', 'b', 'c', 'col']}
        instance._dtypes = {
            'table': {
                'a': np.dtype('<M8[ns]'),
                'b': np.dtype('<M8[ns]'),
                'c': np.dtype('<M8[ns]'),
                'col': pd.Series([1]).dtype,
            }
        }

        # Run
        out = instance.reverse_transform(transformed)

        # Assert
        out = out['table']
        expected_out = pd.DataFrame({
            'a': pd.to_datetime(['2020-01-01T00:00:00', '2020-01-02T00:00:00']),
            'b': pd.to_datetime(['2020-01-01T00:00:01', '2020-01-02T00:00:01']),
            'c': pd.to_datetime(['2020-01-01T00:00:02', '2020-01-02T00:00:02']),
            'col': [1, 2],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform_datetime_dtype_is_object(self):
        """Test it reverses the transformation correctly when the dtype is object."""
        # Setup
        transformed = {
            'table': pd.DataFrame({
                'a': ['2020-01-01T00:00:00', '2020-01-02T00:00:00'],
                'col': [1, 2],
                'a#b': [np.log(1_000_000_001), np.log(1_000_000_001)],
                'b#c': [np.log(1_000_000_001), np.log(1_000_000_001)],
                'a#b#c.nan_component': [None, None],
            })
        }
        instance = Range(
            low_column_name='a',
            middle_column_name='b',
            high_column_name='c',
            table_name='table',
        )
        instance._dtype = np.dtype('O')
        instance._is_datetime = True
        instance._original_data_columns = {'table': ['a', 'b', 'c', 'col']}
        instance._dtypes = {
            'table': {
                'a': np.dtype('O'),
                'b': np.dtype('O'),
                'c': np.dtype('O'),
                'col': pd.Series([1]).dtype,
            }
        }

        # Run
        out = instance.reverse_transform(transformed)

        # Assert
        out = out['table']
        expected_out = pd.DataFrame({
            'a': ['2020-01-01T00:00:00', '2020-01-02T00:00:00'],
            'b': [pd.Timestamp('2020-01-01 00:00:01'), pd.Timestamp('2020-01-02 00:00:01')],
            'c': [pd.Timestamp('2020-01-01 00:00:02'), pd.Timestamp('2020-01-02 00:00:02')],
            'col': [1, 2],
        })
        expected_out['b'] = expected_out['b'].astype(np.dtype('O'))
        expected_out['c'] = expected_out['c'].astype(np.dtype('O'))
        pd.testing.assert_frame_equal(out, expected_out)

    def test_is_valid(self):
        """Test it checks if the data is valid."""
        # Setup
        table_data = {
            'table': pd.DataFrame({
                'a': [1, np.nan, 3, 4, None, 6, 8, 0],
                'b': [4, 2, 2, 4, np.nan, -6, 10, float('nan')],
                'c': [7, 8, 9, 10, 11, 12, 13, 14],
                'col': [7, 8, 9, 10, 11, 12, 13, 14],
            })
        }
        instance = Range(
            low_column_name='a',
            middle_column_name='b',
            high_column_name='c',
            table_name='table',
        )
        instance._fitted = True

        # Run
        out = instance.is_valid(table_data)

        # Assert
        out = out['table']
        expected_out = [True, True, False, False, True, False, True, True]
        np.testing.assert_array_equal(expected_out, out)

    def test_is_valid_strict_boundaries_true(self):
        """Test it checks if the data is valid when strict boundaries are False."""
        # Setup
        table_data = {
            'table': pd.DataFrame({
                'a': [1, np.nan, 3, 3, None, 6, 8, 0],
                'b': [4, 2, 2, 4, np.nan, -6, 10, float('nan')],
                'c': [7, 8, 9, 10, 11, 12, 13, 14],
                'col': [7, 8, 9, 10, 11, 12, 13, 14],
            })
        }
        instance = Range(
            low_column_name='a',
            middle_column_name='b',
            high_column_name='c',
            strict_boundaries=False,
            table_name='table',
        )
        instance._fitted = True

        # Run
        out = instance.is_valid(table_data)

        # Assert
        out = out['table']
        expected_out = [True, True, False, True, True, False, True, True]
        np.testing.assert_array_equal(expected_out, out)

    def test_is_valid_datetimes(self):
        """Test it checks if the data is valid when it contains datetimes."""
        # Setup
        table_data = {
            'table': pd.DataFrame({
                'a': ['2020-5-17', '2021-9-1', None],
                'b': [datetime(2020, 5, 18), datetime(2020, 9, 2), datetime(2020, 9, 2)],
                'c': [datetime(2020, 5, 29), datetime(2021, 9, 3), np.nan],
                'col': [7, 8, 9],
            })
        }

        instance = Range(
            low_column_name='a',
            middle_column_name='b',
            high_column_name='c',
            table_name='table',
        )
        instance._fitted = True
        instance._low_datetime_format = '%Y-%m-%d'
        instance._is_datetime = True
        instance._dtype = 'O'

        # Run
        out = instance.is_valid(table_data)

        # Assert
        out = out['table']
        expected_out = [True, False, True]
        np.testing.assert_array_equal(expected_out, out)
