"""Unit tests for Inequality CAG constraint."""

import re
from datetime import datetime
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from sdv.cag import Inequality
from sdv.cag._errors import PatternNotMetError
from sdv.metadata import Metadata


class TestInequality:
    def test___init___incorrect_column_name(self):
        """Test it raises an error if column_name is not a string."""
        # Run and Assert
        err_msg = '`low_column_name` and `high_column_name` must be strings.'
        with pytest.raises(ValueError, match=err_msg):
            Inequality(low_column_name=1, high_column_name='b')

        with pytest.raises(ValueError, match=err_msg):
            Inequality(low_column_name='a', high_column_name=1)

    def test___init___incorrect_strict_boundaries(self):
        """Test it raises an error if strict_boundaries is not a boolean."""
        # Run and Assert
        err_msg = '`strict_boundaries` must be a boolean.'
        with pytest.raises(ValueError, match=err_msg):
            Inequality(low_column_name='a', high_column_name='b', strict_boundaries=1)

    def test___init___incorrect_table_name(self):
        """Test it raises an error if table_name is not a string."""
        # Run and Assert
        err_msg = '`table_name` must be a string or None.'
        with pytest.raises(ValueError, match=err_msg):
            Inequality(low_column_name='a', high_column_name='b', table_name=1)

    def test___init___(self):
        """Test it initializes correctly."""
        # Run
        instance = Inequality(low_column_name='a', high_column_name='b')

        # Asserts
        assert instance._low_column_name == 'a'
        assert instance._high_column_name == 'b'
        assert instance._diff_column_name == 'a#b'
        assert instance._operator == np.greater_equal
        assert instance._dtype is None
        assert instance._is_datetime is None
        assert instance._nan_column_name is None
        assert instance.table_name is None
        assert instance._low_datetime_format is None
        assert instance._high_datetime_format is None

    def test___init___strict_boundaries_true(self):
        """Test it initializes correctly when strict_boundaries is True."""
        # Run
        instance = Inequality(low_column_name='a', high_column_name='b', strict_boundaries=True)

        # Assert
        assert instance._operator == np.greater

    @patch('sdv.cag.inequality._validate_table_and_column_names')
    def test__validate_pattern_with_metadata(self, validate_table_and_col_names_mock):
        """Test validating the constraint with metadata."""
        # Setup
        instance = Inequality(low_column_name='low', high_column_name='high')
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'low': {'sdtype': 'numerical'},
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
        instance = Inequality(low_column_name='low', high_column_name='high')
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'low': {'sdtype': 'numerical'},
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
        instance = Inequality(low_column_name='low', high_column_name='high')
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'low': {'sdtype': 'numerical'},
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
            "Columns 'low' and 'high' must have the same sdtype. Found 'numerical' and 'datetime'."
        )
        with pytest.raises(PatternNotMetError, match=err_msg):
            instance._validate_pattern_with_metadata(metadata)

    def test__validate_pattern_with_data(self):
        """Test it when the data is not valid."""
        # Setup
        instance = Inequality(low_column_name='low', high_column_name='high', table_name='table')
        data = {'table': pd.DataFrame({'low': [1, 2, 3], 'high': [4, 0, 6]})}
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'low': {'sdtype': 'numerical'},
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
        err_msg = re.escape('The inequality requirement is not met for row indices: [1]')
        with pytest.raises(PatternNotMetError, match=err_msg):
            instance._validate_pattern_with_data(data, metadata)

    def test__validate_pattern_with_data_multiple_rows(self):
        """Test it when the data is not valid for over 5 rows."""
        # Setup
        instance = Inequality(low_column_name='low', high_column_name='high', table_name='table')
        data = {
            'table': pd.DataFrame({
                'low': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                'high': [4, 0, 6, 4, 0, 6, 4, 0, 6, 4],
            })
        }
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'low': {'sdtype': 'numerical'},
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
            'The inequality requirement is not met for row indices: [1, 4, 6, 7, 8, +1 more]'
        )
        with pytest.raises(PatternNotMetError, match=err_msg):
            instance._validate_pattern_with_data(data, metadata)

    def test__validate_pattern_with_data_nans(self):
        """Test it when the data is not valid and contains nans."""
        # Setup
        instance = Inequality(low_column_name='low', high_column_name='high')
        data = {
            'table': pd.DataFrame({
                'low': [1, np.nan, 3, 4, None, 6, 8, 0],
                'high': [4, 2, 2, 4, np.nan, -6, 10, float('nan')],
                'col': [7, 8, 9, 10, 11, 12, 13, 14],
            })
        }
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'low': {'sdtype': 'numerical'},
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
        err_msg = re.escape('The inequality requirement is not met for row indices: [2, 5]')
        with pytest.raises(PatternNotMetError, match=err_msg):
            instance._validate_pattern_with_data(data, metadata)

    def test__validate_pattern_with_data_strict_boundaries_true(self):
        """Test it when the data is not valid and contains nans."""
        # Setup
        instance = Inequality(
            low_column_name='low',
            high_column_name='high',
            strict_boundaries=True,
        )
        data = {
            'table': pd.DataFrame({
                'low': [1, np.nan, 3, 4, None, 6, 8, 0],
                'high': [4, 2, 2, 4, np.nan, -6, 10, float('nan')],
                'col': [7, 8, 9, 10, 11, 12, 13, 14],
            })
        }
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'low': {'sdtype': 'numerical'},
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
        err_msg = re.escape('The inequality requirement is not met for row indices: [2, 3, 5]')
        with pytest.raises(PatternNotMetError, match=err_msg):
            instance._validate_pattern_with_data(data, metadata)

    def test__validate_pattern_with_data_datetime(self):
        """Test it when the data is not valid and contains datetimes."""
        # Setup
        instance = Inequality(
            low_column_name='low',
            high_column_name='high',
            strict_boundaries=True,
        )
        data = {
            'table': pd.DataFrame({
                'low': [datetime(2020, 5, 17), datetime(2021, 9, 1), np.nan],
                'high': [datetime(2020, 5, 18), datetime(2020, 9, 2), datetime(2020, 9, 2)],
                'col': [7, 8, 9],
            })
        }
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'low': {'sdtype': 'datetime'},
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
        err_msg = re.escape('The inequality requirement is not met for row indices: [1]')
        with pytest.raises(PatternNotMetError, match=err_msg):
            instance._validate_pattern_with_data(data, metadata)

    def test__validate_pattern_with_data_datetime_objects(self):
        """Test it when the data is not valid and contains datetimes as objects."""
        # Setup
        instance = Inequality(
            low_column_name='low',
            high_column_name='high',
            strict_boundaries=True,
        )
        data = {
            'table': pd.DataFrame({
                'low': ['2020-5-17', '2021-9-1', np.nan],
                'high': ['2020-5-18', '2020-9-2', '2020-9-2'],
                'col': [7, 8, 9],
            })
        }
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'low': {'sdtype': 'datetime'},
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
        err_msg = re.escape('The inequality requirement is not met for row indices: [1]')
        with pytest.raises(PatternNotMetError, match=err_msg):
            instance._validate_pattern_with_data(data, metadata)

    @patch('sdv.cag.inequality.match_datetime_precision')
    def test__validate_pattern_with_data_datetime_objects_missmatching_formats(
        self, mock_match_datetime_precision
    ):
        """Test it when the data is not valid with datetimes with missmatching formats."""
        # Setup
        instance = Inequality(
            low_column_name='low',
            high_column_name='high',
            strict_boundaries=True,
        )
        data = {
            'table': pd.DataFrame({
                'low': [
                    '2016-07-10 17:04:00',
                    '2016-07-11 13:23:00',
                    '2016-07-12 08:45:30',
                    '2016-07-11 12:00:00',
                    '2016-07-12 10:30:00',
                ],
                'high': [
                    '2016-07-10',
                    '2016-07-11',
                    '2016-07-12',
                    '2016-07-13',
                    '2016-07-14',
                ],
                'col': [7, 8, 9, 10, 11],
            })
        }
        low_return = np.array([
            datetime(2020, 5, 18),
            datetime(2020, 9, 2),
            datetime(2020, 9, 2),
            datetime(2020, 5, 18),
            datetime(2020, 9, 2),
        ])
        high_return = np.array([
            datetime(2020, 5, 17),
            datetime(2021, 9, 1),
            datetime(2020, 5, 17),
            datetime(2021, 9, 1),
            datetime(2021, 9, 1),
        ])
        mock_match_datetime_precision.return_value = (low_return, high_return)
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'low': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d %H:%M:%S'},
                        'high': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
                        'col': {'sdtype': 'id'},
                    },
                    'column_relationships': [
                        {'type': 'relationship', 'column_names': ['low', 'high']}
                    ],
                }
            }
        })

        # Run and Assert
        err_msg = re.escape('The inequality requirement is not met for row indices: [0, 2]')
        with pytest.raises(PatternNotMetError, match=err_msg):
            instance._validate_pattern_with_data(data, metadata)

    def test__get_updated_metadata(self):
        """Test the ``_get_updated_metadata`` method."""
        # Setup
        instance = Inequality(low_column_name='low', high_column_name='high')
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'low': {'sdtype': 'numerical'},
                        'high': {'sdtype': 'numerical'},
                        'col': {'sdtype': 'id'},
                    },
                    'column_relationships': [
                        {'type': 'relationship', 'column_names': ['low', 'high']},
                        {'type': 'relationship', 'column_names': ['low', 'col']},
                        {'type': 'relationship', 'column_names': ['high', 'col']},
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
                        'low#high': {'sdtype': 'numerical'},
                        'low#high.nan_component': {'sdtype': 'categorical'},
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
        data = {'table': pd.DataFrame({'a': [1, 2, 4], 'b': [4, 5, 6]})}
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'a': {'sdtype': 'datetime', 'datetime_format': '%y %m, %d'},
                        'b': {'sdtype': 'datetime', 'datetime_format': '%y %m %d'},
                    },
                }
            }
        })
        instance = Inequality(low_column_name='a', high_column_name='b', table_name='table')
        instance._get_diff_and_nan_column_names = Mock(return_value=('a#b', 'a#b.nan_component'))

        # Run
        instance._fit(data, metadata)

        # Assert
        assert instance._is_datetime is True
        assert instance._dtype == pd.Series([1]).dtype  # exact dtype (32 or 64) depends on OS
        assert instance._low_datetime_format == '%y %m, %d'
        assert instance._high_datetime_format == '%y %m %d'
        assert instance._diff_column_name == 'a#b'
        assert instance._nan_column_name == 'a#b.nan_component'
        instance._get_diff_and_nan_column_names.assert_called_once_with(metadata, 'a#b', 'table')

    @pytest.mark.parametrize(
        'dtype',
        ['float16', 'float32', 'float64', 'Float64', 'Float32', 'Int64', 'Int32', 'Int16', 'Int8'],
    )
    def test__fit_numerical(self, dtype):
        """Test it for numerical columns."""
        # Setup
        table_data = {'table': pd.DataFrame({'a': [1, 2, 4], 'b': [4, 5, 6]}, dtype=dtype)}
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'a': {'sdtype': 'numerical'},
                        'b': {'sdtype': 'numerical'},
                    },
                }
            }
        })
        instance = Inequality(low_column_name='a', high_column_name='b', table_name='table')

        # Run
        instance._fit(table_data, metadata)

        # Assert
        assert instance._dtype == dtype
        assert instance._is_datetime is False
        assert instance._low_datetime_format is None
        assert instance._high_datetime_format is None

    def test__fit_datetime(self):
        """Test it for datetime strings."""
        # Setup
        table_data = {
            'table': pd.DataFrame({
                'a': pd.to_datetime(['2020-01-01']),
                'b': pd.to_datetime(['2020-01-02']),
            })
        }
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'a': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
                        'b': {'sdtype': 'datetime'},
                    },
                }
            }
        })
        instance = Inequality(low_column_name='a', high_column_name='b', table_name='table')

        # Run
        instance._fit(table_data, metadata)

        # Assert
        assert instance._dtype == np.dtype('<M8[ns]')
        assert instance._is_datetime is True
        assert instance._low_datetime_format == '%Y-%m-%d'
        assert instance._high_datetime_format is None

    def test__transform(self):
        """Test it transforms the data correctly."""
        # Setup
        table_data = {
            'table': pd.DataFrame({
                'a': [1, 2, 3],
                'b': [4, 5, 6],
                'c': [7, 8, 9],
            })
        }
        instance = Inequality(low_column_name='a', high_column_name='b', table_name='table')
        instance._diff_column_name = 'a#b'
        instance._nan_column_name = 'a#b.nan_component'

        # Run
        out = instance._transform(table_data)

        # Assert
        out = out['table']
        expected_out = pd.DataFrame({
            'a': [1, 2, 3],
            'c': [7, 8, 9],
            'a#b': [np.log(4)] * 3,
            'a#b.nan_component': [None] * 3,
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test__transform_with_nans(self):
        """Test it transforms the data correctly when it contains nans."""
        # Setup
        instance = Inequality(low_column_name='a', high_column_name='b', table_name='table')
        instance._diff_column_name = 'a#b'
        instance._nan_column_name = 'a#b.nan_component'

        table_data_with_nans = {
            'table': pd.DataFrame({
                'a': [1, np.nan, 3, np.nan],
                'b': [np.nan, 2, 4, np.nan],
            })
        }

        table_data_without_nans = {'table': pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})}

        # Run
        output_with_nans = instance._transform(table_data_with_nans)
        output_without_nans = instance._transform(table_data_without_nans)

        # Assert
        output_with_nans = output_with_nans['table']
        output_without_nans = output_without_nans['table']
        expected_output_with_nans = pd.DataFrame({
            'a': [1.0, 2.0, 3.0, 2.0],
            'a#b': [np.log(2)] * 4,
            'a#b.nan_component': ['b', 'a', 'None', 'a, b'],
        })

        expected_output_without_nans = pd.DataFrame({
            'a': [1, 2, 3],
            'a#b': [np.log(2)] * 3,
            'a#b.nan_component': [None] * 3,
        })

        pd.testing.assert_frame_equal(output_with_nans, expected_output_with_nans)
        pd.testing.assert_frame_equal(output_without_nans, expected_output_without_nans)

    @patch('sdv.cag.inequality._create_unique_name')
    def test__get_new_column_names(self, mock_create_unique_name):
        """Test ``_get_diff_and_nan_column_names`` method."""
        # Setup
        mock_create_unique_name.side_effect = ['a#b', 'a#b.nan_component']
        instance = Inequality(low_column_name='a', high_column_name='b', table_name='table')
        metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'a': {'sdtype': 'numerical'},
                        'b': {'sdtype': 'numerical'},
                        'c': {'sdtype': 'id'},
                    },
                    'column_relationships': [{'type': 'relationship', 'column_names': ['a', 'b']}],
                }
            }
        })

        # Run
        diff_column, nan_column = instance._get_diff_and_nan_column_names(metadata, 'a#b', 'table')

        # Assert
        assert diff_column == 'a#b'
        assert nan_column == 'a#b.nan_component'
        mock_create_unique_name.assert_has_calls([
            call('a#b', {'a', 'b', 'c'}),
            call('a#b.nan_component', {'a', 'b', 'c'}),
        ])

    def test__transform_datetime(self):
        """Test it transforms the data correctly when it contains datetimes."""
        # Setup
        table_data = {
            'table': pd.DataFrame({
                'a': pd.to_datetime(['2020-01-01T00:00:00', '2020-01-02T00:00:00']),
                'b': pd.to_datetime(['2020-01-01T00:00:01', '2020-01-02T00:00:01']),
                'c': [1, 2],
            })
        }
        instance = Inequality(low_column_name='a', high_column_name='b', table_name='table')
        instance._diff_column_name = 'a#b'
        instance._nan_column_name = 'a#b.nan_component'
        instance._is_datetime = True

        # Run
        out = instance._transform(table_data)

        # Assert
        out = out['table']
        expected_out = pd.DataFrame({
            'a': pd.to_datetime(['2020-01-01T00:00:00', '2020-01-02T00:00:00']),
            'c': [1, 2],
            'a#b': [np.log(1_000_000_001), np.log(1_000_000_001)],
            'a#b.nan_component': [None, None],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test__transform_datetime_dtype_object(self):
        """Test it transforms the data correctly when the dtype is object."""
        # Setup
        table_data = {
            'table': pd.DataFrame({
                'a': ['2020-01-01T00:00:00', '2020-01-02T00:00:00'],
                'b': ['2020-01-01T00:00:01', '2020-01-02T00:00:01'],
                'c': [1, 2],
            })
        }
        instance = Inequality(low_column_name='a', high_column_name='b', table_name='table')
        instance._diff_column_name = 'a#b'
        instance._nan_column_name = 'a#b.nan_component'
        instance._is_datetime = True
        instance._dtype = 'O'

        # Run
        out = instance._transform(table_data)

        # Assert
        out = out['table']
        expected_out = pd.DataFrame({
            'a': ['2020-01-01T00:00:00', '2020-01-02T00:00:00'],
            'c': [1, 2],
            'a#b': [np.log(1_000_000_001), np.log(1_000_000_001)],
            'a#b.nan_component': [None] * 2,
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform(self):
        """Test it reverses the transformation correctly."""
        # Setup
        transformed = {
            'table': pd.DataFrame({
                'a': [1, 2, 3],
                'c': [7, 8, 9],
                'a#b': [np.log(4)] * 3,
                'a#b.nan_component': ['None'] * 3,
            })
        }
        instance = Inequality(low_column_name='a', high_column_name='b', table_name='table')
        instance._dtype = pd.Series([1]).dtype  # exact dtype (32 or 64) depends on OS
        instance._diff_column_name = 'a#b'
        instance._nan_column_name = 'a#b.nan_component'
        instance._original_data_columns = {'table': ['a', 'b', 'c']}
        instance._dtypes = {
            'table': {
                'a': pd.Series([1]).dtype,
                'b': pd.Series([1]).dtype,
                'c': pd.Series([1]).dtype,
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
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform_floats(self):
        """Test it reverses the transformation correctly when the dtype is float."""
        # Setup
        transformed = {
            'table': pd.DataFrame({
                'a': [1.1, 2.2, 3.3],
                'c': [7, 8, 9],
                'a#b': [np.log(4)] * 3,
                'a#b.nan_component': ['None'] * 3,
            })
        }
        instance = Inequality(low_column_name='a', high_column_name='b', table_name='table')
        instance._dtype = np.dtype('float')
        instance._diff_column_name = 'a#b'
        instance._nan_column_name = 'a#b.nan_component'
        instance._original_data_columns = {'table': ['a', 'b', 'c']}
        instance._dtypes = {
            'table': {
                'a': np.dtype('float'),
                'b': np.dtype('float'),
                'c': pd.Series([1]).dtype,
            }
        }

        # Run
        out = instance.reverse_transform(transformed)

        # Assert
        out = out['table']
        expected_out = pd.DataFrame({
            'a': [1.1, 2.2, 3.3],
            'b': [4.1, 5.2, 6.3],
            'c': [7, 8, 9],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform_datetime(self):
        """Test it reverses the transformation correctly when the dtype is datetime."""
        # Setup
        transformed = {
            'table': pd.DataFrame({
                'a': pd.to_datetime(['2020-01-01T00:00:00', '2020-01-02T00:00:00']),
                'c': [1, 2],
                'a#b': [np.log(1_000_000_001), np.log(1_000_000_001)],
                'a#b.nan_component': ['None', 'None'],
            })
        }
        instance = Inequality(low_column_name='a', high_column_name='b', table_name='table')
        instance._dtype = np.dtype('<M8[ns]')
        instance._diff_column_name = 'a#b'
        instance._nan_column_name = 'a#b.nan_component'
        instance._is_datetime = True
        instance._original_data_columns = {'table': ['a', 'b', 'c']}
        instance._dtypes = {
            'table': {
                'a': np.dtype('<M8[ns]'),
                'b': np.dtype('<M8[ns]'),
                'c': pd.Series([1]).dtype,
            }
        }

        # Run
        out = instance.reverse_transform(transformed)

        # Assert
        out = out['table']
        expected_out = pd.DataFrame({
            'a': pd.to_datetime(['2020-01-01T00:00:00', '2020-01-02T00:00:00']),
            'b': pd.to_datetime(['2020-01-01T00:00:01', '2020-01-02T00:00:01']),
            'c': [1, 2],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform_datetime_dtype_is_object(self):
        """Test it reverses the transformation correctly when the dtype is object."""
        # Setup
        transformed = {
            'table': pd.DataFrame({
                'a': ['2020-01-01T00:00:00', '2020-01-02T00:00:00'],
                'c': [1, 2],
                'a#b': [np.log(1_000_000_001), np.log(1_000_000_001)],
                'a#b.nan_component': ['None', 'None'],
            })
        }
        instance = Inequality(low_column_name='a', high_column_name='b', table_name='table')
        instance._dtype = np.dtype('O')
        instance._diff_column_name = 'a#b'
        instance._nan_column_name = 'a#b.nan_component'
        instance._is_datetime = True
        instance._original_data_columns = {'table': ['a', 'b', 'c']}
        instance._dtypes = {
            'table': {
                'a': np.dtype('O'),
                'b': np.dtype('O'),
                'c': pd.Series([1]).dtype,
            }
        }

        # Run
        out = instance.reverse_transform(transformed)

        # Assert
        out = out['table']
        expected_out = pd.DataFrame({
            'a': ['2020-01-01T00:00:00', '2020-01-02T00:00:00'],
            'b': [pd.Timestamp('2020-01-01 00:00:01'), pd.Timestamp('2020-01-02 00:00:01')],
            'c': [1, 2],
        })
        expected_out['b'] = expected_out['b'].astype(np.dtype('O'))
        pd.testing.assert_frame_equal(out, expected_out)

    def test__is_valid(self):
        """Test it checks if the data is valid."""
        # Setup
        table_data = {
            'table': pd.DataFrame({
                'a': [1, np.nan, 3, 4, None, 6, 8, 0],
                'b': [4, 2, 2, 4, np.nan, -6, 10, float('nan')],
                'c': [7, 8, 9, 10, 11, 12, 13, 14],
            })
        }
        instance = Inequality(low_column_name='a', high_column_name='b', table_name='table')
        instance._fitted = True

        # Run
        out = instance.is_valid(table_data)

        # Assert
        out = out['table']
        expected_out = [True, True, False, True, True, False, True, True]
        np.testing.assert_array_equal(expected_out, out)

    def test_is_valid_strict_boundaries_true(self):
        """Test it checks if the data is valid when strict boundaries are True."""
        # Setup
        table_data = {
            'table': pd.DataFrame({
                'a': [1, np.nan, 3, 4, None, 6, 8, 0],
                'b': [4, 2, 2, 4, np.nan, -6, 10, float('nan')],
                'c': [7, 8, 9, 10, 11, 12, 13, 14],
            })
        }
        instance = Inequality(
            low_column_name='a',
            high_column_name='b',
            strict_boundaries=True,
            table_name='table',
        )
        instance._fitted = True

        # Run
        out = instance.is_valid(table_data)

        # Assert
        out = out['table']
        expected_out = [True, True, False, False, True, False, True, True]
        np.testing.assert_array_equal(expected_out, out)

    def test_is_valid_datetimes(self):
        """Test it checks if the data is valid when it contains datetimes."""
        # Setup
        table_data = {
            'table': pd.DataFrame({
                'a': [datetime(2020, 5, 17), datetime(2021, 9, 1), np.nan],
                'b': [datetime(2020, 5, 18), datetime(2020, 9, 2), datetime(2020, 9, 2)],
                'c': [7, 8, 9],
            })
        }
        instance = Inequality(low_column_name='a', high_column_name='b', table_name='table')
        instance._fitted = True

        # Run
        out = instance.is_valid(table_data)

        # Assert
        out = out['table']
        expected_out = [True, False, True]
        np.testing.assert_array_equal(expected_out, out)

    def test_is_valid_datetime_objects(self):
        """Test the ``is_valid`` method with datetimes that are as ``dtype`` object."""
        # Setup
        table_data = {
            'table': pd.DataFrame({
                'a': ['2020-5-17', '2021-9-1', np.nan],
                'b': ['2020-5-18', '2020-9-2', '2020-9-2'],
                'c': [7, 8, 9],
            })
        }
        instance = Inequality(low_column_name='a', high_column_name='b', table_name='table')
        instance._is_datetime = True
        instance._dtype = 'O'
        instance._fitted = True

        # Run
        out = instance.is_valid(table_data)

        # Assert
        out = out['table']
        expected_out = [True, False, True]
        np.testing.assert_array_equal(expected_out, out)

    @patch('sdv.cag.inequality.match_datetime_precision')
    def test_is_valid_datetimes_miss_matching_datetime_formats(self, mock_match_datetime_precision):
        """Test it validates the data when it contains datetimes with missmatching formats."""
        # Setup
        table_data = {
            'table': pd.DataFrame({
                'SUBMISSION_TIMESTAMP': [
                    '2016-07-10 17:04:00',
                    '2016-07-11 13:23:00',
                    '2016-07-12 08:45:30',
                    '2016-07-11 12:00:00',
                    '2016-07-12 10:30:00',
                ],
                'DUE_DATE': ['2016-07-10', '2016-07-11', '2016-07-12', '2016-07-13', '2016-07-14'],
                'RANDOM_VALUE': [7, 8, 9, 10, 11],
            })
        }
        instance = Inequality(
            low_column_name='SUBMISSION_TIMESTAMP',
            high_column_name='DUE_DATE',
            table_name='table',
        )
        low_return = np.array([
            datetime(2020, 5, 18),
            datetime(2020, 9, 2),
            datetime(2020, 9, 2),
            datetime(2020, 5, 18),
            datetime(2020, 9, 2),
        ])
        high_return = np.array([
            datetime(2020, 5, 17),
            datetime(2021, 9, 1),
            datetime(2020, 5, 17),
            datetime(2021, 9, 1),
            datetime(2021, 9, 1),
        ])
        instance._dtype = 'O'
        instance._is_datetime = True
        instance._low_datetime_format = '%Y-%m-%d %H:%M:%S'
        instance._high_datetime_format = '%Y-%m-%d'
        mock_match_datetime_precision.return_value = (low_return, high_return)
        instance._fitted = True

        # Run
        out = instance.is_valid(table_data)

        # Assert
        out = out['table']
        expected_out = [False, True, False, True, True]
        np.testing.assert_array_equal(expected_out, out)

        expected_low = np.array(
            [
                '2016-07-10T17:04:00.000000000',
                '2016-07-11T13:23:00.000000000',
                '2016-07-12T08:45:30.000000000',
                '2016-07-11T12:00:00.000000000',
                '2016-07-12T10:30:00.000000000',
            ],
            dtype='datetime64[ns]',
        )

        expected_high = np.array(
            [
                '2016-07-10T00:00:00.000000000',
                '2016-07-11T00:00:00.000000000',
                '2016-07-12T00:00:00.000000000',
                '2016-07-13T00:00:00.000000000',
                '2016-07-14T00:00:00.000000000',
            ],
            dtype='datetime64[ns]',
        )

        call_low = mock_match_datetime_precision.call_args_list[0][1].pop('low')
        call_high = mock_match_datetime_precision.call_args_list[0][1].pop('high')
        np.testing.assert_array_equal(expected_low, call_low)
        np.testing.assert_array_equal(expected_high, call_high)
        expected_formats = {
            'low_datetime_format': '%Y-%m-%d %H:%M:%S',
            'high_datetime_format': '%Y-%m-%d',
        }
        assert expected_formats == mock_match_datetime_precision.call_args_list[0][1]
