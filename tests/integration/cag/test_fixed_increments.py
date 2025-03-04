import numpy as np
import pandas as pd

from sdv.cag import FixedIncrements
from sdv.metadata import Metadata


def test_fixed_increments_integers():
    # Setup
    increment_value = 5
    data = pd.DataFrame({
        'int8': pd.Series([1, 2, 3, 4, 5], dtype='int8') * increment_value,
        'int16': pd.Series([2, 4, 6, 8, 10], dtype='int16') * increment_value,
        'int32': pd.Series([10, 20, 30, 40, 50], dtype='int32') * increment_value,
        'int64': pd.Series([100, 200, 300, 400, 500], dtype='int64') * increment_value,
    })
    metadata = Metadata.load_from_dict({
        'columns': {
            'int8': {'sdtype': 'numerical', 'computer_representation': 'int8'},
            'int16': {'sdtype': 'numerical', 'computer_representation': 'int16'},
            'int32': {'sdtype': 'numerical', 'computer_representation': 'int32'},
            'int64': {'sdtype': 'numerical', 'computer_representation': 'int64'},
        }
    })
    pattern = FixedIncrements(column_name=['int8', 'int16', 'int32', 'int64'],
                              increment_value=increment_value)

    # Run
    pattern.validate(data, metadata)
    updated_metadata = pattern.get_updated_metadata(metadata)
    pattern.fit(data, metadata)
    transformed = pattern.transform(data)
    reverse_transformed = pattern.reverse_transform(transformed)

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'columns': {
            "int8#increment": {
                "sdtype": "numerical"
            },
            "int16#increment": {
                "sdtype": "numerical"
            },
            "int32#increment": {
                "sdtype": "numerical"
            },
            "int64#increment": {
                "sdtype": "numerical"
            }
        }
    }).to_dict()
    assert expected_updated_metadata == updated_metadata.to_dict()
    assert list(transformed.columns) == ['int8', 'int16', 'int32', 'int64']
    pd.testing.assert_frame_equal(data, reverse_transformed)

def test_fixed_increments_integers_with_nans():
    # Setup
    increment_value = 5
    data = pd.DataFrame({
        'Int8': pd.Series([1, 2, pd.NA, 4, 5], dtype='Int8') * increment_value,
        'Int16': pd.Series([2, pd.NA, pd.NA, 8, 10], dtype='Int16') * increment_value,
        'Int32': pd.Series([10, 20, 30, pd.NA, 50], dtype='Int32') * increment_value,
        'Int64': pd.Series([100, 200, 300, 400, pd.NA], dtype='Int64') * increment_value,
    })
    metadata = Metadata.load_from_dict({
        'columns': {
            'Int8': {'sdtype': 'numerical', 'computer_representation': 'Int8'},
            'Int16': {'sdtype': 'numerical', 'computer_representation': 'Int16'},
            'Int32': {'sdtype': 'numerical', 'computer_representation': 'Int32'},
            'Int64': {'sdtype': 'numerical', 'computer_representation': 'Int64'},
        }
    })
    pattern = FixedIncrements(column_name=['Int8', 'Int16', 'Int32', 'Int64'],
                              increment_value=increment_value)

    # Run
    pattern.validate(data, metadata)
    updated_metadata = pattern.get_updated_metadata(metadata)
    pattern.fit(data, metadata)
    transformed = pattern.transform(data)
    reverse_transformed = pattern.reverse_transform(transformed)

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'columns': {
            "Int8#increment": {
                "sdtype": "numerical"
            },
            "Int16#increment": {
                "sdtype": "numerical"
            },
            "Int32#increment": {
                "sdtype": "numerical"
            },
            "Int64#increment": {
                "sdtype": "numerical"
            }
        }
    }).to_dict()
    assert expected_updated_metadata == updated_metadata.to_dict()
    assert list(transformed.columns) == ['Int8', 'Int16', 'Int32', 'Int64']
    pd.testing.assert_frame_equal(data, reverse_transformed)


def test_fixed_increments_with_floats():
    # Setup
    increment_value = 5
    data = pd.DataFrame({
        'float16': pd.Series([1, 2, 3, 4, np.nan], dtype='float16') * increment_value,
        'float32': pd.Series([10, 20, 30, np.nan, 50], dtype='float32') * increment_value,
        'float64': pd.Series([100, 200, 300, 400, np.nan], dtype='float64') * increment_value,
    })
    metadata = Metadata.load_from_dict({
        'columns': {
            'float16': {'sdtype': 'numerical', 'computer_representation': 'float16'},
            'float32': {'sdtype': 'numerical', 'computer_representation': 'float32'},
            'float64': {'sdtype': 'numerical', 'computer_representation': 'float64'},
        }
    })
    pattern = FixedIncrements(column_name=['float16', 'float32', 'float64'],
                              increment_value=increment_value)

    # Run
    pattern.validate(data, metadata)
    updated_metadata = pattern.get_updated_metadata(metadata)
    pattern.fit(data, metadata)
    transformed = pattern.transform(data)
    reverse_transformed = pattern.reverse_transform(transformed)

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'columns': {
            "float16#increment": {
                "sdtype": "numerical"
            },
            "float32#increment": {
                "sdtype": "numerical"
            },
            "float64#increment": {
                "sdtype": "numerical"
            },
        }
    }).to_dict()
    assert expected_updated_metadata == updated_metadata.to_dict()
    assert list(transformed.columns) == ['float16', 'float32', 'float64']
    pd.testing.assert_frame_equal(data, reverse_transformed)

def test_fixed_incremements_with_multi_table():
    """Test that FixedIncrements constraint works with multi-table data."""
    # Setup
    increment_value = 10
    data = {
        'table1': pd.DataFrame({
            'Float32': pd.Series([1, pd.NA, 3, 1, 2, 1], dtype="Float32") * increment_value,
            'Float64': pd.Series([1, 2, pd.NA, pd.NA, 2, 1], dtype="Float64")  * increment_value,
        }),
        'table2': pd.DataFrame({'id': range(5)}),
    }
    metadata = Metadata.load_from_dict({
        'tables': {
            'table1': {
                'columns': {
                    'Float32': {'sdtype': 'numerical'},
                    'Float64': {'sdtype': 'numerical'},
                }
            },
            'table2': {
                'columns': {
                    'id': {'sdtype': 'id'},
                }
            },
        }
    })
    pattern = FixedIncrements(column_name=['Float32', 'Float64'], table_name='table1',
                              increment_value=increment_value)

    # Run
    pattern.validate(data, metadata)
    updated_metadata = pattern.get_updated_metadata(metadata)
    pattern.fit(data, metadata)
    transformed = pattern.transform(data)
    reverse_transformed = pattern.reverse_transform(transformed)

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'tables': {
            'table1': {
                'columns': {
                    'Float32#increment': {'sdtype': 'numerical'},
                    'Float64#increment': {'sdtype': 'numerical'},
                }
            },
            'table2': {
                'columns': {
                    'id': {'sdtype': 'id'},
                }
            },
        }
    }).to_dict()
    assert expected_updated_metadata == updated_metadata.to_dict()
    assert list(transformed['table1'].columns) == ['Float32', 'Float64']
    assert set(data.keys()) == set(reverse_transformed.keys())
    for table_name, table in data.items():
        pd.testing.assert_frame_equal(table, reverse_transformed[table_name])