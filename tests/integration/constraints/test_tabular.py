import numpy as np
import pandas as pd

from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer


def test_fixed_combinations_integers():
    """Test that FixedCombinations constraint works with integer columns."""
    data = pd.DataFrame({
        'A': [1, 2, 3, 1, 2, 1],
        'B': [10, 20, 30, 10, 20, 10],
    })
    metadata = SingleTableMetadata().load_from_dict({
        'columns': {
            'A': {'sdtype': 'categorical'},
            'B': {'sdtype': 'categorical'},
        }
    })

    synthesizer = GaussianCopulaSynthesizer(metadata)
    my_constraint = {
        'constraint_class': 'FixedCombinations',
        'constraint_parameters': {'column_names': ['A', 'B']},
    }
    synthesizer.add_constraints(constraints=[my_constraint])

    # Run
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample(1000)

    # Assert
    assert len(synthetic_data) == 1000
    pd.testing.assert_frame_equal(
        synthetic_data.drop_duplicates(ignore_index=True),
        data.drop_duplicates(ignore_index=True),
        check_like=True,
    )


def test_fixed_combinations_with_nans():
    """Test that FixedCombinations constraint works with NaNs."""
    # Setup
    data = pd.DataFrame({
        'A': [1, 2, np.nan, 1, 2, 1],
        'B': [10, 20, 30, 10, 20, 10],
    })
    metadata = SingleTableMetadata().load_from_dict({
        'columns': {
            'A': {'sdtype': 'categorical'},
            'B': {'sdtype': 'categorical'},
        }
    })

    synthesizer = GaussianCopulaSynthesizer(metadata)
    my_constraint = {
        'constraint_class': 'FixedCombinations',
        'constraint_parameters': {'column_names': ['A', 'B']},
    }
    synthesizer.add_constraints(constraints=[my_constraint])

    # Run
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample(1000)

    # Assert
    assert len(synthetic_data) == 1000
    pd.testing.assert_frame_equal(
        synthetic_data.drop_duplicates(ignore_index=True),
        data.drop_duplicates(ignore_index=True),
        check_like=True,
    )


def test_fixedincrements_with_nullable_pandas_dtypes():
    """Test that FixedIncrements constraint works with nullable pandas dtypes."""
    # Setup
    data = pd.DataFrame({
        'UInt8': pd.Series([1, pd.NA, 3], dtype='UInt8') * 10,
        'UInt16': pd.Series([1, pd.NA, 4], dtype='UInt16') * 10,
        'UInt32': pd.Series([1, pd.NA, 5], dtype='UInt32') * 10,
        'UInt64': pd.Series([1, pd.NA, 6], dtype='UInt64') * 10,
    })
    metadata = SingleTableMetadata().load_from_dict({
        'columns': {
            'UInt8': {'sdtype': 'numerical', 'computer_representation': 'UInt8'},
            'UInt16': {'sdtype': 'numerical', 'computer_representation': 'UInt16'},
            'UInt32': {'sdtype': 'numerical', 'computer_representation': 'UInt32'},
            'UInt64': {'sdtype': 'numerical', 'computer_representation': 'UInt64'},
        }
    })
    gcs = GaussianCopulaSynthesizer(metadata)
    my_constraints = [
        {
            'constraint_class': 'FixedIncrements',
            'constraint_parameters': {'column_name': column, 'increment_value': 10},
        }
        for column in data.columns
    ]
    gcs.add_constraints(my_constraints)

    # Run
    gcs.fit(data)
    synthetic_data = gcs.sample(10)

    # Assert
    synthetic_data.dtypes.to_dict() == data.dtypes.to_dict()
    for column in data.columns:
        assert np.all(synthetic_data[column] % 10 == 0)
