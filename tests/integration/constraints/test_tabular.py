import numpy as np
import pandas as pd
import pytest

from sdv.errors import ConstraintsNotMetError
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer


def test_fixed_combinations_integers():
    """Test that FixedCombinations constraint works with integer columns."""
    data = pd.DataFrame({
        'A': [1, 2, 3, 1, 2, 1],
        'B': [10, 20, 30, 10, 20, 10],
    })
    metadata = Metadata.load_from_dict({
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
    metadata = Metadata.load_from_dict({
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
    metadata = Metadata.load_from_dict({
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


def test_inequality_constraint_with_timestamp_and_date():
    """Test that the inequality constraint passes without strict boundaries.

    This test checks if the `Inequality` constraint can handle two columns
    with different datetime formats when `strict_boundaries` is set to `False`.
    The constraint allows the `SUBMISSION_TIMESTAMP` column to be less than
    or equal to the `DUE_DATE` column, even when they differ in precision but end
    within the same day.
    """
    # Setup
    data = pd.DataFrame(
        data={
            'SUBMISSION_TIMESTAMP': [
                '2016-07-10 17:04:00',
                '2016-07-11 13:23:00',
                '2016-07-12 08:45:30',
                '2016-07-11 12:00:00',
                '2016-07-12 10:30:00',
            ],
            'DUE_DATE': ['2016-07-10', '2016-07-11', '2016-07-12', '2016-07-13', '2016-07-14'],
        }
    )

    metadata = Metadata.load_from_dict({
        'tables': {
            'table': {
                'columns': {
                    'SUBMISSION_TIMESTAMP': {
                        'sdtype': 'datetime',
                        'datetime_format': '%Y-%m-%d %H:%M:%S',
                    },
                    'DUE_DATE': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
                }
            }
        }
    })
    synthesizer = GaussianCopulaSynthesizer(metadata)

    constraint = {
        'constraint_class': 'Inequality',
        'constraint_parameters': {
            'low_column_name': 'SUBMISSION_TIMESTAMP',
            'high_column_name': 'DUE_DATE',
            'strict_boundaries': False,
        },
    }

    synthesizer.add_constraints([constraint])

    # Run
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample(num_rows=10)

    # Assert
    synthetic_data['SUBMISSION_TIMESTAMP'] = pd.to_datetime(
        synthetic_data['SUBMISSION_TIMESTAMP'], errors='coerce'
    )
    synthetic_data['DUE_DATE'] = pd.to_datetime(synthetic_data['DUE_DATE'], errors='coerce')
    invalid_rows = synthetic_data[
        synthetic_data['SUBMISSION_TIMESTAMP'].dt.date > synthetic_data['DUE_DATE'].dt.date
    ]
    assert invalid_rows.empty


def test_inequality_constraint_with_timestamp_and_date_strict_boundaries():
    """Test that the inequality constraint fails with strict boundaries.

    This test evaluates the `Inequality` constraint when `strict_boundaries`
    is set to `True`. The `SUBMISSION_TIMESTAMP` column values must be strictly
    less than the `DUE_DATE` values to satisfy the constraint. If any
    `SUBMISSION_TIMESTAMP` matches or exceeds the `DUE_DATE`, an error should
    be raised.
    """
    # Setup
    data = pd.DataFrame(
        data={
            'SUBMISSION_TIMESTAMP': [
                '2016-07-10 17:04:00',
                '2016-07-11 13:23:00',
                '2016-07-12 08:45:30',
                '2016-07-11 12:00:00',
                '2016-07-12 10:30:00',
            ],
            'DUE_DATE': ['2016-07-10', '2016-07-11', '2016-07-12', '2016-07-13', '2016-07-14'],
        }
    )

    metadata = Metadata.load_from_dict({
        'tables': {
            'table': {
                'columns': {
                    'SUBMISSION_TIMESTAMP': {
                        'sdtype': 'datetime',
                        'datetime_format': '%Y-%m-%d %H:%M:%S',
                    },
                    'DUE_DATE': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
                }
            }
        }
    })

    synthesizer = GaussianCopulaSynthesizer(metadata)
    constraint = {
        'constraint_class': 'Inequality',
        'constraint_parameters': {
            'low_column_name': 'SUBMISSION_TIMESTAMP',
            'high_column_name': 'DUE_DATE',
            'strict_boundaries': True,
        },
    }
    synthesizer.add_constraints([constraint])

    # Run and Assert
    error_msg = "Data is not valid for the 'Inequality' constraint: "
    with pytest.raises(ConstraintsNotMetError) as error:
        synthesizer.fit(data)
        assert error_msg in error
