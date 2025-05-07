"""Module for testing single table synthesizers with constraints."""

import logging
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from copulas.multivariate.gaussian import GaussianMultivariate

from sdv.constraints import Constraint, create_custom_constraint_class
from sdv.constraints.errors import AggregateConstraintsError
from sdv.datasets.demo import download_demo
from sdv.metadata.metadata import Metadata
from sdv.sampling import Condition
from sdv.single_table import GaussianCopulaSynthesizer
from tests.integration.single_table.custom_constraints import MyConstraint


def _isinstance_side_effect(*args, **kwargs):
    if isinstance(args[0], GaussianMultivariate):
        return True
    else:
        return isinstance(args[0], args[1])


DEMO_DATA, DEMO_METADATA = download_demo(modality='single_table', dataset_name='fake_hotel_guests')


@pytest.fixture
def demo_data():
    return DEMO_DATA


@pytest.fixture
def demo_metadata():
    return DEMO_METADATA


def test_fit_with_unique_constraint_on_data_with_only_index_column():
    """Test that the ``fit`` method runs without error when metadata specifies unique constraint,
    ``fit`` is called on data containing a column named index.

    The ``fit`` method is expected to fit the model to data,
    taking into account the metadata and the ``Unique`` constraint.

    Setup:
    - The model is passed the unique constraint and
    the primary key column.

    Input:
    - Data, Unique constraint

    Github Issue:
    - Tests that https://github.com/sdv-dev/SDV/issues/616 does not occur
    """
    # Setup
    test_df = pd.DataFrame({
        'key': [
            1,
            2,
            3,
            4,
            5,
        ],
        'index': [
            'A',
            'B',
            'C',
            'D',
            'E',
        ],
    })

    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('key', 'table', sdtype='id')
    metadata.add_column('index', 'table', sdtype='categorical')
    metadata.set_primary_key('key', 'table')

    model = GaussianCopulaSynthesizer(metadata)
    constraint = {
        'constraint_class': 'Unique',
        'constraint_parameters': {'column_names': ['index']},
    }
    model.add_constraints([constraint])

    # Run
    model.fit(test_df)
    samples = model.sample(2)

    # Assert
    assert len(samples) == 2
    assert samples['index'].is_unique


@pytest.mark.skip('Old-style constraints are deprecated')
def test_fit_with_unique_constraint_on_data_which_has_index_column():
    """Test that the ``fit`` method runs without error when metadata specifies unique constraint,
    ``fit`` is called on data containing a column named index and other columns.

    The ``fit`` method is expected to fit the model to data,
    taking into account the metadata and the ``Unique`` constraint.

    Setup:
    - The model is passed the unique constraint and
    the primary key column.
    - The unique constraint is set on the ``test_column``

    Input:
    - Data, Unique constraint

    Github Issue:
    - Tests that https://github.com/sdv-dev/SDV/issues/616 does not occur
    """
    # Setup
    test_df = pd.DataFrame({
        'key': [
            1,
            2,
            3,
            4,
            5,
        ],
        'index': [
            'A',
            'B',
            'C',
            'D',
            'E',
        ],
        'test_column': [
            'A1',
            'B2',
            'C3',
            'D4',
            'E5',
        ],
    })

    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('key', 'table', sdtype='id')
    metadata.add_column('index', 'table', sdtype='categorical')
    metadata.add_column('test_column', 'table', sdtype='categorical')
    metadata.set_primary_key('key', 'table')

    model = GaussianCopulaSynthesizer(metadata)
    constraint = {
        'constraint_class': 'Unique',
        'constraint_parameters': {'column_names': ['test_column']},
    }
    model.add_constraints([constraint])

    # Run
    model.fit(test_df)
    samples = model.sample(2)

    # Assert
    assert len(samples) == 2
    assert samples['test_column'].is_unique


def test_fit_with_unique_constraint_on_data_subset():
    """Test that the ``fit`` method runs without error when metadata specifies unique constraint,
    ``fit`` is called on a subset of the original data.

    The ``fit`` method is expected to fit the model to the subset of data,
    taking into account the metadata and the ``Unique`` constraint.

    Setup:
    - The model is passed a ``Unique`` constraint and is
    matched to a subset of the specified data.
    Subdividing the data results in missing indexes in the subset contained in the original data.

    Input:
    - Subset of data, unique constraint

    Github Issue:
    - Tests that https://github.com/sdv-dev/SDV/issues/610 does not occur
    """
    # Setup
    test_df = pd.DataFrame({
        'key': [
            1,
            2,
            3,
            4,
            5,
        ],
        'test_column': [
            'A',
            'B',
            'C',
            'D',
            'E',
        ],
    })

    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('key', 'table', sdtype='id')
    metadata.add_column('test_column', 'table', sdtype='categorical')
    metadata.set_primary_key('key', 'table')

    test_df = test_df.iloc[[1, 3, 4]]
    constraint = {
        'constraint_class': 'Unique',
        'constraint_parameters': {'column_names': ['test_column']},
    }
    model = GaussianCopulaSynthesizer(metadata)
    model.add_constraints([constraint])

    # Run
    model.fit(test_df)
    samples = model.sample(2)

    # Assert
    assert len(samples) == 2
    assert samples['test_column'].is_unique


def test_conditional_sampling_with_constraints():
    """Test constraints with conditional sampling. GH#1737"""
    # Setup
    data = pd.DataFrame(
        data={
            'A': [round(i, 2) for i in np.random.uniform(low=0, high=10, size=100)],
            'B': [round(i) for i in np.random.uniform(low=0, high=10, size=100)],
            'C': np.random.choice(['Yes', 'No', 'Maybe'], size=100),
        }
    )

    metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'numerical'},
            'B': {'sdtype': 'numerical'},
            'C': {'sdtype': 'categorical'},
        }
    })

    synth = GaussianCopulaSynthesizer(metadata)

    constraint = {
        'constraint_class': 'ScalarRange',
        'constraint_parameters': {
            'column_name': 'B',
            'low_value': 0,
            'high_value': 10,
            'strict_boundaries': False,
        },
    }

    my_condition = Condition(num_rows=250, column_values={'B': 1})

    # Run
    synth.add_constraints([constraint])
    synth.fit(data)
    samples = synth.sample_from_conditions([my_condition])

    # Assert
    assert samples.columns.tolist() == ['A', 'B', 'C']
    assert samples['A'].between(0, 10).all()
    assert samples['B'].unique() == [1]
    assert samples['C'].isin(['Yes', 'No', 'Maybe']).all()


@pytest.mark.skip('Old-style constraints are deprecated')
@patch('sdv.single_table.base.isinstance')
@patch('sdv.single_table.copulas.multivariate.GaussianMultivariate', spec_set=GaussianMultivariate)
def test_conditional_sampling_constraint_uses_reject_sampling(gm_mock, isinstance_mock):
    """Test that the ``sample`` method handles constraints with conditions.

    The ``sample`` method is expected to properly apply constraint
    transformations by dropping columns that cannot be conditonally sampled
    on due to them being part of a constraint.

    Setup:
    - The model is being passed a ``UniqueCombination`` constraint and then
    asked to sample with two conditions, one of which the constraint depends on.
    The constraint is expected to skip its transformations since only some of
    the columns are provided by the conditions and the model will use reject
    sampling to meet the constraint instead.

    Input:
    - Conditions
    Side Effects:
    - Correct columns to condition on are passed to underlying sample method
    """
    # Setup
    isinstance_mock.side_effect = _isinstance_side_effect
    data = pd.DataFrame({
        'city': ['LA', 'SF', 'CHI', 'LA', 'LA'],
        'state': ['CA', 'CA', 'IL', 'CA', 'CA'],
        'age': [27, 28, 26, 21, 30],
    })

    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('city', 'table', sdtype='categorical')
    metadata.add_column('state', 'table', sdtype='categorical')
    metadata.add_column('age', 'table', sdtype='numerical')

    model = GaussianCopulaSynthesizer(metadata)

    constraint = {
        'constraint_class': 'FixedCombinations',
        'constraint_parameters': {'column_names': ['city', 'state']},
    }
    model.add_constraints([constraint])
    sampled_numeric_data = [
        pd.DataFrame({'city#state': [0.1, 1, 0.75, 0.25, 0.25], 'age': [30, 30, 30, 30, 30]}),
        pd.DataFrame({'city#state': [0.75], 'age': [30]}),
    ]
    gm_mock.return_value.sample.side_effect = sampled_numeric_data
    model.fit(data)

    # Run
    conditions = [Condition({'age': 30, 'state': 'CA'}, num_rows=5)]
    sampled_data = model.sample_from_conditions(conditions=conditions)

    # Assert
    expected_transformed_conditions = {'age': 30}
    expected_data = pd.DataFrame({
        'city': ['LA', 'SF', 'LA', 'LA', 'SF'],
        'state': ['CA', 'CA', 'CA', 'CA', 'CA'],
        'age': [30, 30, 30, 30, 30],
    })
    sample_calls = model._model.sample.mock_calls
    assert len(sample_calls) == 2
    model._model.sample.assert_any_call(5, conditions=expected_transformed_conditions)
    pd.testing.assert_frame_equal(sampled_data, expected_data)


@pytest.mark.skip('Old-style constraints are deprecated')
def test_custom_constraints_from_file(tmpdir):
    """Ensure the correct loading for a custom constraint class defined in another file."""
    data = pd.DataFrame({
        'primary_key': ['user-000', 'user-001', 'user-002'],
        'pii_col': ['223 Williams Rd', '75 Waltham St', '77 Mass Ave'],
        'numerical_col': [2, 3, 4],
        'categorical_col': ['a', 'b', 'a'],
    })

    metadata = Metadata.detect_from_dataframes({'table': data})
    metadata.update_column(table_name='table', column_name='pii_col', sdtype='address', pii=True)
    synthesizer = GaussianCopulaSynthesizer(
        metadata, enforce_min_max_values=False, enforce_rounding=False
    )
    synthesizer.load_custom_constraint_classes(
        'tests/integration/single_table/custom_constraints.py', ['MyConstraint']
    )
    constraint = {
        'constraint_class': 'MyConstraint',
        'constraint_parameters': {'column_names': ['numerical_col']},
    }

    # Run
    synthesizer.add_constraints([constraint])
    processed_data = synthesizer.preprocess(data)

    # Assert Processed Data
    assert all(processed_data['numerical_col'] == data['numerical_col'] ** 2)

    # Run - Fit the model
    synthesizer.fit_processed_data(processed_data)

    # Run - sample
    sampled = synthesizer.sample(10)
    assert all(sampled['numerical_col'] > 1)

    # Run - Save and Sample
    synthesizer.save(tmpdir / 'test.pkl')
    loaded_instance = synthesizer.load(tmpdir / 'test.pkl')
    loaded_sampled = loaded_instance.sample(10)
    assert all(loaded_sampled['numerical_col'] > 1)


@pytest.mark.skip('Old-style constraints are deprecated')
def test_custom_constraints_from_object(tmpdir):
    """Ensure the correct loading for a custom constraint class passed as an object."""
    data = pd.DataFrame({
        'primary_key': ['user-000', 'user-001', 'user-002'],
        'pii_col': ['223 Williams Rd', '75 Waltham St', '77 Mass Ave'],
        'numerical_col': [2, 3, 4],
        'categorical_col': ['a', 'b', 'a'],
    })

    metadata = Metadata.detect_from_dataframes({'table': data})
    metadata.update_column(table_name='table', column_name='pii_col', sdtype='address', pii=True)
    synthesizer = GaussianCopulaSynthesizer(
        metadata, enforce_min_max_values=False, enforce_rounding=False
    )
    synthesizer.add_custom_constraint_class(MyConstraint, 'MyConstraint')
    constraint = {
        'constraint_class': 'MyConstraint',
        'constraint_parameters': {'column_names': ['numerical_col']},
    }

    # Run
    synthesizer.add_constraints([constraint])
    processed_data = synthesizer.preprocess(data)

    # Assert Processed Data
    assert all(processed_data['numerical_col'] == data['numerical_col'] ** 2)

    # Run - Fit the model
    synthesizer.fit_processed_data(processed_data)

    # Run - sample
    sampled = synthesizer.sample(10)
    assert all(sampled['numerical_col'] > 1)

    # Run - Save and Sample
    synthesizer.save(tmpdir / 'test.pkl')
    loaded_instance = synthesizer.load(tmpdir / 'test.pkl')
    loaded_sampled = loaded_instance.sample(10)
    assert all(loaded_sampled['numerical_col'] > 1)


@pytest.mark.skip('Old-style constraints are deprecated')
def test_synthesizer_with_inequality_constraint(demo_data, demo_metadata):
    """Ensure that the ``Inequality`` constraint can sample from the model."""
    # Setup
    synthesizer = GaussianCopulaSynthesizer(demo_metadata)
    checkin_lessthan_checkout = {
        'constraint_class': 'Inequality',
        'constraint_parameters': {
            'low_column_name': 'checkin_date',
            'high_column_name': 'checkout_date',
        },
    }

    synthesizer.add_constraints([checkin_lessthan_checkout])
    synthesizer.fit(demo_data)

    # Run and Assert
    sampled = synthesizer.sample(num_rows=500)
    synthesizer.validate(sampled)
    _sampled = sampled[~sampled['checkout_date'].isna()]
    assert all(pd.to_datetime(_sampled['checkin_date']) < pd.to_datetime(_sampled['checkout_date']))


@pytest.mark.skip('Old-style constraints are deprecated')
def test_inequality_constraint_with_datetimes_and_nones():
    """Test that the ``Inequality`` constraint works with ``None`` and ``datetime``."""
    # Setup
    data = pd.DataFrame(
        data={
            'A': [None, None, '2020-01-02', '2020-03-04'] * 2,
            'B': [None, '2021-03-04', '2021-12-31', None] * 2,
        }
    )

    metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
            'B': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
        }
    })

    metadata.validate()
    synth = GaussianCopulaSynthesizer(metadata)
    synth.add_constraints([
        {
            'constraint_class': 'Inequality',
            'constraint_parameters': {'low_column_name': 'A', 'high_column_name': 'B'},
        }
    ])
    synth.validate(data)

    # Run
    synth.fit(data)
    sampled = synth.sample(10)

    # Assert
    synth.validate(sampled)
    expected_sampled = pd.DataFrame({
        'A': [
            '2020-01-02',
            '2020-01-02',
            np.nan,
            np.nan,
            '2020-01-02',
            np.nan,
            '2020-01-02',
            np.nan,
            '2020-01-02',
            np.nan,
        ],
        'B': [
            '2021-12-30',
            '2021-12-30',
            '2021-12-30',
            '2021-12-30',
            np.nan,
            '2021-12-30',
            '2021-12-30',
            '2021-12-30',
            np.nan,
            '2021-12-30',
        ],
    })
    pd.testing.assert_frame_equal(expected_sampled, sampled)


@pytest.mark.skip('Old-style constraints are deprecated')
def test_scalar_inequality_constraint_with_datetimes_and_nones():
    """Test that the ``ScalarInequality`` constraint works with ``None`` and ``datetime``."""
    # Setup
    data = pd.DataFrame(
        data={
            'A': [None, None, '2020-01-02', '2020-03-04'],
            'B': [None, '2021-03-04', '2021-12-31', None],
        }
    )

    metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
            'B': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
        }
    })

    metadata.validate()
    synth = GaussianCopulaSynthesizer(metadata)
    synth.add_constraints([
        {
            'constraint_class': 'ScalarInequality',
            'constraint_parameters': {'column_name': 'A', 'relation': '>=', 'value': '2019-01-01'},
        }
    ])
    synth.validate(data)

    # Run
    synth.fit(data)
    sampled = synth.sample(5)

    # Assert
    synth.validate(sampled)
    expected_sampled = pd.DataFrame({
        'A': {
            0: np.nan,
            1: '2020-01-19',
            2: np.nan,
            3: np.nan,
            4: '2020-01-31',
        },
        'B': {
            0: np.nan,
            1: np.nan,
            2: np.nan,
            3: np.nan,
            4: '2021-06-06',
        },
    })
    pd.testing.assert_frame_equal(expected_sampled, sampled)


@pytest.mark.skip('Old-style constraints are deprecated')
def test_scalar_range_constraint_with_datetimes_and_nones():
    """Test that the ``ScalarRange`` constraint works with ``None`` and ``datetime``."""
    # Setup
    data = pd.DataFrame(
        data={
            'A': [None, None, '2020-01-02', '2020-03-04'],
            'B': [None, '2021-03-04', '2021-12-31', None],
        }
    )

    metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
            'B': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
        }
    })

    metadata.validate()
    synth = GaussianCopulaSynthesizer(metadata, default_distribution='truncnorm')
    synth.add_constraints([
        {
            'constraint_class': 'ScalarRange',
            'constraint_parameters': {
                'column_name': 'A',
                'low_value': '2019-10-30',
                'high_value': '2020-03-04',
                'strict_boundaries': False,
            },
        }
    ])
    synth.validate(data)

    # Run
    synth.fit(data)
    sampled = synth.sample(10)

    # Assert
    synth.validate(sampled)
    expected_sampled = pd.DataFrame({
        'A': {
            0: np.nan,
            1: np.nan,
            2: '2020-02-07',
            3: np.nan,
            4: '2020-02-29',
            5: '2020-02-29',
            6: np.nan,
            7: np.nan,
            8: '2020-01-26',
            9: '2020-02-02',
        },
        'B': {
            0: np.nan,
            1: np.nan,
            2: np.nan,
            3: np.nan,
            4: '2021-06-21',
            5: np.nan,
            6: '2021-09-28',
            7: '2021-06-19',
            8: np.nan,
            9: np.nan,
        },
    })
    pd.testing.assert_frame_equal(expected_sampled, sampled)


@pytest.mark.skip('Old-style constraints are deprecated')
def test_range_constraint_with_datetimes_and_nones():
    """Test that the ``Range`` constraint works with ``None`` and ``datetime``."""
    # Setup
    data = pd.DataFrame(
        data={
            'A': [None, None, '2020-01-02', '2020-03-04'],
            'B': [None, '2021-03-04', '2021-12-31', None],
            'C': [None, '2022-03-04', '2022-12-31', None],
        }
    )

    metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
            'B': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
            'C': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
        }
    })

    metadata.validate()
    synth = GaussianCopulaSynthesizer(metadata)
    synth.add_constraints([
        {
            'constraint_class': 'Range',
            'constraint_parameters': {
                'low_column_name': 'A',
                'middle_column_name': 'B',
                'high_column_name': 'C',
                'strict_boundaries': False,
            },
        }
    ])
    synth.validate(data)

    # Run
    synth.fit(data)
    sampled = synth.sample(10)

    # Assert
    expected_sampled = pd.DataFrame({
        'A': [
            '2020-01-02',
            '2020-01-02',
            np.nan,
            '2020-01-02',
            '2020-01-02',
            np.nan,
            '2020-01-02',
            '2020-01-02',
            '2020-01-02',
            np.nan,
        ],
        'B': [
            np.nan,
            np.nan,
            '2021-12-30',
            np.nan,
            np.nan,
            '2021-12-30',
            np.nan,
            '2021-12-30',
            np.nan,
            '2021-12-30',
        ],
        'C': [
            np.nan,
            np.nan,
            '2022-12-30',
            np.nan,
            np.nan,
            '2022-12-30',
            np.nan,
            '2022-12-30',
            np.nan,
            '2022-12-30',
        ],
    })
    pd.testing.assert_frame_equal(expected_sampled, sampled)
    synth.validate(sampled)


@pytest.mark.skip('Old-style constraints are deprecated')
def test_inequality_constraint_all_possible_nans_configurations():
    """Test that the inequality constraint works with all possible NaN configurations."""
    # Setup
    data = pd.DataFrame(data={'A': [0, 1, np.nan, np.nan, 2], 'B': [2, np.nan, 3, np.nan, 3]})

    metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'numerical'},
            'B': {'sdtype': 'numerical'},
        }
    })

    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.add_constraints([
        {
            'constraint_class': 'Inequality',
            'constraint_parameters': {'low_column_name': 'A', 'high_column_name': 'B'},
        }
    ])

    # Run
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample(10000)

    # Assert
    assert (~(pd.isna(synthetic_data['A'])) & ~(pd.isna(synthetic_data['B']))).any()
    assert ((pd.isna(synthetic_data['A'])) & ~(pd.isna(synthetic_data['B']))).any()
    assert (~(pd.isna(synthetic_data['A'])) & (pd.isna(synthetic_data['B']))).any()
    assert (~(pd.isna(synthetic_data['A'])) & ~(pd.isna(synthetic_data['B']))).any()


@pytest.mark.skip('Old-style constraints are deprecated')
def test_range_constraint_all_possible_nans_configurations():
    """Test that the range constraint works with all possible NaN configurations."""
    # Setup
    data = pd.DataFrame(
        data={
            'low': [1, 4, np.nan, 0, 4, np.nan, np.nan, 5, np.nan],
            'middle': [2, 5, 3, np.nan, 5, np.nan, 5, np.nan, np.nan],
            'high': [3, 7, 8, 4, np.nan, 9, np.nan, np.nan, np.nan],
        }
    )

    metadata_dict = {
        'columns': {
            'low': {'sdtype': 'numerical'},
            'middle': {'sdtype': 'numerical'},
            'high': {'sdtype': 'numerical'},
        }
    }

    metadata = Metadata.load_from_dict(metadata_dict)
    synthesizer = GaussianCopulaSynthesizer(metadata)

    my_constraint = {
        'constraint_class': 'Range',
        'constraint_parameters': {
            'low_column_name': 'low',
            'middle_column_name': 'middle',
            'high_column_name': 'high',
        },
    }

    # Run
    synthesizer.add_constraints(constraints=[my_constraint])
    synthesizer.fit(data)

    s_data = synthesizer.sample(2000)

    # Assert
    synt_data_not_nan_low_middle = s_data[~(pd.isna(s_data['low'])) & ~(pd.isna(s_data['middle']))]
    synt_data_not_nan_middle_high = s_data[
        ~(pd.isna(s_data['middle'])) & ~(pd.isna(s_data['high']))
    ]
    synt_data_not_nan_low_high = s_data[~(pd.isna(s_data['low'])) & ~(pd.isna(s_data['high']))]

    is_nan_low = pd.isna(s_data['low'])
    is_nan_middle = pd.isna(s_data['middle'])
    is_nan_high = pd.isna(s_data['high'])

    assert all(synt_data_not_nan_low_middle['low'] <= synt_data_not_nan_low_middle['middle'])
    assert all(synt_data_not_nan_middle_high['middle'] <= synt_data_not_nan_middle_high['high'])
    assert all(synt_data_not_nan_low_high['low'] <= synt_data_not_nan_low_high['high'])

    assert any(is_nan_low & is_nan_middle & is_nan_high)
    assert any(is_nan_low & is_nan_middle & ~is_nan_high)
    assert any(is_nan_low & ~is_nan_middle & is_nan_high)
    assert any(is_nan_low & ~is_nan_middle & ~is_nan_high)
    assert any(~is_nan_low & is_nan_middle & is_nan_high)
    assert any(~is_nan_low & is_nan_middle & ~is_nan_high)
    assert any(~is_nan_low & ~is_nan_middle & is_nan_high)
    assert any(~is_nan_low & ~is_nan_middle & ~is_nan_high)


def test_custom_constraint_with_key():
    """Test that a custom constraint can work with a primary key."""

    # Setup
    def is_valid(column_names, data):
        return data['key'] == data['letter'] + '_' + data['number']

    def transform(column_names, data):
        new_data = data.drop(['letter', 'number'], axis=1)
        return new_data

    def reverse_transform(column_names, data):
        columns = data['key'].str.split('_', expand=True)
        data['letter'] = columns[0]
        data['number'] = columns[1]
        return data

    custom_constraint = create_custom_constraint_class(
        is_valid_fn=is_valid, transform_fn=transform, reverse_transform_fn=reverse_transform
    )

    data = pd.DataFrame({
        'key': ['a_1', 'b_2', 'c_3'],
        'letter': ['a', 'b', 'c'],
        'number': ['1', '2', '3'],
        'other': [7, 8, 9],
    })
    metadata = Metadata.detect_from_dataframes({'table': data})
    metadata.update_column('key', 'table', sdtype='id', regex_format=r'\w_\d')
    metadata.set_primary_key('key', 'table')
    synth = GaussianCopulaSynthesizer(metadata)
    synth.add_custom_constraint_class(custom_constraint, 'custom')

    id_must_match = {
        'constraint_class': 'custom',
        'constraint_parameters': {
            'column_names': ['letter', 'number'],
        },
    }
    synth.add_constraints([id_must_match])

    # Run
    synth.fit(data)
    sampled = synth.sample(100)

    # Assert
    synth.validate(sampled)


def test_timezone_aware_constraints():
    """Test that constraints work with timezone aware datetime columns GH#1576."""
    # Setup
    data = pd.DataFrame({'col1': ['2020-02-02'], 'col2': ['2020-02-05']})
    data['col1'] = pd.to_datetime(data['col1']).dt.tz_localize('UTC')
    data['col2'] = pd.to_datetime(data['col2']).dt.tz_localize('UTC')

    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('col1', 'table', sdtype='datetime')
    metadata.add_column('col2', 'table', sdtype='datetime')

    my_constraint = {
        'constraint_class': 'Inequality',
        'constraint_parameters': {
            'low_column_name': 'col1',
            'high_column_name': 'col2',
            'strict_boundaries': True,
        },
    }

    # Run
    synth = GaussianCopulaSynthesizer(metadata)
    synth.add_constraints(constraints=[my_constraint])
    synth.fit(data)
    samples = synth.sample(100)

    # Assert
    assert all(samples['col1'] < samples['col2'])


@pytest.mark.skip('Old-style constraints are deprecated')
def test_custom_and_overlapping_constraint_errors(caplog, demo_data, demo_metadata):
    """Test a synthesizer when constraints overlap or custom constraints raise an error.

    If a custom constraint raises an error, we want to log it but not crash. If one constraint
    drops columns that another constraint needs, we also want to log this information but not
    crash.
    """
    # Setup
    synth = GaussianCopulaSynthesizer(demo_metadata)

    def is_valid(column_names, data):
        return ~pd.isna(data[column_names[0]])

    def transform(column_names, data):
        raise ValueError('Transform error')

    def reverse_transform(column_names, data):
        return data

    custom_constraint = create_custom_constraint_class(
        is_valid_fn=is_valid, transform_fn=transform, reverse_transform_fn=reverse_transform
    )
    synth.add_custom_constraint_class(custom_constraint, 'custom')
    checkin_checkout_constraint = {
        'constraint_class': 'Inequality',
        'constraint_parameters': {
            'low_column_name': 'checkin_date',
            'high_column_name': 'checkout_date',
        },
    }
    error_constraint = {
        'constraint_class': 'custom',
        'constraint_parameters': {
            'column_names': ['room_rate'],
        },
    }
    overlapped_constraint = {
        'constraint_class': 'ScalarInequality',
        'constraint_parameters': {
            'column_name': 'checkout_date',
            'relation': '>',
            'value': '01 Jan 1990',
        },
    }
    synth.add_constraints(
        constraints=[checkin_checkout_constraint, error_constraint, overlapped_constraint]
    )

    # Run
    with caplog.at_level(logging.INFO, logger='sdv.data_processing.data_processor'):
        synth.fit(demo_data)

    # Assert
    expected_logs = [
        "Unable to transform ScalarInequality with columns ['checkout_date'] because they are not "
        'all available in the data. This happens due to multiple, overlapping constraints.',
        "Unable to transform CustomConstraint with columns ['room_rate'] due to an error in "
        'transform: \nTransform error\nUsing the reject sampling approach instead.',
    ]
    log_messages = [record[2] for record in caplog.record_tuples]
    for log in expected_logs:
        assert log in log_messages


@pytest.mark.skip('Old-style constraints are deprecated')
def test_aggregate_constraint_errors(demo_data, demo_metadata):
    """Test that if there are multiple constraint errors, they are raised together."""

    # Setup
    class BadConstraint(Constraint):
        def __init__(self, column_name):
            self.column_name = column_name

        def _transform(self, table_data):
            raise ValueError('Bad constraint')

    synth = GaussianCopulaSynthesizer(demo_metadata)
    bad_constraint1 = {
        'constraint_class': 'BadConstraint',
        'constraint_parameters': {'column_name': 'room_rate'},
    }
    bad_constraint2 = {
        'constraint_class': 'BadConstraint',
        'constraint_parameters': {'column_name': 'checkin_date'},
    }
    synth.add_constraints(constraints=[bad_constraint1, bad_constraint2])

    # Run and Assert
    message = '\nBad constraint\n\nBad constraint'
    with pytest.raises(AggregateConstraintsError, match=message):
        synth.fit(demo_data)


@pytest.mark.skip('Old-style constraints are deprecated')
def test_constraint_datetime_check():
    """Test datetime columns are correctly identified in constraints. GH#1692"""
    # Setup
    data = {
        'table': pd.DataFrame(
            data={
                'low_col': ['21 Sep, 15', '23 Aug, 14', '29 May, 12'],
                'high_col': ['02 Nov, 15', '12 Oct, 14', '08 Jul, 12'],
            }
        )
    }
    metadata = Metadata.load_from_dict({
        'tables': {
            'table': {
                'columns': {
                    'low_col': {'sdtype': 'datetime', 'datetime_format': '%d %b, %y'},
                    'high_col': {'sdtype': 'datetime', 'datetime_format': '%d %b, %y'},
                }
            }
        }
    })

    my_constraint = {
        'constraint_class': 'Inequality',
        'constraint_parameters': {
            'low_column_name': 'low_col',
            'high_column_name': 'high_col',
            'strict_boundaries': False,
        },
    }

    # Run
    metadata.validate()
    metadata.validate_data(data)

    synth = GaussianCopulaSynthesizer(metadata)
    synth.add_constraints([my_constraint])
    synth.fit(data['table'])
    samples = synth.sample(3)

    # Assert
    expected_dataframe = pd.DataFrame({
        'low_col': ['18 Jul, 15', '09 Aug, 15', '24 Jun, 15'],
        'high_col': ['05 Sep, 15', '26 Sep, 15', '12 Aug, 15'],
    })
    pd.testing.assert_frame_equal(samples, expected_dataframe)
