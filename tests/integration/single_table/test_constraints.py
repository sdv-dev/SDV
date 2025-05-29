"""Module for testing single table synthesizers with constraints."""

import logging
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from copulas.multivariate.gaussian import GaussianMultivariate

from sdv.cag import FixedCombinations, Inequality, Range
from sdv.datasets.demo import download_demo
from sdv.metadata.metadata import Metadata
from sdv.multi_table import HMASynthesizer
from sdv.sampling import Condition
from sdv.single_table import GaussianCopulaSynthesizer
from tests.integration.single_table.custom_constraints import MyConstraint, MySingleTableConstraint


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


def test_conditional_sampling_with_constraints(demo_data, demo_metadata):
    """Test constraints with conditional sampling. GH#1737"""
    # Setup
    column_metadata = demo_metadata.tables['fake_hotel_guests'].columns['checkin_date']
    datetime_format = column_metadata['datetime_format']
    constraint = Inequality(
        low_column_name='checkin_date',
        high_column_name='checkout_date',
    )
    synth = GaussianCopulaSynthesizer(demo_metadata)
    my_condition = Condition(num_rows=250, column_values={'checkin_date': '04 Feb 2020'})

    # Run
    synth.add_constraints([constraint])
    synth.fit(demo_data)
    samples = synth.sample_from_conditions([my_condition])

    # Assert
    assert samples.columns.tolist() == demo_data.columns.to_list()
    assert all(samples['checkin_date'] == '04 Feb 2020')
    valid_dates = samples[['checkin_date', 'checkout_date']].dropna()
    checkin_dates = pd.to_datetime(valid_dates['checkin_date'], format=datetime_format)
    checkout_dates = pd.to_datetime(valid_dates['checkout_date'], format=datetime_format)
    assert all(checkin_dates <= checkout_dates)


def test_conditional_sampling_with_constraints_transforms_if_possible(demo_data, demo_metadata):
    """Test constraints with conditional sampling. GH#1737"""
    # Setup
    constraint = Inequality(
        low_column_name='checkin_date',
        high_column_name='checkout_date',
    )
    synth = GaussianCopulaSynthesizer(demo_metadata)
    my_condition = Condition(
        num_rows=250, column_values={'checkin_date': '04 Feb 2020', 'checkout_date': '10 Feb 2020'}
    )

    # Run
    synth.add_constraints([constraint])
    synth.fit(demo_data)
    samples = synth.sample_from_conditions([my_condition])

    # Assert
    assert samples.columns.tolist() == demo_data.columns.to_list()
    assert all(samples['checkin_date'] == '04 Feb 2020')
    assert all(samples['checkout_date'] == '10 Feb 2020')


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

    constraint = FixedCombinations(column_names=['city', 'state'])
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


def test_custom_constraints_from_object(tmpdir):
    """Ensure the correct loading for a custom constraint class passed as an object."""
    data = {
        'table': pd.DataFrame({
            'primary_key': ['user-000', 'user-001', 'user-002'],
            'pii_col': ['223 Williams Rd', '75 Waltham St', '77 Mass Ave'],
            'numerical_col': [2, 3, 4],
            'categorical_col': ['a', 'b', 'a'],
        }),
        'table2': pd.DataFrame({
            'numerical_col2': [22, 33, 44],
            'categorical_col2': ['aa', 'bb', 'aa'],
        }),
    }

    metadata = Metadata.detect_from_dataframes(data)
    metadata.update_column(table_name='table', column_name='pii_col', sdtype='address', pii=True)
    synthesizer = HMASynthesizer(metadata)
    constraint = MyConstraint(column_names=['numerical_col'], table_name='table')

    # Run
    synthesizer.add_constraints([constraint])
    processed_data = synthesizer.preprocess(data)

    # Assert Processed Data
    assert all(processed_data['table']['numerical_col'] == data['table']['numerical_col'] ** 2)

    # Run - Fit the model
    synthesizer.fit_processed_data(processed_data)

    # Run - sample
    sampled = synthesizer.sample(10)
    assert all(sampled['table']['numerical_col'] > 1)

    # Run - Save and Sample
    synthesizer.save(tmpdir / 'test.pkl')
    loaded_instance = synthesizer.load(tmpdir / 'test.pkl')
    loaded_sampled = loaded_instance.sample(10)
    assert all(loaded_sampled['table']['numerical_col'] > 1)


def test_single_table_custom_constraints_from_object(tmpdir):
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
    constraint = MySingleTableConstraint(column_names=['numerical_col'])

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


def test_synthesizer_with_inequality_constraint(demo_data, demo_metadata):
    """Ensure that the ``Inequality`` constraint can sample from the model."""
    # Setup
    column_metadata = demo_metadata.tables['fake_hotel_guests'].columns['checkin_date']
    datetime_format = column_metadata['datetime_format']
    synthesizer = GaussianCopulaSynthesizer(demo_metadata)
    checkin_lessthan_checkout = Inequality(
        low_column_name='checkin_date', high_column_name='checkout_date'
    )
    synthesizer.add_constraints([checkin_lessthan_checkout])
    synthesizer.fit(demo_data)

    # Run and Assert
    sampled = synthesizer.sample(num_rows=500)
    synthesizer.validate(sampled)
    _sampled = sampled[~sampled['checkout_date'].isna()]
    checkin_dates = pd.to_datetime(_sampled['checkin_date'], format=datetime_format)
    checkout_dates = pd.to_datetime(_sampled['checkout_date'], format=datetime_format)
    assert all(checkin_dates < checkout_dates)


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
    synth.add_constraints([Inequality(low_column_name='A', high_column_name='B')])
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
        Range(
            low_column_name='A',
            middle_column_name='B',
            high_column_name='C',
            strict_boundaries=False,
        )
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
    synthesizer.add_constraints([Inequality(low_column_name='A', high_column_name='B')])

    # Run
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample(10000)

    # Assert
    assert (~(pd.isna(synthetic_data['A'])) & ~(pd.isna(synthetic_data['B']))).any()
    assert ((pd.isna(synthetic_data['A'])) & ~(pd.isna(synthetic_data['B']))).any()
    assert (~(pd.isna(synthetic_data['A'])) & (pd.isna(synthetic_data['B']))).any()
    assert (~(pd.isna(synthetic_data['A'])) & ~(pd.isna(synthetic_data['B']))).any()


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

    my_constraint = Range(
        low_column_name='low', middle_column_name='middle', high_column_name='high'
    )

    # Run
    synthesizer.add_constraints([my_constraint])
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

    my_constraint = Inequality(
        low_column_name='col1', high_column_name='col2', strict_boundaries=True
    )

    # Run
    synth = GaussianCopulaSynthesizer(metadata)
    synth.add_constraints([my_constraint])
    synth.fit(data)
    samples = synth.sample(100)

    # Assert
    assert all(samples['col1'] < samples['col2'])


def test_overlapping_constraint_logs(caplog, demo_data, demo_metadata):
    """Test a synthesizer when constraints overlap .

    If one constraint drops columns that another constraint needs, we also want to log this
    information but not crash.
    """
    # Setup
    demo_metadata = Metadata.load_from_dict(demo_metadata.to_dict())
    column_metadata = demo_metadata.tables['fake_hotel_guests'].columns['checkout_date']
    datetime_format = column_metadata['datetime_format']
    demo_metadata.add_column('billing_date', sdtype='datetime', datetime_format=datetime_format)
    demo_data['billing_date'] = pd.to_datetime(demo_data['checkout_date']) + pd.Timedelta(5, 'D')
    demo_data['billing_date'] = demo_data['billing_date'].dt.strftime(datetime_format)

    synth = GaussianCopulaSynthesizer(demo_metadata)

    checkin_checkout_constraint = Inequality(
        low_column_name='checkin_date', high_column_name='checkout_date'
    )
    overlapped_constraint = Inequality(
        low_column_name='checkout_date', high_column_name='billing_date'
    )

    # Run
    with caplog.at_level(logging.INFO, logger='sdv.single_table.base'):
        synth.add_constraints([checkin_checkout_constraint, overlapped_constraint])

    synth.fit(demo_data)

    # Assert
    expected_logs = ['Enforcing constraint Inequality using reject sampling.']
    log_messages = [record[2] for record in caplog.record_tuples]
    for log in expected_logs:
        assert log in log_messages


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

    my_constraint = Inequality(
        low_column_name='low_col', high_column_name='high_col', strict_boundaries=False
    )

    # Run
    metadata.validate()
    metadata.validate_data(data)

    synth = GaussianCopulaSynthesizer(metadata)
    synth.add_constraints([my_constraint])
    synth.fit(data['table'])
    samples = synth.sample(3)

    # Assert
    expected_dataframe = pd.DataFrame({
        'low_col': ['13 Sep, 15', '19 Jan, 15', '02 Jun, 14'],
        'high_col': ['23 Oct, 15', '09 Mar, 15', '12 Jul, 14'],
    })
    pd.testing.assert_frame_equal(samples, expected_dataframe)
