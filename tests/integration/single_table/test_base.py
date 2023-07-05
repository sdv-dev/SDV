import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pkg_resources
import pytest
from copulas.multivariate.gaussian import GaussianMultivariate
from rdt.transformers import AnonymizedFaker, FloatFormatter, LabelEncoder, RegexGenerator

from sdv.datasets.demo import download_demo
from sdv.metadata import SingleTableMetadata
from sdv.sampling import Condition
from sdv.single_table import (
    CopulaGANSynthesizer, CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer)
from sdv.single_table.base import BaseSingleTableSynthesizer
from tests.integration.single_table.custom_constraints import MyConstraint

METADATA = SingleTableMetadata.load_from_dict({
    'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1',
    'columns': {
        'column1': {
            'sdtype': 'numerical'
        },
        'column2': {
            'sdtype': 'numerical'
        },
        'column3': {
            'sdtype': 'numerical'
        }
    }
})

SYNTHESIZERS = [
    pytest.param(CTGANSynthesizer(METADATA, epochs=1, cuda=False), id='CTGANSynthesizer'),
    pytest.param(TVAESynthesizer(METADATA, epochs=1, cuda=False), id='TVAESynthesizer'),
    pytest.param(GaussianCopulaSynthesizer(METADATA), id='GaussianCopulaSynthesizer'),
    pytest.param(CopulaGANSynthesizer(METADATA, epochs=1, cuda=False), id='CopulaGANSynthesizer'),
]


def _isinstance_side_effect(*args, **kwargs):
    if isinstance(args[0], GaussianMultivariate):
        return True
    else:
        return isinstance(args[0], args[1])


@pytest.mark.parametrize('synthesizer', SYNTHESIZERS)
def test_conditional_sampling_graceful_reject_sampling_true_dict(synthesizer):
    data = pd.DataFrame({
        'column1': list(range(100)),
        'column2': list(range(100)),
        'column3': list(range(100))
    })

    synthesizer.fit(data)
    conditions = [
        Condition({
            'column1': 28,
            'column2': 37,
            'column3': 93
        })
    ]

    with pytest.raises(ValueError):  # noqa: PT011
        synthesizer.sample_from_conditions(conditions=conditions)


@pytest.mark.parametrize('synthesizer', SYNTHESIZERS)
def test_conditional_sampling_graceful_reject_sampling_true_dataframe(synthesizer):
    data = pd.DataFrame({
        'column1': list(range(100)),
        'column2': list(range(100)),
        'column3': list(range(100))
    })

    synthesizer.fit(data)
    conditions = pd.DataFrame({
        'column1': [28],
        'column2': [37],
        'column3': [93]
    })

    with pytest.raises(ValueError, match='a'):
        synthesizer.sample_remaining_columns(conditions)


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
        ]
    })

    metadata = SingleTableMetadata()
    metadata.add_column('key', sdtype='id')
    metadata.add_column('index', sdtype='categorical')
    metadata.set_primary_key('key')

    model = GaussianCopulaSynthesizer(metadata)
    constraint = {
        'constraint_class': 'Unique',
        'constraint_parameters': {
            'column_names': ['index']
        }
    }
    model.add_constraints([constraint])

    # Run
    model.fit(test_df)
    samples = model.sample(2)

    # Assert
    assert len(samples) == 2
    assert samples['index'].is_unique


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
        ]
    })

    metadata = SingleTableMetadata()
    metadata.add_column('key', sdtype='id')
    metadata.add_column('index', sdtype='categorical')
    metadata.add_column('test_column', sdtype='categorical')
    metadata.set_primary_key('key')

    model = GaussianCopulaSynthesizer(metadata)
    constraint = {
        'constraint_class': 'Unique',
        'constraint_parameters': {
            'column_names': ['test_column']
        }
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
        ]
    })

    metadata = SingleTableMetadata()
    metadata.add_column('key', sdtype='id')
    metadata.add_column('test_column', sdtype='categorical')
    metadata.set_primary_key('key')

    test_df = test_df.iloc[[1, 3, 4]]
    constraint = {
        'constraint_class': 'Unique',
        'constraint_parameters': {
            'column_names': ['test_column']
        }
    }
    model = GaussianCopulaSynthesizer(metadata)
    model.add_constraints([constraint])

    # Run
    model.fit(test_df)
    samples = model.sample(2)

    # Assert
    assert len(samples) == 2
    assert samples['test_column'].is_unique


@patch('sdv.single_table.base.isinstance')
@patch('sdv.single_table.copulas.multivariate.GaussianMultivariate',
       spec_set=GaussianMultivariate)
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
        'age': [27, 28, 26, 21, 30]
    })

    metadata = SingleTableMetadata()
    metadata.add_column('city', sdtype='categorical')
    metadata.add_column('state', sdtype='categorical')
    metadata.add_column('age', sdtype='numerical')

    model = GaussianCopulaSynthesizer(metadata)

    constraint = {
        'constraint_class': 'FixedCombinations',
        'constraint_parameters': {
            'column_names': ['city', 'state']
        }
    }
    model.add_constraints([constraint])
    sampled_numeric_data = [
        pd.DataFrame({
            'city#state': [0, 1, 2, 0, 0],
            'age': [30, 30, 30, 30, 30]
        }),
        pd.DataFrame({
            'city#state': [1],
            'age': [30]
        })
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
        'age': [30, 30, 30, 30, 30]
    })
    sample_calls = model._model.sample.mock_calls
    assert len(sample_calls) == 2
    model._model.sample.assert_any_call(5, conditions=expected_transformed_conditions)
    pd.testing.assert_frame_equal(sampled_data, expected_data)


def test_sample_from_conditions_with_batch_size():
    """Test the ``sample_from_conditions`` method with a different ``batch_size``.

    If a smaller ``batch_size`` is passed, then the conditions should be broken down into
    batches of that size. If the ``batch_size`` is larger than the condition length, then
    the condition length should be used.

    - Input:
        - Conditions one of length 100 and another of length 10
        - Batch size of length 50

    - Output:
        - Sampled data
    """
    # Setup
    data = pd.DataFrame({
        'column1': list(range(100)),
        'column2': list(range(100)),
        'column3': list(range(100))
    })

    metadata = SingleTableMetadata()
    metadata.add_column('column1', sdtype='numerical')
    metadata.add_column('column2', sdtype='numerical')
    metadata.add_column('column3', sdtype='numerical')

    model = GaussianCopulaSynthesizer(metadata)
    model.fit(data)
    conditions = [
        Condition({'column1': 10}, num_rows=100),
        Condition({'column1': 50}, num_rows=10)
    ]

    # Run
    sampled_data = model.sample_from_conditions(conditions, batch_size=50)

    # Assert
    expected = pd.Series([10] * 100 + [50] * 10, name='column1')
    pd.testing.assert_series_equal(sampled_data['column1'], expected)


def test_multiple_fits():
    """Test the synthesizer refits correctly on new data.

    The synthesizer should refit the formatters and constraints.
    """
    # Setup
    data_1 = pd.DataFrame({
        'city': ['LA', 'SF', 'CHI', 'LA', 'LA'],
        'state': ['CA', 'CA', 'IL', 'CA', 'CA'],
        'measurement': [27.123, 28.756, 26.908, 21.002, 30.987]
    })
    data_2 = pd.DataFrame({
        'city': ['LA', 'LA', 'CHI', 'LA', 'LA'],
        'state': ['CA', 'CA', 'IL', 'CA', 'CA'],
        'measurement': [27.1, 28.7, 26.9, 21.2, 30.9]
    })
    metadata = SingleTableMetadata()
    metadata.add_column('city', sdtype='categorical')
    metadata.add_column('state', sdtype='categorical')
    metadata.add_column('measurement', sdtype='numerical')
    constraint = {
        'constraint_class': 'FixedCombinations',
        'constraint_parameters': {
            'column_names': ['city', 'state']
        }
    }
    model = GaussianCopulaSynthesizer(metadata)
    model.add_constraints([constraint])

    # Run
    model.fit(data_1)
    model.fit(data_2)

    # Assert
    assert ('SF', 'CA') not in model._data_processor._constraints[0]._combinations_to_uuids
    assert model._data_processor.formatters['measurement']._rounding_digits == 1


@pytest.mark.parametrize('synthesizer', SYNTHESIZERS)
def test_sampling(synthesizer):
    """Test that samples are different when ``reset_sampling`` is not called."""
    sample_1 = synthesizer.sample(10)
    sample_2 = synthesizer.sample(10)

    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(sample_1, sample_2)


@pytest.mark.parametrize('synthesizer', SYNTHESIZERS)
def test_sampling_reset_sampling(synthesizer):
    """Test ``sample`` method for each synthesizer using ``reset_sampling``."""
    metadata = SingleTableMetadata.load_from_dict({
        'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1',
        'columns': {
            'column1': {
                'sdtype': 'numerical'
            },
            'column2': {
                'sdtype': 'address'
            },
            'column3': {
                'sdtype': 'email'
            },
            'column4': {
                'sdtype': 'ssn',
                'pii': True
            }
        }
    })
    data = pd.DataFrame({
        'column1': list(range(100)),
        'column2': [str(i) for i in (range(100))],
        'column3': [str(i) for i in (range(100))],
        'column4': [str(i) for i in (range(100))],
    })

    if isinstance(synthesizer, (CTGANSynthesizer, TVAESynthesizer)):
        synthesizer = synthesizer.__class__(metadata, cuda=False)
    else:
        synthesizer = synthesizer.__class__(metadata)

    synthesizer.fit(data)

    sampled1 = synthesizer.sample(10)
    synthesizer.reset_sampling()
    sampled2 = synthesizer.sample(10)

    pd.testing.assert_frame_equal(sampled1, sampled2)


def test_config_creation_doesnt_raise_error():
    """Test https://github.com/sdv-dev/SDV/issues/1110."""
    # Setup
    test_data = pd.DataFrame({
        'address_col': ['223 Williams Rd', '75 Waltham St', '77 Mass Ave'],
        'numerical_col': [1, 2, 3],
    })
    test_metadata = SingleTableMetadata()

    # Run
    test_metadata.detect_from_dataframe(test_data)
    test_metadata.update_column(
        column_name='address_col',
        sdtype='address',
        pii=False
    )

    synthesizer = GaussianCopulaSynthesizer(test_metadata)
    synthesizer.fit(test_data)


def test_transformers_correctly_auto_assigned():
    """Ensure the correct transformers and parameters are auto assigned to the data."""
    # Setup
    data = pd.DataFrame({
        'primary_key': ['user-000', 'user-001', 'user-002'],
        'pii_col': ['223 Williams Rd', '75 Waltham St', '77 Mass Ave'],
        'numerical_col': [1, 2, 3],
        'categorical_col': ['a', 'b', 'a'],
    })

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    metadata.update_column(column_name='primary_key', sdtype='id', regex_format='user-[0-9]{3}')
    metadata.set_primary_key('primary_key')
    metadata.update_column(column_name='pii_col', sdtype='address', pii=True)
    synthesizer = GaussianCopulaSynthesizer(
        metadata, enforce_min_max_values=False, enforce_rounding=False)

    # Run
    synthesizer.auto_assign_transformers(data)
    transformers = synthesizer.get_transformers()

    # Assert
    assert isinstance(transformers['numerical_col'], FloatFormatter)
    assert isinstance(transformers['pii_col'], AnonymizedFaker)
    assert isinstance(transformers['primary_key'], RegexGenerator)
    assert isinstance(transformers['categorical_col'], LabelEncoder)

    assert transformers['numerical_col'].missing_value_replacement == 'mean'
    assert transformers['numerical_col'].missing_value_generation == 'random'
    assert transformers['numerical_col'].learn_rounding_scheme is False
    assert transformers['numerical_col'].enforce_min_max_values is False

    assert transformers['pii_col'].provider_name == 'address'
    assert transformers['pii_col'].function_name == 'address'

    assert transformers['primary_key'].regex_format == 'user-[0-9]{3}'
    assert transformers['primary_key'].enforce_uniqueness is True

    assert transformers['categorical_col'].add_noise is True


def test_modeling_with_complex_datetimes():
    """Test that models work with datetimes passed as strings or ints with complex format."""
    # Setup
    data = pd.DataFrame(data={
        'string_column': [
            '20220902110443000000',
            '20220916230356000000',
            '20220826173917000000',
            '20220826212135000000',
            '20220929111311000000'
        ],
        'int_column': [
            20220902110443000000,
            20220916230356000000,
            20220826173917000000,
            20220826212135000000,
            20220929111311000000
        ]
    })

    test_metadata = {
        'columns': {
            'string_column': {
                'sdtype': 'datetime',
                'datetime_format': '%Y%m%d%H%M%S%f'
            },
            'int_column': {
                'sdtype': 'datetime',
                'datetime_format': '%Y%m%d%H%M%S%f'
            }
        }
    }

    # Run
    metadata = SingleTableMetadata.load_from_dict(test_metadata)
    metadata.validate()
    synth = GaussianCopulaSynthesizer(metadata)
    synth.validate(data)
    synth.fit(data)
    sampled = synth.sample(10)

    # Assert
    synth.validate(sampled)


def test_custom_constraints_from_file(tmpdir):
    """Ensure the correct loading for a custom constraint class defined in another file."""
    data = pd.DataFrame({
        'primary_key': ['user-000', 'user-001', 'user-002'],
        'pii_col': ['223 Williams Rd', '75 Waltham St', '77 Mass Ave'],
        'numerical_col': [2, 3, 4],
        'categorical_col': ['a', 'b', 'a'],
    })

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    metadata.update_column(column_name='pii_col', sdtype='address', pii=True)
    synthesizer = GaussianCopulaSynthesizer(
        metadata,
        enforce_min_max_values=False,
        enforce_rounding=False
    )
    synthesizer.load_custom_constraint_classes(
        'tests/integration/single_table/custom_constraints.py',
        ['MyConstraint']
    )
    constraint = {
        'constraint_class': 'MyConstraint',
        'constraint_parameters': {
            'column_names': ['numerical_col']
        }
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


def test_custom_constraints_from_object(tmpdir):
    """Ensure the correct loading for a custom constraint class passed as an object."""
    data = pd.DataFrame({
        'primary_key': ['user-000', 'user-001', 'user-002'],
        'pii_col': ['223 Williams Rd', '75 Waltham St', '77 Mass Ave'],
        'numerical_col': [2, 3, 4],
        'categorical_col': ['a', 'b', 'a'],
    })

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    metadata.update_column(column_name='pii_col', sdtype='address', pii=True)
    synthesizer = GaussianCopulaSynthesizer(
        metadata,
        enforce_min_max_values=False,
        enforce_rounding=False
    )
    synthesizer.add_custom_constraint_class(MyConstraint, 'MyConstraint')
    constraint = {
        'constraint_class': 'MyConstraint',
        'constraint_parameters': {
            'column_names': ['numerical_col']
        }
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


def test_auto_assign_transformers_and_update_with_pii():
    """Ensure the ability to update a transformer with any given ``pii`` sdtype.

    This test is designed to auto-assign the transformers to an updated metadata that contains
    an ``pii`` field set as ``sdtype`` but no ``pii`` in the ``metadata`` itself. This should
    still assign the expected transformer to it.
    """
    # Setup
    data = pd.DataFrame(data={
        'id': ['N', 'A', 'K', 'F', 'P'],
        'numerical': [1, 2, 3, 2, 1],
        'name': ['A', 'A', 'B', 'B', 'B']
    })

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)

    # Run
    metadata.update_column(column_name='id', sdtype='first_name')
    metadata.update_column(column_name='name', sdtype='name')
    metadata.set_primary_key('id')
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.auto_assign_transformers(data)

    # Assert
    id_transformer = synthesizer.get_transformers()['id']
    name_transformer = synthesizer.get_transformers()['name']
    assert id_transformer.provider_name == 'person'
    assert id_transformer.function_name == 'first_name'
    assert id_transformer.enforce_uniqueness is True

    assert name_transformer.provider_name == 'person'
    assert name_transformer.function_name == 'name'
    assert name_transformer.enforce_uniqueness is False


def test_refitting_a_model():
    """Test that refitting a model resets the sampling state of the generators."""
    # Setup
    data = pd.DataFrame(data={
        'id': [0, 1, 2, 3, 4],
        'numerical': [1, 2, 3, 2, 1],
        'name': ['A', 'A', 'B', 'B', 'B']
    })

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    metadata.update_column(column_name='name', sdtype='name')
    metadata.update_column('id', sdtype='id')
    metadata.set_primary_key('id')

    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(data)
    first_sample = synthesizer.sample(10)

    # Run
    synthesizer.fit(data)
    second_sample = synthesizer.sample(10)

    # Assert
    assert all(second_sample['name'] == first_sample['name'])
    assert all(second_sample['id'] == first_sample['id'])


def test_get_info():
    """Test the correct dictionary is returned.

    Check the return dictionary is valid both before and after fitting a synthesizer.
    """
    # Setup
    data = pd.DataFrame({'col': [1, 2, 3]})
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    metadata = SingleTableMetadata()
    metadata.add_column('col', sdtype='numerical')
    synthesizer = GaussianCopulaSynthesizer(metadata)

    # Run
    info = synthesizer.get_info()

    # Assert
    assert info == {
        'class_name': 'GaussianCopulaSynthesizer',
        'creation_date': today,
        'is_fit': False,
        'last_fit_date': None,
        'fitted_sdv_version': None
    }

    # Run
    synthesizer.fit(data)
    info = synthesizer.get_info()

    # Assert
    version = pkg_resources.get_distribution('sdv').version
    assert info == {
        'class_name': 'GaussianCopulaSynthesizer',
        'creation_date': today,
        'is_fit': True,
        'last_fit_date': today,
        'fitted_sdv_version': version
    }


def test_synthesizer_with_inequality_constraint():
    """Ensure that the ``Inequality`` constraint can sample from the model."""
    # Setup
    real_data, metadata = download_demo(
        modality='single_table',
        dataset_name='fake_hotel_guests'
    )
    synthesizer = GaussianCopulaSynthesizer(metadata)
    checkin_lessthan_checkout = {
        'constraint_class': 'Inequality',
        'constraint_parameters': {
            'low_column_name': 'checkin_date',
            'high_column_name': 'checkout_date'
        }
    }

    synthesizer.add_constraints([checkin_lessthan_checkout])
    synthesizer.fit(real_data)

    # Run and Assert
    sampled = synthesizer.sample(num_rows=500)
    synthesizer.validate(sampled)
    _sampled = sampled[~sampled['checkout_date'].isna()]
    assert all(
        pd.to_datetime(_sampled['checkin_date']) < pd.to_datetime(_sampled['checkout_date'])
    )


def test_inequality_constraint_with_datetimes_and_nones():
    """Test that the ``Inequality`` constraint works with ``None`` and ``datetime``."""
    # Setup
    data = pd.DataFrame(data={
        'A': [None, None, '2020-01-02', '2020-03-04'] * 2,
        'B': [None, '2021-03-04', '2021-12-31', None] * 2
    })

    metadata = SingleTableMetadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
            'B': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'}
        }
    })

    metadata.validate()
    synth = GaussianCopulaSynthesizer(metadata)
    synth.add_constraints([
        {
            'constraint_class': 'Inequality',
            'constraint_parameters': {
                'low_column_name': 'A',
                'high_column_name': 'B'
            }
        }
    ])
    synth.validate(data)

    # Run
    synth.fit(data)
    sampled = synth.sample(10)

    # Assert
    expected_sampled = pd.DataFrame({
        'A': {
            0: '2020-01-02',
            1: '2019-10-30',
            2: np.nan,
            3: np.nan,
            4: '2020-01-02',
            5: np.nan,
            6: '2019-10-30',
            7: np.nan,
            8: '2020-01-02',
            9: np.nan
        },
        'B': {
            0: '2021-12-30',
            1: '2021-10-27',
            2: '2021-10-27',
            3: '2021-10-27',
            4: np.nan,
            5: '2021-10-27',
            6: '2021-10-27',
            7: '2021-12-30',
            8: np.nan,
            9: '2021-10-27'
        }
    })
    pd.testing.assert_frame_equal(expected_sampled, sampled)


def test_scalar_inequality_constraint_with_datetimes_and_nones():
    """Test that the ``ScalarInequality`` constraint works with ``None`` and ``datetime``."""
    # Setup
    data = pd.DataFrame(data={
        'A': [None, None, '2020-01-02', '2020-03-04'],
        'B': [None, '2021-03-04', '2021-12-31', None]
    })

    metadata = SingleTableMetadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
            'B': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'}
        }
    })

    metadata.validate()
    synth = GaussianCopulaSynthesizer(metadata)
    synth.add_constraints([
        {
            'constraint_class': 'ScalarInequality',
            'constraint_parameters': {
                'column_name': 'A',
                'relation': '>=',
                'value': '2019-01-01'
            }
        }
    ])
    synth.validate(data)

    # Run
    synth.fit(data)
    sampled = synth.sample(5)

    # Assert
    expected_sampled = pd.DataFrame({
        'A': {
            0: np.nan,
            1: '2020-01-19',
            2: np.nan,
            3: '2020-01-29',
            4: '2020-01-31',
        },
        'B': {
            0: '2021-07-28',
            1: '2021-07-14',
            2: '2021-07-26',
            3: '2021-07-02',
            4: '2021-06-06',
        }
    })
    pd.testing.assert_frame_equal(expected_sampled, sampled)


def test_scalar_range_constraint_with_datetimes_and_nones():
    """Test that the ``ScalarRange`` constraint works with ``None`` and ``datetime``."""
    # Setup
    data = pd.DataFrame(data={
        'A': [None, None, '2020-01-02', '2020-03-04'],
        'B': [None, '2021-03-04', '2021-12-31', None]
    })

    metadata = SingleTableMetadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
            'B': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'}
        }
    })

    metadata.validate()
    synth = GaussianCopulaSynthesizer(metadata)
    synth.add_constraints([
        {
            'constraint_class': 'ScalarRange',
            'constraint_parameters': {
                'column_name': 'A',
                'low_value': '2019-10-30',
                'high_value': '2020-03-04',
                'strict_boundaries': False
            }
        }
    ])
    synth.validate(data)

    # Run
    synth.fit(data)
    sampled = synth.sample(10)

    # Assert
    expected_sampled = pd.DataFrame({
        'A': {
            0: '2020-03-03',
            1: np.nan,
            2: '2020-03-03',
            3: np.nan,
            4: np.nan,
            5: '2020-03-03',
            6: np.nan,
            7: np.nan,
            8: np.nan,
            9: '2020-02-27',
        },
        'B': {
            0: np.nan,
            1: np.nan,
            2: np.nan,
            3: np.nan,
            4: np.nan,
            5: '2021-04-14',
            6: np.nan,
            7: '2021-05-21',
            8: np.nan,
            9: np.nan,
        }
    })
    pd.testing.assert_frame_equal(expected_sampled, sampled)


def test_range_constraint_with_datetimes_and_nones():
    """Test that the ``Range`` constraint works with ``None`` and ``datetime``."""
    # Setup
    data = pd.DataFrame(data={
        'A': [None, None, '2020-01-02', '2020-03-04'],
        'B': [None, '2021-03-04', '2021-12-31', None],
        'C': [None, '2022-03-04', '2022-12-31', None],
    })

    metadata = SingleTableMetadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
            'B': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
            'C': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'}
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
                'strict_boundaries': False
            }
        }
    ])
    synth.validate(data)

    # Run
    synth.fit(data)
    sampled = synth.sample(10)

    # Assert
    expected_sampled = pd.DataFrame({
        'A': {
            0: '2020-01-02',
            1: '2020-01-02',
            2: np.nan,
            3: '2020-01-02',
            4: '2019-10-30',
            5: np.nan,
            6: '2020-01-02',
            7: '2019-10-30',
            8: '2019-10-30',
            9: np.nan
        },
        'B': {
            0: '2021-12-30',
            1: '2021-12-30',
            2: '2021-10-27',
            3: np.nan,
            4: '2021-10-27',
            5: '2021-10-27',
            6: np.nan,
            7: '2021-10-27',
            8: np.nan,
            9: '2021-10-27'
        },
        'C': {
            0: '2022-12-30',
            1: '2022-12-30',
            2: '2022-10-27',
            3: np.nan,
            4: '2022-10-27',
            5: '2022-10-27',
            6: np.nan,
            7: '2022-10-27',
            8: np.nan,
            9: '2022-10-27'
        }
    })
    pd.testing.assert_frame_equal(expected_sampled, sampled)


def test_inequality_constraint_all_possible_nans_configurations():
    """Test that the inequality constraint works with all possible NaN configurations."""
    # Setup
    data = pd.DataFrame(data={
        'A': [0, 1, np.nan, np.nan, 2],
        'B': [2, np.nan, 3, np.nan, 3]
    })

    metadata = SingleTableMetadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'numerical'},
            'B': {'sdtype': 'numerical'},
        }
    })

    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.add_constraints(
        [
            {
                'constraint_class': 'Inequality',
                'constraint_parameters': {
                    'low_column_name': 'A',
                    'high_column_name': 'B'
                }
            }
        ]
    )

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
    data = pd.DataFrame(data={
        'low': [1, 4, np.nan, 0, 4, np.nan, np.nan, 5, np.nan],
        'middle': [2, 5, 3, np.nan, 5, np.nan, 5, np.nan, np.nan],
        'high': [3, 7, 8, 4, np.nan, 9, np.nan, np.nan, np.nan]
    })

    metadata_dict = {
        'columns': {
            'low': {'sdtype': 'numerical'},
            'middle': {'sdtype': 'numerical'},
            'high': {'sdtype': 'numerical'}
        }
    }

    metadata = SingleTableMetadata.load_from_dict(metadata_dict)
    synthesizer = GaussianCopulaSynthesizer(metadata)

    my_constraint = {
        'constraint_class': 'Range',
        'constraint_parameters': {
            'low_column_name': 'low',
            'middle_column_name': 'middle',
            'high_column_name': 'high'
        }
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


def test_save_and_load(tmp_path):
    """Test that synthesizers can be saved and loaded properly."""
    # Setup
    metadata = SingleTableMetadata()
    instance = BaseSingleTableSynthesizer(metadata)
    synthesizer_path = tmp_path / 'synthesizer.pkl'
    instance.save(synthesizer_path)

    # Run
    loaded_instance = BaseSingleTableSynthesizer.load(synthesizer_path)

    # Assert
    assert isinstance(loaded_instance, BaseSingleTableSynthesizer)
    assert instance.metadata.columns == {}
    assert instance.metadata.primary_key is None
    assert instance.metadata.alternate_keys == []
    assert instance.metadata.sequence_key is None
    assert instance.metadata.sequence_index is None
    assert instance.metadata._version == 'SINGLE_TABLE_V1'
