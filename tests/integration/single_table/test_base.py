from unittest.mock import patch

import pandas as pd
import pytest
from copulas.multivariate.gaussian import GaussianMultivariate
from rdt.transformers import AnonymizedFaker, FloatFormatter, LabelEncoder, RegexGenerator

from sdv.metadata import SingleTableMetadata
from sdv.sampling import Condition
from sdv.single_table.copulagan import CopulaGANSynthesizer
from sdv.single_table.copulas import GaussianCopulaSynthesizer
from sdv.single_table.ctgan import CTGANSynthesizer, TVAESynthesizer

METADATA = SingleTableMetadata._load_from_dict({
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

MODELS = [
    pytest.param(CTGANSynthesizer(METADATA, epochs=1), id='CTGANSynthesizer'),
    pytest.param(TVAESynthesizer(METADATA, epochs=1), id='TVAESynthesizer'),
    pytest.param(GaussianCopulaSynthesizer(METADATA), id='GaussianCopulaSynthesizer'),
    pytest.param(CopulaGANSynthesizer(METADATA, epochs=1), id='CopulaGANSynthesizer'),
]


def _isinstance_side_effect(*args, **kwargs):
    if isinstance(args[0], GaussianMultivariate):
        return True
    else:
        return isinstance(args[0], args[1])


@pytest.mark.parametrize('model', MODELS)
def test_conditional_sampling_graceful_reject_sampling_true_dict(model):
    data = pd.DataFrame({
        'column1': list(range(100)),
        'column2': list(range(100)),
        'column3': list(range(100))
    })

    model.fit(data)
    conditions = [
        Condition({
            'column1': 28,
            'column2': 37,
            'column3': 93
        })
    ]

    with pytest.raises(ValueError):  # noqa: PT011
        model.sample_conditions(conditions=conditions)


@pytest.mark.parametrize('model', MODELS)
def test_conditional_sampling_graceful_reject_sampling_true_dataframe(model):
    data = pd.DataFrame({
        'column1': list(range(100)),
        'column2': list(range(100)),
        'column3': list(range(100))
    })

    model.fit(data)
    conditions = pd.DataFrame({
        'column1': [28],
        'column2': [37],
        'column3': [93]
    })

    with pytest.raises(ValueError, match='a'):
        model.sample_remaining_columns(conditions)


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
    metadata.add_column('key', sdtype='numerical')
    metadata.add_column('index', sdtype='categorical')
    metadata.set_primary_key('key')
    metadata.add_constraint('Unique', column_names=['index'])

    model = GaussianCopulaSynthesizer(metadata)

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
    metadata.add_column('key', sdtype='numerical')
    metadata.add_column('index', sdtype='categorical')
    metadata.add_column('test_column', sdtype='categorical')
    metadata.set_primary_key('key')
    metadata.add_constraint('Unique', column_names=['test_column'])

    model = GaussianCopulaSynthesizer(metadata)

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
    metadata.add_column('key', sdtype='numerical')
    metadata.add_column('test_column', sdtype='categorical')
    metadata.set_primary_key('key')
    metadata.add_constraint('Unique', column_names=['test_column'])

    test_df = test_df.iloc[[1, 3, 4]]
    model = GaussianCopulaSynthesizer(metadata)

    # Run
    model.fit(test_df)
    samples = model.sample(2)

    # Assert
    assert len(samples) == 2
    assert samples['test_column'].is_unique


@patch('sdv.tabular.base.isinstance')
@patch('sdv.tabular.copulas.copulas.multivariate.GaussianMultivariate',
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
    metadata.add_constraint('FixedCombinations', column_names=['city', 'state'])

    model = GaussianCopulaSynthesizer(metadata)
    sampled_numeric_data = [
        pd.DataFrame({
            'city#state.value': [0, 1, 2, 0, 0],
            'age.value': [30, 30, 30, 30, 30]
        }),
        pd.DataFrame({
            'city#state.value': [1],
            'age.value': [30]
        })
    ]
    gm_mock.return_value.sample.side_effect = sampled_numeric_data
    model.fit(data)

    # Run
    conditions = [Condition({'age': 30, 'state': 'CA'}, num_rows=5)]
    sampled_data = model.sample_conditions(conditions=conditions)

    # Assert
    expected_transformed_conditions = {'age.value': 30}
    expected_data = pd.DataFrame({
        'city': ['LA', 'SF', 'LA', 'LA', 'SF'],
        'state': ['CA', 'CA', 'CA', 'CA', 'CA'],
        'age': [30, 30, 30, 30, 30]
    })
    sample_calls = model._model.sample.mock_calls
    assert len(sample_calls) == 2
    model._model.sample.assert_any_call(5, conditions=expected_transformed_conditions)
    pd.testing.assert_frame_equal(sampled_data, expected_data)


def test_sample_conditions_with_batch_size():
    """Test the ``sample_conditions`` method with a different ``batch_size``.

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
    sampled_data = model.sample_conditions(conditions, batch_size=50)

    # Assert
    expected = pd.Series([10] * 100 + [50] * 10, name='column1')
    pd.testing.assert_series_equal(sampled_data['column1'], expected)


@pytest.mark.parametrize('model', MODELS)
def test_sampling_with_randomize_samples_true(model):
    data = pd.DataFrame({
        'column1': list(range(100)),
        'column2': list(range(100)),
        'column3': list(range(100))
    })

    model.fit(data)

    sampled1 = model.sample(10, randomize_samples=True)
    sampled2 = model.sample(10, randomize_samples=True)

    assert not sampled1.equals(sampled2)


@pytest.mark.parametrize('model', MODELS)
def test_sampling_with_randomize_samples_false(model):
    data = pd.DataFrame({
        'column1': list(range(100)),
        'column2': list(range(100)),
        'column3': list(range(100))
    })

    model.fit(data)

    sampled1 = model.sample(10, randomize_samples=False)
    sampled2 = model.sample(10, randomize_samples=False)

    pd.testing.assert_frame_equal(sampled1, sampled2)


@pytest.mark.parametrize('model', MODELS)
def test_sampling_with_randomize_samples_alternating(model):
    data = pd.DataFrame({
        'column1': list(range(100)),
        'column2': list(range(100)),
        'column3': list(range(100))
    })

    model.fit(data)

    sampled_fixed1 = model.sample(10, randomize_samples=False)
    sampled_random1 = model.sample(10, randomize_samples=True)
    sampled_fixed2 = model.sample(10, randomize_samples=False)
    sampled_random2 = model.sample(10, randomize_samples=True)

    pd.testing.assert_frame_equal(sampled_fixed1, sampled_fixed2)
    assert not sampled_random1.equals(sampled_fixed1)
    assert not sampled_random1.equals(sampled_random2)
    assert not sampled_random2.equals(sampled_fixed1)


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
    metadata.update_column(column_name='primary_key', sdtype='text', regex_format='user-[0-9]{3}')
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
    assert transformers['numerical_col'].model_missing_values is True
    assert transformers['numerical_col'].learn_rounding_scheme is False
    assert transformers['numerical_col'].enforce_min_max_values is False

    assert transformers['pii_col'].provider_name == 'address'
    assert transformers['pii_col'].function_name == 'address'

    assert transformers['primary_key'].regex_format == 'user-[0-9]{3}'
    assert transformers['primary_key'].enforce_uniqueness is True

    assert transformers['categorical_col'].add_noise is True
