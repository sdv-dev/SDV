import datetime

import pandas as pd
import pkg_resources
import pytest
from rdt.transformers import AnonymizedFaker, FloatFormatter, RegexGenerator, UniformEncoder

from sdv.metadata import SingleTableMetadata
from sdv.sampling import Condition
from sdv.single_table import (
    CopulaGANSynthesizer, CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer)
from sdv.single_table.base import BaseSingleTableSynthesizer

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


def test_sample_from_conditions_negative_float():
    """Test it when the condition is a negative float (GH#1161)."""
    # Setup
    data = pd.DataFrame({
        'column1': [-float(i) for i in range(100)],
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
        Condition({'column1': -10.}, num_rows=100),
        Condition({'column1': -50}, num_rows=10)
    ]

    # Run
    sampled_data = model.sample_from_conditions(conditions)

    # Assert
    expected = pd.Series([-10.] * 100 + [-50.] * 10, name='column1')
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
    assert isinstance(transformers['categorical_col'], UniformEncoder)

    assert transformers['numerical_col'].missing_value_replacement == 'mean'
    assert transformers['numerical_col'].missing_value_generation == 'random'
    assert transformers['numerical_col'].learn_rounding_scheme is False
    assert transformers['numerical_col'].enforce_min_max_values is False

    assert transformers['pii_col'].provider_name == 'address'
    assert transformers['pii_col'].function_name == 'address'

    assert transformers['primary_key'].regex_format == 'user-[0-9]{3}'
    assert transformers['primary_key'].enforce_uniqueness is True


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
