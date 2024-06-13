import datetime
import re

import numpy as np
import pandas as pd
import pytest
from deepecho import load_demo

from sdv.datasets.demo import download_demo
from sdv.errors import SynthesizerInputError
from sdv.metadata import SingleTableMetadata
from sdv.sequential import PARSynthesizer


def _get_par_data_and_metadata():
    date = datetime.datetime.strptime('2020-01-01', '%Y-%m-%d')
    data = pd.DataFrame({
        'column1': [1.0, 2.0, 1.5, 1.3],
        'date': [date, date, date, date],
        'column2': ['b', 'a', 'a', 'c'],
        'entity': [1, 1, 2, 2],
        'context': ['a', 'a', 'b', 'b']
    })
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    metadata.update_column('entity', sdtype='id')
    metadata.set_sequence_key('entity')
    metadata.set_sequence_index('date')
    return data, metadata


def test_par():
    """Test the ``PARSynthesizer`` end to end."""
    # Setup
    data = load_demo()
    data['date'] = pd.to_datetime(data['date'])
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    metadata.update_column('store_id', sdtype='id')
    metadata.set_sequence_key('store_id')
    metadata.set_sequence_index('date')
    model = PARSynthesizer(
        metadata=metadata,
        context_columns=['region'],
        epochs=1,
    )

    # Run
    model.fit(data)
    sampled = model.sample(100)

    # Assert
    assert sampled.shape == data.shape
    assert (sampled.dtypes == data.dtypes).all()
    assert (sampled.notna().sum(axis=1) != 0).all()
    loss_values = model.get_loss_values()
    assert len(loss_values) == 1
    assert all(sampled.groupby('store_id')['date'].is_monotonic_increasing)
    assert all(sampled.groupby('store_id')['date'].agg(lambda x: x.is_unique))


def test_column_after_date_simple():
    """Test that adding a column after the ``sequence_index`` column works."""
    # Setup
    date = datetime.datetime.strptime('2020-01-01', '%Y-%m-%d')
    data = pd.DataFrame({
        'col': [1, 1],
        'date': [date, date],
        'col2': ['hello', 'world'],
    })
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    metadata.update_column('col', sdtype='id')
    metadata.set_sequence_key('col')
    metadata.set_sequence_index('date')

    # Run
    model = PARSynthesizer(metadata=metadata, epochs=1)
    model.fit(data)
    sampled = model.sample(1)

    # Assert
    assert sampled.shape == data.shape
    assert (sampled.dtypes == data.dtypes).all()
    assert (sampled.notna().sum(axis=1) != 0).all()


def test_column_after_date_complex():
    """Test that adding multiple columns after the ``sequence_index`` column works."""
    # Setup
    data, metadata = _get_par_data_and_metadata()

    # Run
    model = PARSynthesizer(metadata=metadata, context_columns=['context'], epochs=1)
    model.fit(data)
    sampled = model.sample(2)

    # Assert
    assert sampled.shape == data.shape
    assert (sampled.dtypes == data.dtypes).all()
    assert (sampled.notna().sum(axis=1) != 0).all()


def test_save_and_load(tmp_path):
    """Test that synthesizers can be saved and loaded properly."""
    # Setup
    _, metadata = _get_par_data_and_metadata()
    instance = PARSynthesizer(metadata)
    synthesizer_path = tmp_path / 'synthesizer.pkl'
    instance.save(synthesizer_path)

    # Run
    loaded_instance = PARSynthesizer.load(synthesizer_path)

    # Assert
    assert isinstance(loaded_instance, PARSynthesizer)
    assert metadata == instance.metadata


def test_sythesize_sequences(tmp_path):
    """End to end test for synthesizing sequences.

    The following functionalities are being tested:
        * Fit a ``PARSynthesizer`` with the demo dataset.
        * Fit a ``PARSynthesizer`` with custom context.
        * Sample from the model.
        * Conditionally sample from the model.
        * Save and Load.
    """
    # Setup
    real_data, metadata = download_demo(
        modality='sequential',
        dataset_name='nasdaq100_2019'
    )
    assert real_data[real_data['Symbol'] == 'AMZN']['Sector'].unique()
    synthesizer = PARSynthesizer(
        metadata,
        epochs=5,
        context_columns=['Sector', 'Industry']
    )
    custom_synthesizer = PARSynthesizer(
        metadata,
        epochs=5,
        context_columns=['Sector', 'Industry'],
        verbose=True
    )
    scenario_context = pd.DataFrame(data={
        'Symbol': ['COMPANY-A', 'COMPANY-B', 'COMPANY-C', 'COMPANY-D', 'COMPANY-E'],
        'Sector': ['Technology'] * 2 + ['Consumer Services'] * 3,
        'Industry': [
            'Computer Manufacturing', 'Computer Software: Prepackaged Software',
            'Hotels/Resorts', 'Restaurants', 'Clothing/Shoe/Accessory Stores'
        ]
    })

    # Run - Fit
    synthesizer.fit(real_data)
    custom_synthesizer.fit(real_data)

    # Run - Sample
    synthetic_data = synthesizer.sample(num_sequences=10)
    custom_synthetic_data = custom_synthesizer.sample(num_sequences=3, sequence_length=2)
    custom_synthetic_data_conditional = custom_synthesizer.sample_sequential_columns(
        context_columns=scenario_context,
        sequence_length=2
    )

    # Save and Load
    model_path = tmp_path / 'my_synthesizer.pkl'
    synthesizer.save(model_path)
    loaded_synthesizer = PARSynthesizer.load(model_path)
    loaded_sample = loaded_synthesizer.sample(100)

    # Assert
    assert all(custom_synthetic_data_conditional['Symbol'].value_counts() == 2)
    companies = ['COMPANY-A', 'COMPANY-B', 'COMPANY-C', 'COMPANY-D', 'COMPANY-E']
    assert companies in custom_synthetic_data_conditional['Symbol'].unique()
    assert custom_synthetic_data_conditional['Sector'].value_counts()['Technology'] == 4
    assert custom_synthetic_data_conditional['Sector'].value_counts()['Consumer Services'] == 6
    industries = [
        'Computer Manufacturing',
        'Computer Software: Prepackaged Software',
        'Hotels/Resorts',
        'Restaurants',
        'Clothing/Shoe/Accessory Stores'
    ]
    assert industries in custom_synthetic_data_conditional['Industry'].unique()

    assert model_path.exists()
    assert model_path.is_file()
    assert loaded_synthesizer.get_info() == synthesizer.get_info()
    assert loaded_synthesizer.metadata.to_dict() == metadata.to_dict()
    synthesizer.validate(synthetic_data)
    synthesizer.validate(custom_synthetic_data)
    synthesizer.validate(custom_synthetic_data_conditional)
    synthesizer.validate(loaded_sample)
    loaded_synthesizer.validate(synthetic_data)
    loaded_synthesizer.validate(loaded_sample)


def test_par_subset_of_data():
    """Test it when the data index is not continuous GH#1973."""
    # download data
    data, metadata = download_demo(modality='sequential', dataset_name='nasdaq100_2019',)

    # modify the data by choosing a subset of it
    data_subset = data.copy()
    np.random.seed(1234)
    symbols = data['Symbol'].unique()

    # only select a subset of data in each sequence
    for i, symbol in enumerate(symbols):
        symbol_mask = data_subset['Symbol'] == symbol
        data_subset = data_subset.drop(
            data_subset[symbol_mask].sample(frac=i / (2 * len(symbols))).index)

    # now run PAR
    synthesizer = PARSynthesizer(metadata, epochs=5, verbose=True)
    synthesizer.fit(data_subset)
    synthetic_data = synthesizer.sample(num_sequences=5)

    # assert that the synthetic data doesn't contain NaN values in sequence index column
    assert not pd.isna(synthetic_data['Date']).any()


def test_par_subset_of_data_simplified():
    """Test it when the data index is not continuous for a simple dataset GH#1973."""
    # Setup
    data = pd.DataFrame({
        'id': [1, 2, 3],
        'date': ['2020-01-01', '2020-01-02', '2020-01-03'],
    })
    data.index = [0, 1, 5]
    metadata = SingleTableMetadata.load_from_dict({
        'sequence_index': 'date',
        'sequence_key': 'id',
        'columns': {
            'id': {
                'sdtype': 'id',
            },
            'date': {
                'sdtype': 'datetime',
            },
        },
        'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1'
    })
    synthesizer = PARSynthesizer(metadata, epochs=0)

    # Run
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample(num_sequences=50)

    # Assert
    assert not pd.isna(synthetic_data['date']).any()


def test_par_missing_sequence_index():
    """Test if PAR Synthesizer can run without a sequence key"""
    # Setup
    metadata_dict = {
        'columns': {
            'value': {
                'sdtype': 'numerical'
            },
            'e_id': {
                'sdtype': 'id'
            }
        },
        'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1',
        'sequence_key': 'e_id'
    }

    metadata = SingleTableMetadata().load_from_dict(metadata_dict)

    data = pd.DataFrame({
        'value': [10, 20, 30],
        'e_id': [1, 2, 3]
    })

    # Run
    synthesizer = PARSynthesizer(metadata)
    synthesizer.fit(data)
    sampled = synthesizer.sample(num_sequences=3)

    # Assert
    assert sampled.shape == data.shape
    assert (sampled.dtypes == data.dtypes).all()


def test_constraints_on_par():
    """Test if only simple constraints work on PARSynthesizer."""
    # Setup
    real_data, metadata = download_demo(
        modality='sequential',
        dataset_name='nasdaq100_2019'
    )

    synthesizer = PARSynthesizer(
        metadata,
        epochs=5,
        context_columns=['Sector', 'Industry']
    )

    market_constraint = {
        'constraint_class': 'Positive',
        'constraint_parameters': {
            'column_name': 'MarketCap',
            'strict_boundaries': True
        }
    }
    volume_constraint = {
        'constraint_class': 'Positive',
        'constraint_parameters': {
            'column_name': 'Volume',
            'strict_boundaries': True
        }
    }

    context_constraint = {
        'constraint_class': 'Mock',
        'constraint_parameters': {
            'column_name': 'Sector',
            'strict_boundaries': True
        }
    }

    # Run
    synthesizer.add_constraints([volume_constraint, market_constraint])
    synthesizer.fit(real_data)
    samples = synthesizer.sample(50, 10)

    # Assert
    assert not (samples['MarketCap'] < 0).any().any()
    assert not (samples['Volume'] < 0).any().any()
    mixed_constraint_error_msg = re.escape(
        'The PARSynthesizer cannot accommodate constraints '
        'with a mix of context and non-context columns.'
    )

    with pytest.raises(SynthesizerInputError, match=mixed_constraint_error_msg):
        synthesizer.add_constraints([volume_constraint, context_constraint])


def test_par_unique_sequence_index_with_enforce_min_max():
    """Test to see if there are duplicate sequence index values
    when sequence_length is higher than real data
    """
    # Setup
    test_id = list(range(10))
    s_key = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    visits = [
        '2021-01-01', '2021-01-03', '2021-01-05', '2021-01-07', '2021-01-09',
        '2021-09-11', '2021-09-17', '2021-10-01', '2021-10-08', '2021-11-01'
    ]
    pre_date = [
        '2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05',
        '2021-04-01', '2021-04-02', '2021-04-03', '2021-04-04', '2021-04-05'
    ]
    test_df = pd.DataFrame({
        'id': test_id,
        's_key': s_key,
        'visits': visits,
        'pre_date': pre_date
    })
    test_df[['visits', 'pre_date']] = test_df[['visits', 'pre_date']].apply(
        pd.to_datetime, format='%Y-%m-%d', errors='coerce')
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(test_df)
    metadata.update_column(column_name='s_key', sdtype='id')
    metadata.set_sequence_key('s_key')
    metadata.set_sequence_index('visits')
    synthesizer = PARSynthesizer(metadata, enforce_min_max_values=True,
                                 enforce_rounding=False, epochs=100, verbose=True)

    # Run
    synthesizer.fit(test_df)
    synth_df = synthesizer.sample(num_sequences=50, sequence_length=50)

    # Assert
    for i in synth_df['s_key'].unique():
        seq_df = synth_df[synth_df['s_key'] == i]
        has_duplicates = seq_df['visits'].duplicated().any()
        assert not has_duplicates
