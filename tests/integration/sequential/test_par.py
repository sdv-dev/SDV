import datetime

import pandas as pd
from deepecho import load_demo

from sdv.datasets.demo import download_demo
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
    real_data[real_data['Symbol'] == 'AMZN']['Sector'].unique()
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
