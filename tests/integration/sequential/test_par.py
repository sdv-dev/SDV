import datetime
import re

import numpy as np
import pandas as pd
import pytest
from deepecho import load_demo
from rdt.transformers.categorical import UniformEncoder

from sdv.datasets.demo import download_demo
from sdv.errors import SynthesizerInputError
from sdv.metadata.metadata import Metadata
from sdv.sequential import PARSynthesizer


def _get_par_data_and_metadata():
    date = datetime.datetime.strptime('2020-01-01', '%Y-%m-%d')
    data = pd.DataFrame({
        'column1': [1.0, 2.0, 1.5, 1.3],
        'date': [date, date, date, date],
        'column2': ['b', 'a', 'a', 'c'],
        'entity': [1, 1, 2, 2],
        'context': ['a', 'a', 'b', 'b'],
        'context_date': [date, date, date, date],
    })
    metadata = Metadata.detect_from_dataframes({'table': data})
    metadata.update_column('entity', 'table', sdtype='id')
    metadata.set_sequence_key('entity', 'table')

    metadata.set_sequence_index('date', 'table')

    return data, metadata


def test_par():
    """Test the ``PARSynthesizer`` end to end."""
    # Setup
    data = load_demo()
    data['date'] = pd.to_datetime(data['date'])
    metadata = Metadata.detect_from_dataframes({'table': data})
    metadata.update_column('store_id', 'table', sdtype='id')
    metadata.set_sequence_key('store_id', 'table')

    metadata.set_sequence_index('date', 'table')

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
    assert all(sampled['total_sales'].round(2) == sampled['total_sales'])


def test_column_after_date_simple():
    """Test that adding a column after the ``sequence_index`` column works."""
    # Setup
    date = datetime.datetime.strptime('2020-01-01', '%Y-%m-%d')
    data = pd.DataFrame({
        'col': [1, 1],
        'date': [date, date],
        'col2': ['hello', 'world'],
    })
    metadata = Metadata.detect_from_dataframes({'table': data})
    metadata.update_column('col', 'table', sdtype='id')
    metadata.set_sequence_key('col', 'table')

    metadata.set_sequence_index('date', 'table')

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
    model = PARSynthesizer(metadata=metadata, context_columns=['context', 'context_date'], epochs=1)
    model.fit(data)
    sampled = model.sample(2)
    context_columns = data[['context', 'context_date']]
    sample_with_conditions = model.sample_sequential_columns(context_columns=context_columns)

    # Assert
    assert sampled.shape == data.shape
    assert (sampled.dtypes == data.dtypes).all()
    assert (sampled.notna().sum(axis=1) != 0).all()

    expected_date = datetime.datetime.strptime('2020-01-01', '%Y-%m-%d')
    assert all(sample_with_conditions['context_date'] == expected_date)
    assert all(sample_with_conditions['context'].isin(['a', 'b']))


def test_save_and_load(tmp_path):
    """Test that synthesizers can be saved and loaded properly."""
    # Setup
    _, metadata = _get_par_data_and_metadata()
    instance = PARSynthesizer(metadata, epochs=1)
    synthesizer_path = tmp_path / 'synthesizer.pkl'
    instance.save(synthesizer_path)

    # Run
    loaded_instance = PARSynthesizer.load(synthesizer_path)

    # Assert
    assert isinstance(loaded_instance, PARSynthesizer)
    assert metadata.to_dict() == instance.metadata.to_dict()


def test_synthesize_sequences(tmp_path):
    """End to end test for synthesizing sequences.

    The following functionalities are being tested:
        * Fit a ``PARSynthesizer`` with the demo dataset.
        * Fit a ``PARSynthesizer`` with custom context.
        * Sample from the model.
        * Conditionally sample from the model.
        * Save and Load.
    """
    # Setup
    real_data, metadata = download_demo(modality='sequential', dataset_name='nasdaq100_2019')
    assert real_data[real_data['Symbol'] == 'AMZN']['Sector'].unique()
    synthesizer = PARSynthesizer(metadata, epochs=1, context_columns=['Sector', 'Industry'])
    custom_synthesizer = PARSynthesizer(
        metadata, epochs=1, context_columns=['Sector', 'Industry'], verbose=True
    )
    scenario_context = pd.DataFrame(
        data={
            'Symbol': ['COMPANY-A', 'COMPANY-B', 'COMPANY-C', 'COMPANY-D', 'COMPANY-E'],
            'Sector': ['Technology'] * 2 + ['Consumer Services'] * 3,
            'Industry': [
                'Computer Manufacturing',
                'Computer Software: Prepackaged Software',
                'Hotels/Resorts',
                'Restaurants',
                'Clothing/Shoe/Accessory Stores',
            ],
        }
    )

    # Run - Fit
    synthesizer.fit(real_data)
    custom_synthesizer.fit(real_data)

    # Run - Sample
    synthetic_data = synthesizer.sample(num_sequences=10)
    custom_synthetic_data = custom_synthesizer.sample(num_sequences=3, sequence_length=2)
    custom_synthetic_data_conditional = custom_synthesizer.sample_sequential_columns(
        context_columns=scenario_context, sequence_length=2
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
        'Clothing/Shoe/Accessory Stores',
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
    data, metadata = download_demo(
        modality='sequential',
        dataset_name='nasdaq100_2019',
    )

    # modify the data by choosing a subset of it
    data_subset = data.copy()
    np.random.seed(1234)
    symbols = data['Symbol'].unique()

    # only select a subset of data in each sequence
    for i, symbol in enumerate(symbols):
        symbol_mask = data_subset['Symbol'] == symbol
        data_subset = data_subset.drop(
            data_subset[symbol_mask].sample(frac=i / (2 * len(symbols))).index
        )

    # now run PAR
    synthesizer = PARSynthesizer(metadata, epochs=1, verbose=True)
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
    metadata = Metadata.load_from_dict({
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
        'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1',
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
        'columns': {'value': {'sdtype': 'numerical'}, 'e_id': {'sdtype': 'id'}},
        'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1',
        'sequence_key': 'e_id',
    }

    metadata = Metadata().load_from_dict(metadata_dict)

    data = pd.DataFrame({'value': [10, 20, 30], 'e_id': [1, 2, 3]})

    # Run
    synthesizer = PARSynthesizer(metadata, epochs=1)
    synthesizer.fit(data)
    sampled = synthesizer.sample(num_sequences=3)

    # Assert
    assert sampled.shape == data.shape
    assert (sampled.dtypes == data.dtypes).all()


@pytest.mark.skip('Old-style constraints are deprecated')
def test_constraints_on_par():
    """Test if only simple constraints work on PARSynthesizer."""
    # Setup
    real_data, metadata = download_demo(modality='sequential', dataset_name='nasdaq100_2019')

    synthesizer = PARSynthesizer(metadata, epochs=1, context_columns=['Sector', 'Industry'])

    market_constraint = {
        'constraint_class': 'Positive',
        'constraint_parameters': {'column_name': 'MarketCap', 'strict_boundaries': True},
    }
    volume_constraint = {
        'constraint_class': 'Positive',
        'constraint_parameters': {'column_name': 'Volume', 'strict_boundaries': True},
    }

    context_constraint = {
        'constraint_class': 'Mock',
        'constraint_parameters': {'column_name': 'Sector', 'strict_boundaries': True},
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
        '2021-01-01',
        '2021-01-03',
        '2021-01-05',
        '2021-01-07',
        '2021-01-09',
        '2021-09-11',
        '2021-09-17',
        '2021-10-01',
        '2021-10-08',
        '2021-11-01',
    ]
    pre_date = [
        '2020-01-01',
        '2020-01-02',
        '2020-01-03',
        '2020-01-04',
        '2020-01-05',
        '2021-04-01',
        '2021-04-02',
        '2021-04-03',
        '2021-04-04',
        '2021-04-05',
    ]
    test_df = pd.DataFrame({'id': test_id, 's_key': s_key, 'visits': visits, 'pre_date': pre_date})
    test_df[['visits', 'pre_date']] = test_df[['visits', 'pre_date']].apply(
        pd.to_datetime, format='%Y-%m-%d', errors='coerce'
    )
    metadata = Metadata.detect_from_dataframes({'table': test_df})
    metadata.update_column(table_name='table', column_name='s_key', sdtype='id')
    metadata.set_sequence_key('s_key', 'table')

    metadata.set_sequence_index('visits', 'table')

    synthesizer = PARSynthesizer(
        metadata, enforce_min_max_values=True, enforce_rounding=False, epochs=1, verbose=True
    )

    # Run
    synthesizer.fit(test_df)
    synth_df = synthesizer.sample(num_sequences=50, sequence_length=50)

    # Assert
    for i in synth_df['s_key'].unique():
        seq_df = synth_df[synth_df['s_key'] == i]
        has_duplicates = seq_df['visits'].duplicated().any()
        assert not has_duplicates


def test_par_sequence_index_is_numerical():
    metadata_dict = {
        'sequence_index': 'time_in_cycles',
        'columns': {
            'engine_no': {'sdtype': 'id'},
            'time_in_cycles': {'sdtype': 'numerical'},
        },
        'sequence_key': 'engine_no',
        'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1',
    }
    metadata = Metadata.load_from_dict(metadata_dict)
    data = pd.DataFrame({'engine_no': [0, 0, 1, 1], 'time_in_cycles': [1, 2, 3, 4]})

    s1 = PARSynthesizer(metadata, epochs=1)
    s1.fit(data)
    sample = s1.sample(2, 5)
    assert sample.columns.to_list() == data.columns.to_list()


def test_init_error_sequence_key_in_context():
    # Setup
    metadata_dict = {
        'columns': {
            'A': {'sdtype': 'id'},
            'B': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
        },
        'sequence_key': 'A',
    }
    metadata = Metadata.load_from_dict(metadata_dict)
    sequence_key_context_column_error_msg = re.escape(
        "The sequence key ['A'] cannot be a context column. "
        'To proceed, please remove the sequence key from the context_columns parameter.'
    )
    # Run and Assert
    with pytest.raises(SynthesizerInputError, match=sequence_key_context_column_error_msg):
        PARSynthesizer(metadata, context_columns=['A'], epochs=1)


def test_par_with_datetime_context():
    """Test PARSynthesizer with a datetime as a context column"""
    # Setup
    data = pd.DataFrame(
        data={
            'user_id': ['ID_00'] * 5 + ['ID_01'] * 5,
            'birthdate': ['1995-05-06'] * 5 + ['1982-01-21'] * 5,
            'timestamp': ['2023-06-21', '2023-06-22', '2023-06-23', '2023-06-24', '2023-06-25'] * 2,
            'heartrate': [67, 66, 68, 65, 64, 80, 82, 91, 88, 84],
        }
    )

    metadata = Metadata.load_from_dict({
        'columns': {
            'user_id': {'sdtype': 'id', 'regex_format': 'ID_[0-9]{2}'},
            'birthdate': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
            'timestamp': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
            'heartrate': {'sdtype': 'numerical'},
        },
        'sequence_key': 'user_id',
        'sequence_index': 'timestamp',
    })

    # Run
    synth = PARSynthesizer(metadata, epochs=1, verbose=True, context_columns=['birthdate'])

    synth.fit(data)
    sample = synth.sample(num_sequences=1)
    expected_birthdate = pd.Series(['1984-02-23'] * 5, name='birthdate')

    # Assert
    pd.testing.assert_series_equal(sample['birthdate'], expected_birthdate)


def test_par_categorical_column_represented_by_floats():
    """Test to see if categorical columns work fine with float representation."""
    # Setup
    data, metadata = download_demo('sequential', 'nasdaq100_2019')
    data['category'] = [100.0 if i % 2 == 0 else 50.0 for i in data.index]
    metadata.add_column('category', 'nasdaq100_2019', sdtype='categorical')

    # Run
    synth = PARSynthesizer(metadata, epochs=1)
    synth.fit(data)
    sampled = synth.sample(num_sequences=10)

    # Assert
    synth.validate(sampled)
    assert sampled['category'].isin(data['category']).all()


def test_par_categorical_column_updated_to_float():
    """Test updating the transformer of a categorical column to float works GH #2482.

    Run on 100k rows. If the model treats the numerical data properly, it takes ~3s.
    If it treats it as categorical, it runs out of RAM.
    """
    # Setup
    data = pd.DataFrame(
        data={
            'sequence_key': ['key-' + str(int(i / 100)) for i in range(100000)],
            'column': np.random.choice(['value-' + str(i) for i in range(100)], size=100000),
        }
    )

    metadata = Metadata.load_from_dict({
        'tables': {
            'table': {
                'sequence_key': 'sequence_key',
                'columns': {
                    'sequence_key': {'sdtype': 'id'},
                    'column': {'sdtype': 'categorical'},
                },
            }
        }
    })

    # Run
    synthesizer = PARSynthesizer(metadata, epochs=1)
    synthesizer.auto_assign_transformers(data)
    synthesizer.update_transformers({'column': UniformEncoder()})
    synthesizer.fit(data)
    sampled = synthesizer.sample(num_sequences=10)

    # Assert
    assert sampled['column'].isin(data['column']).all()
