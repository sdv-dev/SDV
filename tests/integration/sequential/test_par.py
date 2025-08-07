import datetime
import re
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from deepecho import load_demo
from rdt.transformers.categorical import UniformEncoder

from sdv.cag import FixedCombinations, OneHotEncoding
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


def test_with_constraints():
    """Test constraint works on PARSynthesizer."""
    # Setup
    real_data, metadata = download_demo(modality='sequential', dataset_name='nasdaq100_2019')
    synthesizer = PARSynthesizer(metadata, epochs=1)
    constraint = FixedCombinations(column_names=['Sector', 'Industry'])

    # Run
    synthesizer.add_constraints([constraint])
    synthesizer.fit(real_data)
    samples = synthesizer.sample(50, 10)

    # Assert
    real_data_pairs = zip(
        real_data['Sector'].apply(lambda x: None if pd.isna(x) else x),
        real_data['Industry'].apply(lambda x: None if pd.isna(x) else x),
    )
    sample_pairs = zip(
        samples['Sector'].apply(lambda x: None if pd.isna(x) else x),
        samples['Industry'].apply(lambda x: None if pd.isna(x) else x),
    )
    original_combos = set(real_data_pairs)
    synthetic_combos = set(sample_pairs)
    assert synthetic_combos.issubset(original_combos)


def test_constraints_and_context_column():
    """Test constraint works with context columns."""
    # Setup
    real_data, metadata = download_demo(modality='sequential', dataset_name='nasdaq100_2019')
    synthesizer = PARSynthesizer(metadata, epochs=1, context_columns=['Sector', 'Industry'])
    constraint = FixedCombinations(column_names=['Sector', 'Industry'])

    # Run
    synthesizer.add_constraints([constraint])
    synthesizer.fit(real_data)
    samples = synthesizer.sample(50, 10)

    # Assert
    real_data_pairs = zip(
        real_data['Sector'].apply(lambda x: None if pd.isna(x) else x),
        real_data['Industry'].apply(lambda x: None if pd.isna(x) else x),
    )
    sample_pairs = zip(
        samples['Sector'].apply(lambda x: None if pd.isna(x) else x),
        samples['Industry'].apply(lambda x: None if pd.isna(x) else x),
    )
    original_combos = set(real_data_pairs)
    synthetic_combos = set(sample_pairs)
    assert synthetic_combos.issubset(original_combos)


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


@patch('sdv.sequential.par.PARModel', None)
@patch('sdv.sequential.par.import_error')
def test___init___without_torch(mock_import_error):
    """Test PAR raises a custom error when initialized with torch not installed."""
    # Setup
    _, metadata = _get_par_data_and_metadata()
    mock_import_error.name = 'torch'
    mock_import_error.msg = "No module named 'torch'"
    msg = "No module named 'torch'. Please install torch in order to use the 'PARSynthesizer'."

    # Run and Assert
    with pytest.raises(ModuleNotFoundError, match=msg):
        PARSynthesizer(metadata)


def test_par_with_all_null_column():
    """Test that the method handles all-null columns correctly."""
    # Setup
    data = pd.DataFrame(
        data={
            'sequence_key': ['sequence-' + str(int(i / 5)) for i in range(100)],
            'numerical_col': np.random.randint(low=0, high=100, size=100),
            'categorical_col': np.random.choice(['A', 'B', 'C'], size=100),
            'all_null_col': [np.nan] * 100,
        }
    )

    metadata = Metadata.load_from_dict({
        'tables': {
            'table': {
                'columns': {
                    'sequence_key': {'sdtype': 'id'},
                    'numerical_col': {'sdtype': 'numerical'},
                    'categorical_col': {'sdtype': 'categorical'},
                    'all_null_col': {'sdtype': 'numerical'},
                },
                'sequence_key': 'sequence_key',
            }
        }
    })

    synthesizer = PARSynthesizer(metadata, epochs=1)

    # Run
    synthesizer.fit(data)
    result = synthesizer.sample(num_sequences=2)

    # Assert
    assert 'all_null_col' in result.columns
    assert result['all_null_col'].isna().all()
    assert len(result) > 0


def test_par_unique_sequence_key_with_regex():
    """Test that the method handles unique sequence key with regex correctly."""
    # Setup
    data = pd.DataFrame(
        data={
            'sequence_key': ['seq-0'] * 5 + ['seq-1'] * 2 + ['seq-2'] * 3,
            'column_A': np.random.randint(low=0, high=10, size=10),
            'column_B': np.random.choice(['Yes', 'No', 'Maybe'], size=10),
        }
    )

    metadata = Metadata.load_from_dict({
        'tables': {
            'table': {
                'sequence_key': 'sequence_key',
                'columns': {
                    'sequence_key': {'sdtype': 'id', 'regex_format': 'seq-[0-9]'},
                    'column_A': {'sdtype': 'numerical'},
                    'column_B': {'sdtype': 'categorical'},
                },
            }
        }
    })

    # Run
    synthesizer = PARSynthesizer(metadata, epochs=1)
    synthesizer.fit(data)
    sample = synthesizer.sample(num_sequences=20)

    # Assert
    assert sample['sequence_key'].nunique() == 20
    transformer = synthesizer._context_synthesizer.get_transformers()['sequence_key']
    transformer.cardinality_rule == 'unique'


def test_par_with_context_column_as_id():
    """Test PARSynthesizer with a context column as an id column."""
    # Setup
    data = pd.DataFrame(
        data={
            'event_id': ['event-000'] * 5 + ['event-001'] * 2 + ['event-002'] * 3,
            'event_source': ['source-AAA'] * 5 + ['source-BBB'] * 2 + ['source-CCC'] * 3,
            'column_A': np.random.randint(low=0, high=10, size=10),
            'column_B': np.random.choice(['Yes', 'No', 'Maybe'], size=10),
        }
    )

    metadata = Metadata.load_from_dict({
        'tables': {
            'table': {
                'sequence_key': 'event_id',
                'columns': {
                    'event_id': {'sdtype': 'id', 'regex_format': 'event-[0-9]{3,4}'},
                    'event_source': {'sdtype': 'id', 'regex_format': 'source-[A-Z]{3,5}'},
                    'column_A': {'sdtype': 'numerical'},
                    'column_B': {'sdtype': 'categorical'},
                },
            }
        }
    })

    # Run
    synthesizer = PARSynthesizer(metadata, epochs=1, context_columns=['event_source'])
    synthesizer.fit(data)
    sampled = synthesizer.sample(num_sequences=2)

    # Assert
    assert sampled['event_id'].isin(data['event_id']).all()
    assert sampled['column_B'].isin(data['column_B']).all()


def test_par_with_multiple_id_context_columns():
    """Test PARSynthesizer with multiple context columns of id sdtype."""
    # Setup
    data = pd.DataFrame(
        data={
            'sequence_id': ['seq-001'] * 4 + ['seq-002'] * 4 + ['seq-003'] * 2,
            'user_id': ['user-A01'] * 4 + ['user-B02'] * 4 + ['user-C03'] * 2,
            'device_id': ['device-X'] * 4 + ['device-Y'] * 4 + ['device-Z'] * 2,
            'session_id': ['sess-123'] * 4 + ['sess-456'] * 4 + ['sess-789'] * 2,
            'timestamp': ['2023-06-21', '2023-06-22', '2023-06-23', '2023-06-24', '2023-06-25'] * 2,
            'value': np.random.randint(10, 100, size=10),
        }
    )

    metadata = Metadata.load_from_dict({
        'tables': {
            'table': {
                'sequence_key': 'sequence_id',
                'sequence_index': 'timestamp',
                'columns': {
                    'sequence_id': {'sdtype': 'id', 'regex_format': 'seq-[0-9]{3}'},
                    'user_id': {'sdtype': 'id', 'regex_format': 'user-[A-Z][0-9]{2}'},
                    'device_id': {'sdtype': 'id', 'regex_format': 'device-[A-Z]'},
                    'session_id': {'sdtype': 'id', 'regex_format': 'sess-[0-9]{3}'},
                    'timestamp': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
                    'value': {'sdtype': 'numerical'},
                },
            }
        }
    })

    # Run
    synthesizer = PARSynthesizer(
        metadata, epochs=1, context_columns=['user_id', 'device_id', 'session_id']
    )
    synthesizer.fit(data)
    sampled = synthesizer.sample(num_sequences=4)

    # Assert
    assert len(sampled) > 0
    assert set(sampled.columns) == set(data.columns)
    for seq_id in sampled['sequence_id'].unique():
        seq_data = sampled[sampled['sequence_id'] == seq_id]
        assert seq_data['user_id'].nunique() == 1
        assert seq_data['device_id'].nunique() == 1
        assert seq_data['session_id'].nunique() == 1


def test_par_with_pii_context_columns():
    """Test PARSynthesizer with PII context columns."""
    # Setup
    data = pd.DataFrame(
        data={
            'sequence_id': ['seq-001'] * 4 + ['seq-002'] * 4 + ['seq-003'] * 2,
            'user_id': ['user-A01'] * 4 + ['user-B02'] * 4 + ['user-C03'] * 2,
            'device_id': ['device-X'] * 4 + ['device-Y'] * 4 + ['device-Z'] * 2,
            'session_id': ['sess-123'] * 4 + ['sess-456'] * 4 + ['sess-789'] * 2,
            'timestamp': ['2023-06-21', '2023-06-22', '2023-06-23', '2023-06-24', '2023-06-25'] * 2,
            'value': np.random.randint(10, 100, size=10),
        }
    )

    metadata = Metadata.load_from_dict({
        'tables': {
            'table': {
                'sequence_key': 'sequence_id',
                'sequence_index': 'timestamp',
                'columns': {
                    'sequence_id': {'sdtype': 'id', 'regex_format': 'seq-[0-9]{3}'},
                    'user_id': {'sdtype': 'ssn'},
                    'device_id': {'sdtype': 'email'},
                    'session_id': {'sdtype': 'id', 'regex_format': 'sess-[0-9]{3}'},
                    'timestamp': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
                    'value': {'sdtype': 'numerical'},
                },
            }
        }
    })

    # Run
    synthesizer = PARSynthesizer(
        metadata, epochs=1, context_columns=['user_id', 'device_id', 'session_id']
    )
    synthesizer.fit(data)
    sampled = synthesizer.sample(num_sequences=4)

    # Assert
    assert len(sampled) > 0
    assert set(sampled.columns) == set(data.columns)
    for seq_id in sampled['sequence_id'].unique():
        seq_data = sampled[sampled['sequence_id'] == seq_id]
        assert seq_data['user_id'].nunique() == 1
        assert seq_data['device_id'].nunique() == 1
        assert seq_data['session_id'].nunique() == 1


def test_par_with_mixed_context_columns_including_id():
    """Test PARSynthesizer with mixed context columns including id sdtype."""
    # Setup
    data = pd.DataFrame(
        data={
            'patient_id': ['P001'] * 5 + ['P002'] * 5 + ['P003'] * 5,
            'hospital_id': ['H_123'] * 5 + ['H_456'] * 5 + ['H_789'] * 5,
            'doctor_category': ['Cardiology'] * 5 + ['Neurology'] * 5 + ['Orthopedics'] * 5,
            'treatment_date': pd.date_range('2023-01-01', periods=15, freq='1D'),
            'vital_reading': np.random.uniform(70, 120, size=15),
            'recovery_score': np.random.randint(1, 11, size=15),
        }
    )

    metadata = Metadata.load_from_dict({
        'tables': {
            'table': {
                'sequence_key': 'patient_id',
                'sequence_index': 'treatment_date',
                'columns': {
                    'patient_id': {'sdtype': 'id', 'regex_format': 'P[0-9]{3}'},
                    'hospital_id': {'sdtype': 'id', 'regex_format': 'H_[0-9]{3}'},
                    'doctor_category': {'sdtype': 'categorical'},
                    'treatment_date': {'sdtype': 'datetime'},
                    'vital_reading': {'sdtype': 'numerical'},
                    'recovery_score': {'sdtype': 'numerical'},
                },
            }
        }
    })

    # Run
    synthesizer = PARSynthesizer(
        metadata, epochs=1, context_columns=['hospital_id', 'doctor_category']
    )
    synthesizer.fit(data)
    sampled = synthesizer.sample(num_sequences=4)

    # Assert
    assert len(sampled) > 0
    assert set(sampled.columns) == set(data.columns)

    # Check context columns remain constant within sequences
    for patient_id in sampled['patient_id'].unique():
        patient_data = sampled[sampled['patient_id'] == patient_id]
        assert patient_data['hospital_id'].nunique() == 1
        assert patient_data['doctor_category'].nunique() == 1


def test_par_sample_sequential_columns_with_id_context():
    """Test sample_sequential_columns method with id sdtype context columns."""
    # Setup
    data = pd.DataFrame(
        data={
            'order_id': ['ORD-001'] * 3 + ['ORD-002'] * 3 + ['ORD-003'] * 4,
            'customer_id': ['CUST-A'] * 3 + ['CUST-B'] * 3 + ['CUST-C'] * 4,
            'product_category': ['Electronics'] * 3 + ['Books'] * 3 + ['Clothing'] * 4,
            'order_date': pd.to_datetime(
                ['2023-01-01', '2023-01-02', '2023-01-03'] * 3 + ['2023-01-01']
            ),
            'quantity': [1, 2, 1, 3, 1, 2, 2, 1, 1, 3],
        }
    )

    metadata = Metadata.load_from_dict({
        'tables': {
            'table': {
                'sequence_key': 'order_id',
                'sequence_index': 'order_date',
                'columns': {
                    'order_id': {'sdtype': 'id', 'regex_format': 'ORD-[0-9]{3}'},
                    'customer_id': {'sdtype': 'id', 'regex_format': 'CUST-[A-Z]'},
                    'product_category': {'sdtype': 'categorical'},
                    'order_date': {'sdtype': 'datetime'},
                    'quantity': {'sdtype': 'numerical'},
                },
            }
        }
    })

    # Prepare context columns for conditional sampling
    context_df = pd.DataFrame({
        'order_id': ['ORD-NEW1', 'ORD-NEW2'],
        'customer_id': ['CUST-A', 'CUST-B'],
        'product_category': ['Electronics', 'Books'],
    })

    # Run
    synthesizer = PARSynthesizer(
        metadata, epochs=1, context_columns=['customer_id', 'product_category']
    )
    synthesizer.fit(data)
    sampled = synthesizer.sample_sequential_columns(context_columns=context_df, sequence_length=3)

    # Assert
    assert len(sampled) == 6  # 2 sequences * 3 length
    assert set(sampled.columns) == set(data.columns)

    seq1_data = sampled[sampled['order_id'] == 'ORD-NEW1']
    seq2_data = sampled[sampled['order_id'] == 'ORD-NEW2']

    assert all(seq1_data['product_category'] == 'Electronics')
    assert all(seq2_data['product_category'] == 'Books')


def test_par_save_load_with_id_context_columns(tmp_path):
    """Test save and load functionality with id sdtype context columns."""
    # Setup
    data = pd.DataFrame(
        data={
            'machine_id': ['M001'] * 5 + ['M002'] * 5,
            'operator_id': ['OP_A'] * 5 + ['OP_B'] * 5,
            'shift_id': ['SHIFT_DAY'] * 5 + ['SHIFT_NIGHT'] * 5,
            'timestamp': pd.date_range('2023-05-01', periods=10, freq='4H'),
            'temperature': np.random.uniform(20, 80, size=10),
            'pressure': np.random.uniform(100, 200, size=10),
        }
    )

    metadata = Metadata.load_from_dict({
        'tables': {
            'table': {
                'sequence_key': 'machine_id',
                'sequence_index': 'timestamp',
                'columns': {
                    'machine_id': {'sdtype': 'id', 'regex_format': 'M[0-9]{3}'},
                    'operator_id': {'sdtype': 'id', 'regex_format': 'OP_[A-Z]'},
                    'shift_id': {'sdtype': 'id', 'regex_format': 'SHIFT_[A-Z]+'},
                    'timestamp': {'sdtype': 'datetime'},
                    'temperature': {'sdtype': 'numerical'},
                    'pressure': {'sdtype': 'numerical'},
                },
            }
        }
    })

    # Create and fit synthesizer
    synthesizer = PARSynthesizer(metadata, epochs=1, context_columns=['operator_id', 'shift_id'])
    synthesizer.fit(data)

    # Save synthesizer
    save_path = tmp_path / 'par_synthesizer_with_id_context.pkl'
    synthesizer.save(save_path)

    # Load synthesizer
    loaded_synthesizer = PARSynthesizer.load(save_path)
    loaded_sample = loaded_synthesizer.sample(num_sequences=2)

    # Assert
    assert save_path.exists()
    assert isinstance(loaded_synthesizer, PARSynthesizer)
    assert loaded_synthesizer.context_columns == synthesizer.context_columns
    assert loaded_synthesizer.metadata.to_dict() == synthesizer.metadata.to_dict()

    # Test that loaded synthesizer produces valid samples
    assert len(loaded_sample) > 0
    assert set(loaded_sample.columns) == set(data.columns)

    # Verify context columns remain constant in loaded synthesizer samples
    for machine_id in loaded_sample['machine_id'].unique():
        machine_data = loaded_sample[loaded_sample['machine_id'] == machine_id]
        assert machine_data['operator_id'].nunique() == 1
        assert machine_data['shift_id'].nunique() == 1

    # Validate samples from loaded synthesizer
    loaded_synthesizer.validate(loaded_sample)
    synthesizer.validate(loaded_sample)


def test_add_constraints_mixed_context_and_non_context():
    """Test adding mixed constraints (some context, some non-context)."""
    from sdv.cag import FixedCombinations

    # Setup
    data = pd.DataFrame({
        'seq_id': ['seq_0'] * 4 + ['seq_1'] * 3 + ['seq_2'] * 3,
        'context_col': ['A'] * 4 + ['B'] * 3 + ['A'] * 3,
        'other_context': ['X'] * 4 + ['Y'] * 3 + ['Z'] * 3,
        'seq_col': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'numerical': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    })

    metadata = Metadata.load_from_dict({
        'tables': {
            'table': {
                'sequence_key': 'seq_id',
                'columns': {
                    'seq_id': {'sdtype': 'id'},
                    'context_col': {'sdtype': 'categorical'},
                    'other_context': {'sdtype': 'categorical'},
                    'seq_col': {'sdtype': 'numerical'},
                    'numerical': {'sdtype': 'numerical'},
                },
            }
        }
    })

    synthesizer = PARSynthesizer(
        metadata, epochs=1, context_columns=['context_col', 'other_context']
    )

    context_constraint = FixedCombinations(column_names=['context_col', 'other_context'])

    # Run
    synthesizer.add_constraints([context_constraint])
    synthesizer.fit(data)
    samples = synthesizer.sample(num_sequences=5)

    # Assert
    real_data_pairs = set(zip(data['context_col'], data['other_context']))
    sample_pairs = set(zip(samples['context_col'], samples['other_context']))
    assert sample_pairs.issubset(real_data_pairs)

    synthesizer.validate(samples)


def test_add_constraints_with_context_columns():
    """Test adding constraints with context columns."""
    # Setup
    data = pd.DataFrame(
        data={
            'seq_id': ['seq_0'] * 4 + ['seq_1'] * 3 + ['seq_2'] * 3,
            'context_0': [0] * 4 + [1] * 3 + [0] * 3,
            'context_1': [1] * 4 + [0] * 3 + [0] * 3,
            'context_2': [0] * 4 + [0] * 3 + [1] * 3,
            'other_context_col': [0.10] * 4 + [0.23] * 3 + [0.24] * 3,
            'seq_0': [0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
            'seq_1': [1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            'seq_2': [0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
            'numerical': [0.23, 0.34, 0.56, 0.67, 0.22, 0.23, 0.26, 0.20, 0.34, 0.45],
        }
    )

    metadata = Metadata.load_from_dict({
        'tables': {
            'table': {
                'sequence_key': 'seq_id',
                'columns': {
                    'seq_id': {'sdtype': 'id', 'regex_format': r'seq_\d{1,2}'},
                    'context_0': {'sdtype': 'categorical'},
                    'context_1': {'sdtype': 'categorical'},
                    'context_2': {'sdtype': 'categorical'},
                    'other_context_col': {'sdtype': 'numerical'},
                    'seq_0': {'sdtype': 'categorical'},
                    'seq_1': {'sdtype': 'categorical'},
                    'seq_2': {'sdtype': 'categorical'},
                    'numerical': {'sdtype': 'numerical'},
                },
            }
        }
    })

    synthesizer = PARSynthesizer(metadata, context_columns=['context_0', 'context_1', 'context_2'])

    context_constraint = OneHotEncoding(column_names=['context_0', 'context_1', 'context_2'])

    seq_constraint = OneHotEncoding(column_names=['seq_0', 'seq_1', 'seq_2'])

    synthesizer.add_constraints([context_constraint, seq_constraint])
    synthesizer.fit(data)
    samples = synthesizer.sample(5)
    synthesizer.validate(samples)
