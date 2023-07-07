import datetime
import re

import numpy as np
import pandas as pd
import pkg_resources
import pytest
from faker import Faker

from sdv.datasets.demo import download_demo
from sdv.datasets.local import load_csvs
from sdv.evaluation.multi_table import evaluate_quality, get_column_pair_plot, get_column_plot
from sdv.metadata.multi_table import MultiTableMetadata
from sdv.multi_table import HMASynthesizer
from tests.integration.single_table.custom_constraints import MyConstraint


def test_hma():
    """End to end integration tests with ``HMASynthesizer``.

    The test consist on loading the demo data, convert the old metadata to the new format
    and then fit a ``HMASynthesizer``. After fitting two samples are being generated, one with
    a 0.5 scale and one with 1.5 scale.
    """
    # Setup
    data, metadata = download_demo('multi_table', 'got_families')
    hmasynthesizer = HMASynthesizer(metadata)

    # Run
    hmasynthesizer.fit(data)
    normal_sample = hmasynthesizer.sample(0.5)
    increased_sample = hmasynthesizer.sample(1.5)

    # Assert
    assert list(normal_sample) == ['characters', 'character_families', 'families']
    assert list(increased_sample) == ['characters', 'character_families', 'families']
    for table_name, table in normal_sample.items():
        assert all(table.columns == data[table_name].columns)

    for normal_table, increased_table in zip(normal_sample.values(), increased_sample.values()):
        assert increased_table.size > normal_table.size


def test_hma_reset_sampling():
    """End to end integration test that uses ``reset_sampling``.

    This test uses ``reset_sampling`` to ensure that the model will generate the same data
    as the first sample after this method has been called.
    """
    # Setup
    faker = Faker()
    data, metadata = download_demo('multi_table', 'got_families')
    metadata.add_column(
        'characters',
        'ssn',
        sdtype='ssn',
    )
    data['characters']['ssn'] = [faker.lexify() for _ in range(len(data['characters']))]
    for table in metadata.tables.values():
        table.alternate_keys = []

    hmasynthesizer = HMASynthesizer(metadata)

    # Run
    hmasynthesizer.fit(data)
    first_sample = hmasynthesizer.sample()
    second_sample = hmasynthesizer.sample()
    hmasynthesizer.reset_sampling()
    reset_first_sample = hmasynthesizer.sample()
    reset_second_sample = hmasynthesizer.sample()

    # Assert
    for table, reset_table in zip(first_sample.values(), reset_first_sample.values()):
        pd.testing.assert_frame_equal(table, reset_table)

    for table, reset_table in zip(second_sample.values(), reset_second_sample.values()):
        pd.testing.assert_frame_equal(table, reset_table)

    for sample_1, sample_2 in zip(first_sample.values(), second_sample.values()):
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(sample_1, sample_2)


def test_get_info():
    """Test the correct dictionary is returned.

    Check the return dictionary is valid both before and after fitting the synthesizer.
    """
    # Setup
    data = {'tab': pd.DataFrame({'col': [1, 2, 3]})}
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    metadata = MultiTableMetadata()
    metadata.add_table('tab')
    metadata.add_column('tab', 'col', sdtype='numerical')
    synthesizer = HMASynthesizer(metadata)

    # Run
    info = synthesizer.get_info()

    # Assert
    assert info == {
        'class_name': 'HMASynthesizer',
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
        'class_name': 'HMASynthesizer',
        'creation_date': today,
        'is_fit': True,
        'last_fit_date': today,
        'fitted_sdv_version': version
    }


def test_hma_set_parameters():
    """Test the ``set_table_parameters``.

    Validate that the ``set_table_parameters`` sets new parameters to the synthesizers.
    """
    # Setup
    data, metadata = download_demo('multi_table', 'got_families')
    hmasynthesizer = HMASynthesizer(metadata)

    # Run
    hmasynthesizer.set_table_parameters('characters', {'default_distribution': 'gamma'})
    hmasynthesizer.set_table_parameters('families', {'default_distribution': 'uniform'})
    hmasynthesizer.set_table_parameters('character_families', {'default_distribution': 'norm'})

    # Assert
    assert hmasynthesizer.get_table_parameters('characters') == {'default_distribution': 'gamma'}
    assert hmasynthesizer.get_table_parameters('families') == {'default_distribution': 'uniform'}
    assert hmasynthesizer.get_table_parameters('character_families') == {
        'default_distribution': 'norm'
    }

    assert hmasynthesizer._table_synthesizers['characters'].default_distribution == 'gamma'
    assert hmasynthesizer._table_synthesizers['families'].default_distribution == 'uniform'
    assert hmasynthesizer._table_synthesizers['character_families'].default_distribution == 'norm'


def get_custom_constraint_data_and_metadata():
    """Return data and metadata for the custom constraint tests."""
    parent_data = pd.DataFrame({
        'primary_key': [1000, 1001, 1002],
        'numerical_col': [2, 3, 4],
        'categorical_col': ['a', 'b', 'a'],
    })
    child_data = pd.DataFrame({
        'user_id': [1000, 1001, 1000],
        'id': [1, 2, 3],
        'random': ['a', 'b', 'c'],
        'numerical_col': [0.2, 0.7, 1.3],
        'numerical_col_2': [2, 4, 6],
    })

    metadata = MultiTableMetadata()
    metadata.detect_table_from_dataframe('parent', parent_data)
    metadata.update_column('parent', 'primary_key', sdtype='id')
    metadata.detect_table_from_dataframe('child', child_data)
    metadata.update_column('child', 'user_id', sdtype='id')
    metadata.update_column('child', 'id', sdtype='id')
    metadata.set_primary_key('parent', 'primary_key')
    metadata.set_primary_key('child', 'id')
    metadata.add_relationship(
        parent_primary_key='primary_key',
        parent_table_name='parent',
        child_foreign_key='user_id',
        child_table_name='child'
    )

    return parent_data, child_data, metadata


def test_hma_custom_constraint():
    """Test an example of using a custom constraint."""
    # Setup
    parent_data, child_data, metadata = get_custom_constraint_data_and_metadata()
    synthesizer = HMASynthesizer(metadata)
    constraint = {
        'table_name': 'parent',
        'constraint_class': 'MyConstraint',
        'constraint_parameters': {
            'column_names': ['numerical_col']
        }
    }
    synthesizer.add_custom_constraint_class(MyConstraint, 'MyConstraint')

    # Run
    synthesizer.add_constraints(constraints=[constraint])
    processed_data = synthesizer.preprocess({'parent': parent_data, 'child': child_data})

    # Assert Processed Data
    np.testing.assert_equal(
        processed_data['parent']['numerical_col'].array,
        (parent_data['numerical_col'] ** 2.0).array
    )

    # Run - Fit the model
    synthesizer.fit_processed_data(processed_data)

    # Run - sample
    sampled = synthesizer.sample(10)
    assert all(sampled['parent']['numerical_col'] > 1)


def test_hma_custom_constraint_2_tables():
    """Test an example of using a custom constraint.

    Check that the same custom constraint can be applied to two different tables and columns.
    """
    # Setup
    parent_data, child_data, metadata = get_custom_constraint_data_and_metadata()
    synthesizer = HMASynthesizer(metadata)

    constraint_parent = {
        'table_name': 'parent',
        'constraint_class': 'MyConstraint',
        'constraint_parameters': {
            'column_names': ['numerical_col']
        }
    }

    constraint_child = {
        'table_name': 'child',
        'constraint_class': 'MyConstraint',
        'constraint_parameters': {
            'column_names': ['numerical_col_2']
        }
    }
    synthesizer.add_custom_constraint_class(MyConstraint, 'MyConstraint')

    # Run
    synthesizer.add_constraints(constraints=[constraint_parent, constraint_child])
    processed_data = synthesizer.preprocess({'parent': parent_data, 'child': child_data})

    # Assert Processed Data
    np.testing.assert_equal(
        processed_data['parent']['numerical_col'].array,
        (parent_data['numerical_col'] ** 2.0).array
    )
    np.testing.assert_equal(
        processed_data['child']['numerical_col_2'].array,
        (child_data['numerical_col_2'] ** 2.0).array
    )

    # Run - Fit the model
    synthesizer.fit_processed_data(processed_data)

    # Run - sample
    sampled = synthesizer.sample(10)
    assert all(sampled['parent']['numerical_col'] > 1)
    assert all(sampled['child']['numerical_col_2'] > 1)
    assert not all(sampled['child']['numerical_col'] > 1)


def test_hma_custom_constraint_loaded_from_file():
    """Test an example of using a custom constraint loaded from a file."""
    # Setup
    parent_data, child_data, metadata = get_custom_constraint_data_and_metadata()
    synthesizer = HMASynthesizer(metadata)
    constraint = {
        'table_name': 'parent',
        'constraint_class': 'MyConstraint',
        'constraint_parameters': {
            'column_names': ['numerical_col']
        }
    }
    synthesizer.load_custom_constraint_classes(
        'tests/integration/single_table/custom_constraints.py',
        ['MyConstraint']
    )

    # Run
    synthesizer.add_constraints(constraints=[constraint])
    processed_data = synthesizer.preprocess({'parent': parent_data, 'child': child_data})

    # Assert Processed Data
    np.testing.assert_equal(
        processed_data['parent']['numerical_col'].array,
        (parent_data['numerical_col'] ** 2.0).array
    )

    # Run - Fit the model
    synthesizer.fit_processed_data(processed_data)

    # Run - sample
    sampled = synthesizer.sample(10)
    assert all(sampled['parent']['numerical_col'] > 1)


def test_hma_with_inequality_constraint():
    """Test that ensures that when new columns are created by the constraint this still works."""
    # Setup
    parent_table = pd.DataFrame(data={
        'id': [1, 2, 3, 4, 5],
        'column': [1.2, 2.1, 2.2, 2.1, 1.4]
    })

    child_table = pd.DataFrame(data={
        'id': [1, 2, 3, 4, 5],
        'parent_id': [1, 1, 3, 2, 1],
        'low_column': [1, 3, 3, 1, 2],
        'high_column': [2, 4, 5, 2, 4]
    })

    data = {
        'parent_table': parent_table,
        'child_table': child_table
    }

    metadata = MultiTableMetadata()
    metadata.detect_table_from_dataframe(table_name='parent_table', data=parent_table)
    metadata.update_column('parent_table', 'id', sdtype='id')
    metadata.detect_table_from_dataframe(table_name='child_table', data=child_table)
    metadata.update_column('child_table', 'id', sdtype='id')
    metadata.update_column('child_table', 'parent_id', sdtype='id')

    metadata.set_primary_key(table_name='parent_table', column_name='id')
    metadata.set_primary_key(table_name='child_table', column_name='id')

    metadata.add_relationship(
        parent_table_name='parent_table',
        child_table_name='child_table',
        parent_primary_key='id',
        child_foreign_key='parent_id'
    )

    constraint = {
        'constraint_class': 'Inequality',
        'table_name': 'child_table',
        'constraint_parameters': {
            'low_column_name': 'low_column',
            'high_column_name': 'high_column'
        }
    }

    synthesizer = HMASynthesizer(metadata)

    # Run
    synthesizer.add_constraints(constraints=[constraint])
    synthesizer.fit(data)
    sampled = synthesizer.sample(10)

    # Assert
    assert all(sampled['child_table']['low_column'] < sampled['child_table']['high_column'])


def test_fit_processed_multiple_calls():
    """Test that ``fit_processed_data`` does not modify input data."""
    # Setup
    parent_data, child_data, metadata = get_custom_constraint_data_and_metadata()
    synthesizer = HMASynthesizer(metadata)

    # Run
    preprocessed = synthesizer.preprocess({'parent': parent_data, 'child': child_data})
    parent_copy = preprocessed['parent'].copy()
    child_copy = preprocessed['child'].copy()

    synthesizer.fit_processed_data(preprocessed)

    # Assert
    assert preprocessed.keys() == {'parent', 'child'}
    pd.testing.assert_frame_equal(parent_copy, preprocessed['parent'], check_like=True)
    pd.testing.assert_frame_equal(child_copy, preprocessed['child'], check_like=True)

    # Re-run to ensure it does not error
    synthesizer.fit_processed_data(preprocessed)


def test_save_and_load(tmp_path):
    """Test saving and loading a multi-table synthesizer."""
    # Setup
    _, _, metadata = get_custom_constraint_data_and_metadata()
    synthesizer = HMASynthesizer(metadata)
    model_path = tmp_path / 'synthesizer.pkl'

    # Run
    synthesizer.save(model_path)

    # Assert
    assert model_path.exists()
    assert model_path.is_file()
    loaded_synthesizer = HMASynthesizer.load(model_path)

    assert isinstance(synthesizer, HMASynthesizer)
    assert loaded_synthesizer.get_info() == synthesizer.get_info()
    assert loaded_synthesizer.metadata.to_dict() == metadata.to_dict()


def test_hma_primary_key_and_foreign_key_only():
    """Test that ``HMASynthesizer`` can handle tables with primary and foreign keys only."""
    # Setup
    users = pd.DataFrame({
        'user_id': [1, 2, 3],
        'user_name': ['John', 'Doe', 'Johanna']
    })
    sessions = pd.DataFrame({
        'session_id': ['a', 'b', 'c'],
        'clicks': [10, 20, 30]
    })
    games = pd.DataFrame({
        'game_id': ['a1', 'b2', 'c3'],
        'session_id': ['a', 'b', 'c'],
        'user_id': [1, 2, 3]
    })

    data = {
        'users': users,
        'sessions': sessions,
        'games': games
    }

    metadata = MultiTableMetadata()
    for table_name, table in data.items():
        metadata.detect_table_from_dataframe(table_name, table)

    metadata.update_column('users', 'user_id', sdtype='id')
    metadata.update_column('sessions', 'session_id', sdtype='id')
    metadata.update_column('games', 'game_id', sdtype='id')
    metadata.update_column('games', 'session_id', sdtype='id')
    metadata.update_column('games', 'user_id', sdtype='id')
    metadata.set_primary_key('users', 'user_id')
    metadata.set_primary_key('sessions', 'session_id')
    metadata.set_primary_key('games', 'game_id')
    metadata.add_relationship('users', 'games', 'user_id', 'user_id')
    metadata.add_relationship('sessions', 'games', 'session_id', 'session_id')

    hmasynthesizer = HMASynthesizer(metadata)

    # Fit
    hmasynthesizer.fit(data)

    # Sample
    sample = hmasynthesizer.sample()

    # Assert
    assert all(sample['games']['user_id'].isin(sample['users']['user_id']))
    assert all(sample['games']['session_id'].isin(sample['sessions']['session_id']))


def test_synthesize_multiple_tables_using_hma(tmp_path):
    """End to end test for multiple tables using ``HMASynthesizer``.

    The following functionalities are being tested:
        * Create, fit and sample from model
        * Anonymization
        * Evaluating synthetic data
        * Saving, loading and sampling from the loaded model
        * Using a custom configuration for the ``HMASynthesizer``
    """

    # Loading the demo data
    real_data, metadata = download_demo(
        modality='multi_table',
        dataset_name='fake_hotels'
    )

    # Creating a Synthesizer
    synthesizer = HMASynthesizer(metadata)
    synthesizer.fit(real_data)

    # Generating Synthetic Data
    synthetic_data = synthesizer.sample(scale=2)

    # Assert new data is bigger than real_data
    for table_name in metadata.tables:
        assert len(synthetic_data[table_name]) > len(real_data[table_name])

    # Assert Anonymization
    sensitive_columns = ['guest_email', 'billing_address', 'credit_card_number']
    for column in sensitive_columns:
        assert synthetic_data['guests'][column].isin(real_data['guests'][column]).sum() == 0

    # Evaluate Real vs Synthetic Data
    quality_report = evaluate_quality(
        real_data,
        synthetic_data,
        metadata,
        verbose=False
    )

    column_plot = get_column_plot(
        real_data=real_data,
        synthetic_data=synthetic_data,
        column_name='has_rewards',
        table_name='guests',
        metadata=metadata
    )

    column_pair_plot = get_column_pair_plot(
        real_data=real_data,
        synthetic_data=synthetic_data,
        column_names=['room_rate', 'room_type'],
        table_name='guests',
        metadata=metadata
    )

    # Assert
    assert quality_report.get_score() > 0
    assert column_plot
    assert column_pair_plot

    # Save and Load
    model_path = tmp_path / 'synthesizer.pkl'

    synthesizer.save(model_path)

    # Assert
    assert model_path.exists()
    assert model_path.is_file()
    loaded_synthesizer = HMASynthesizer.load(model_path)

    assert isinstance(synthesizer, HMASynthesizer)
    assert loaded_synthesizer.get_info() == synthesizer.get_info()
    assert loaded_synthesizer.metadata.to_dict() == metadata.to_dict()
    loaded_synthesizer.sample()

    # HMA Customization
    custom_synthesizer = HMASynthesizer(
        metadata
    )

    custom_synthesizer.set_table_parameters(
        table_name='hotels',
        table_parameters={
            'default_distribution': 'truncnorm'
        }
    )

    custom_synthesizer.fit(real_data)
    learned_distributions = custom_synthesizer.get_learned_distributions(table_name='hotels')

    # Assert
    assert list(learned_distributions['rating']['learned_parameters']) == [
        'a',
        'b',
        'loc',
        'scale'
    ]
    assert learned_distributions['rating']['distribution'] == 'truncnorm'


def test_use_own_data_using_hma(tmp_path):
    """End to end test for loading and preparing data.

    Tests loading data from CSVs, auto-detecting metadata from the dataframes, updating the
    metadata with correct sdtypes, and finally fitting the synthesizer.
    """
    # Setup
    data_folder = tmp_path / 'datasets'
    download_demo(
        modality='multi_table',
        dataset_name='fake_hotels',
        output_folder_name=data_folder
    )

    # Run - load CSVs
    datasets = load_csvs(data_folder)

    # Assert - loaded CSVs correctly
    assert datasets.keys() == {'guests', 'hotels'}

    # Metadata
    metadata = MultiTableMetadata()

    metadata.detect_table_from_dataframe(
        table_name='guests',
        data=datasets['guests']
    )
    metadata.detect_table_from_dataframe(
        table_name='hotels',
        data=datasets['hotels']
    )

    # Assert - detected metadata correctly
    for table in metadata.tables:
        assert metadata.tables[table].columns.keys() == set(datasets[table].columns)

    # Update metadata
    metadata.update_column(
        table_name='guests',
        column_name='checkin_date',
        sdtype='datetime',
        datetime_format='%d %b %Y'
    )
    metadata.update_column(
        table_name='guests',
        column_name='checkout_date',
        sdtype='datetime',
        datetime_format='%d %b %Y'
    )
    metadata.update_column(
        table_name='hotels',
        column_name='hotel_id',
        sdtype='id',
        regex_format='HID_[0-9]{3,4}'
    )
    metadata.update_column(
        table_name='guests',
        column_name='hotel_id',
        sdtype='id',
        regex_format='HID_[0-9]{3,4}'
    )
    metadata.update_column(
        table_name='guests',
        column_name='guest_email',
        sdtype='email',
        pii=True
    )
    metadata.update_column(
        table_name='guests',
        column_name='billing_address',
        sdtype='address',
        pii=True
    )
    metadata.update_column(
        table_name='guests',
        column_name='credit_card_number',
        sdtype='credit_card_number',
        pii=True
    )
    metadata.set_primary_key(
        table_name='hotels',
        column_name='hotel_id'
    )
    metadata.set_primary_key(
        table_name='guests',
        column_name='guest_email'
    )
    metadata.add_alternate_keys(
        table_name='guests',
        column_names=['credit_card_number']
    )
    metadata.add_relationship(
        parent_table_name='hotels',
        child_table_name='guests',
        parent_primary_key='hotel_id',
        child_foreign_key='hotel_id'
    )

    # Assert - check updated metadata
    metadata.validate()
    hotels_metadata = metadata.tables['hotels']
    assert hotels_metadata.primary_key == 'hotel_id'
    assert hotels_metadata.columns['hotel_id']['sdtype'] == 'id'
    assert hotels_metadata.columns['hotel_id']['regex_format'] == 'HID_[0-9]{3,4}'

    guests_metadata = metadata.tables['guests']
    assert guests_metadata.primary_key == 'guest_email'
    assert guests_metadata.alternate_keys == ['credit_card_number']
    assert guests_metadata.columns['checkin_date']['sdtype'] == 'datetime'
    assert guests_metadata.columns['checkin_date']['datetime_format'] == '%d %b %Y'
    assert guests_metadata.columns['checkout_date']['sdtype'] == 'datetime'
    assert guests_metadata.columns['checkout_date']['datetime_format'] == '%d %b %Y'
    assert guests_metadata.columns['hotel_id']['sdtype'] == 'id'
    assert guests_metadata.columns['hotel_id']['regex_format'] == 'HID_[0-9]{3,4}'
    assert guests_metadata.columns['guest_email']['sdtype'] == 'email'
    assert guests_metadata.columns['guest_email']['pii'] is True
    assert guests_metadata.columns['billing_address']['sdtype'] == 'address'
    assert guests_metadata.columns['billing_address']['pii'] is True
    assert guests_metadata.columns['credit_card_number']['sdtype'] == 'credit_card_number'
    assert guests_metadata.columns['credit_card_number']['pii'] is True

    # Save and load metadata
    metadata_path = tmp_path / 'metadata.json'
    metadata.save_to_json(metadata_path)
    loaded_metadata = MultiTableMetadata.load_from_json(metadata_path)

    # Assert loaded metadata matches saved
    assert metadata.to_dict() == loaded_metadata.to_dict()

    # Fit synthesizer
    synthesizer = HMASynthesizer(metadata)
    synthesizer.validate(datasets)
    synthesizer.fit(datasets)
    synthetic_data = synthesizer.sample(scale=1)
    synthesizer.validate(synthetic_data)

    for table in metadata.tables:
        assert set(synthetic_data[table].columns) == set(datasets[table].columns)


def test_progress_bar_print(capsys):
    """Test that the progress bar prints correctly."""
    # Setup
    data, metadata = download_demo('multi_table', 'got_families')
    hmasynthesizer = HMASynthesizer(metadata)

    key_phrases = [
        r'Preprocess Tables:',
        r'Learning relationships:',
        r"\(1/2\) Tables 'characters' and 'character_families' \('character_id'\):",
        r"\(2/2\) Tables 'families' and 'character_families' \('family_id'\):"
    ]

    # Run
    hmasynthesizer.fit(data)
    hmasynthesizer.sample(0.5)

    captured = capsys.readouterr()

    # Assert
    for pattern in key_phrases:
        match = re.search(pattern, captured.out + captured.err)
        assert match is not None
