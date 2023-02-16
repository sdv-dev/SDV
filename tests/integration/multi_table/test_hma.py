import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pkg_resources
import pytest
from faker import Faker

from sdv.datasets.demo import download_demo
from sdv.metadata.multi_table import MultiTableMetadata
from sdv.multi_table import HMASynthesizer
from tests.integration.single_table.custom_constraints import MyConstraint


def test_hma(tmpdir):
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


def test_hma_reset_sampling(tmpdir):
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
    parent_data = pd.DataFrame({
        'primary_key': [1000, 1001, 1002],
        'numerical_col': [2, 3, 4],
        'categorical_col': ['a', 'b', 'a'],
    })
    child_data = pd.DataFrame({
        'user_id': [1000, 1001, 1000],
        'id': [1, 2, 3],
        'random': ['a', 'b', 'c']
    })

    metadata = MultiTableMetadata()
    metadata.detect_table_from_dataframe('parent', parent_data)
    metadata.detect_table_from_dataframe('child', child_data)
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
    parent_data, child_data, metadata = get_custom_constraint_data_and_metadata()
    synthesizer = HMASynthesizer(metadata)
    constraint = {
        'table_name': 'parent',
        'constraint_class': 'MyConstraint',
        'constraint_parameters': {
            'column_names': ['numerical_col']
        }
    }
    synthesizer.add_custom_constraint_class('parent', MyConstraint, 'MyConstraint')

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


def test_hma_custom_constraint_loaded_from_file():
    """Test an example of using a custom constraint loaded from a file."""
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
        'parent',
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
    metadata.detect_table_from_dataframe(table_name='child_table', data=child_table)

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


def test_save_and_load():
    """Test saving and loading a multi-table synthesizer."""
    # Setup
    _, _, metadata = get_custom_constraint_data_and_metadata()
    synthesizer = HMASynthesizer(metadata)
    temp_dir = TemporaryDirectory()
    model_path = Path(temp_dir.name) / 'synthesizer.pkl'

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

    metadata.update_column('sessions', 'session_id', sdtype='text')
    metadata.update_column('games', 'session_id', sdtype='text')
    metadata.update_column('games', 'game_id', sdtype='text')
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
