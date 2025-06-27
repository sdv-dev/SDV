import datetime
import importlib.metadata
import logging
import math
import re
import warnings
from unittest.mock import Mock

import faker
import numpy as np
import pandas as pd
import pytest
from faker import Faker
from rdt.transformers import FloatFormatter
from sdmetrics.reports.multi_table import DiagnosticReport

from sdv import version
from sdv.cag import FixedCombinations, Inequality
from sdv.cag._errors import ConstraintNotMetError
from sdv.datasets.demo import download_demo
from sdv.datasets.local import load_csvs
from sdv.errors import InvalidDataError, SamplingError, SynthesizerInputError, VersionError
from sdv.evaluation.multi_table import evaluate_quality, get_column_pair_plot, get_column_plot
from sdv.metadata import MultiTableMetadata
from sdv.metadata.metadata import Metadata
from sdv.multi_table import HMASynthesizer
from tests.integration.single_table.custom_constraints import MyConstraint
from tests.utils import catch_sdv_logs


class TestHMASynthesizer:
    def test_hma(self):
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
        assert set(normal_sample) == {'characters', 'character_families', 'families'}
        assert set(increased_sample) == {'characters', 'character_families', 'families'}
        for table_name, table in normal_sample.items():
            assert set(table.columns) == set(data[table_name])

        for normal_table, increased_table in zip(normal_sample.values(), increased_sample.values()):
            assert increased_table.size > normal_table.size

    def test_hma_metadata(self):
        """End to end integration tests with ``HMASynthesizer``.

        The test consist on loading the demo data, convert the old metadata to the new format
        and then fit a ``HMASynthesizer``. After fitting two samples are being generated, one with
        a 0.5 scale and one with 1.5 scale.
        """
        # Setup
        data, multi_metadata = download_demo('multi_table', 'got_families')
        metadata = Metadata.load_from_dict(multi_metadata.to_dict())
        hmasynthesizer = HMASynthesizer(metadata)

        # Run
        hmasynthesizer.fit(data)
        normal_sample = hmasynthesizer.sample(0.5)
        increased_sample = hmasynthesizer.sample(1.5)

        # Assert
        assert set(normal_sample) == {'characters', 'character_families', 'families'}
        assert set(increased_sample) == {'characters', 'character_families', 'families'}
        for table_name, table in normal_sample.items():
            assert set(table.columns) == set(data[table_name])

        for normal_table, increased_table in zip(normal_sample.values(), increased_sample.values()):
            assert increased_table.size > normal_table.size

    def test_hma_reset_sampling(self):
        """End to end integration test that uses ``reset_sampling``.

        This test uses ``reset_sampling`` to ensure that the model will generate the same data
        as the first sample after this method has been called.
        """
        # Setup
        faker = Faker()
        data, metadata = download_demo('multi_table', 'got_families')
        metadata.add_column(
            'ssn',
            'characters',
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

    def test_get_info(self):
        """Test the correct dictionary is returned.

        Check the return dictionary is valid both before and after fitting the synthesizer.
        """
        # Setup
        data = {'tab': pd.DataFrame({'col': [1, 2, 3]})}
        today = datetime.datetime.today().strftime('%Y-%m-%d')
        metadata = Metadata()
        metadata.add_table('tab')
        metadata.add_column('col', 'tab', sdtype='numerical')
        synthesizer = HMASynthesizer(metadata)

        # Run
        info = synthesizer.get_info()

        # Assert
        assert info == {
            'class_name': 'HMASynthesizer',
            'creation_date': today,
            'is_fit': False,
            'last_fit_date': None,
            'fitted_sdv_version': None,
        }

        # Run
        synthesizer.fit(data)
        info = synthesizer.get_info()

        # Assert
        version = importlib.metadata.version('sdv')
        assert info == {
            'class_name': 'HMASynthesizer',
            'creation_date': today,
            'is_fit': True,
            'last_fit_date': today,
            'fitted_sdv_version': version,
        }

    def test_hma_set_table_parameters(self):
        """Test the ``set_table_parameters``.

        Validate that the ``set_table_parameters`` sets new parameters to the synthesizers.
        """
        # Setup
        _data, metadata = download_demo('multi_table', 'got_families')
        hmasynthesizer = HMASynthesizer(metadata)

        # Run
        hmasynthesizer.set_table_parameters('characters', {'default_distribution': 'gamma'})
        hmasynthesizer.set_table_parameters('families', {'default_distribution': 'uniform'})
        hmasynthesizer.set_table_parameters('character_families', {'default_distribution': 'norm'})

        # Assert
        character_params = hmasynthesizer.get_table_parameters('characters')
        assert character_params['synthesizer_name'] == 'GaussianCopulaSynthesizer'
        assert character_params['synthesizer_parameters'] == {
            'default_distribution': 'gamma',
            'enforce_min_max_values': True,
            'enforce_rounding': True,
            'locales': ['en_US'],
            'numerical_distributions': {},
        }
        families_params = hmasynthesizer.get_table_parameters('families')
        assert families_params['synthesizer_name'] == 'GaussianCopulaSynthesizer'
        assert families_params['synthesizer_parameters'] == {
            'default_distribution': 'uniform',
            'enforce_min_max_values': True,
            'enforce_rounding': True,
            'locales': ['en_US'],
            'numerical_distributions': {},
        }
        char_families_params = hmasynthesizer.get_table_parameters('character_families')
        assert char_families_params['synthesizer_name'] == 'GaussianCopulaSynthesizer'
        assert char_families_params['synthesizer_parameters'] == {
            'default_distribution': 'norm',
            'enforce_min_max_values': True,
            'enforce_rounding': True,
            'locales': ['en_US'],
            'numerical_distributions': {},
        }

        assert hmasynthesizer._table_synthesizers['characters'].default_distribution == 'gamma'
        assert hmasynthesizer._table_synthesizers['families'].default_distribution == 'uniform'
        assert (
            hmasynthesizer._table_synthesizers['character_families'].default_distribution == 'norm'
        )

    def get_custom_constraint_data_and_metadata(self):
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

        metadata = Metadata()
        metadata.detect_table_from_dataframe('parent', parent_data)
        metadata.update_column('primary_key', 'parent', sdtype='id')
        metadata.detect_table_from_dataframe('child', child_data)
        metadata.update_column('user_id', 'child', sdtype='id')
        metadata.update_column('id', 'child', sdtype='id')
        metadata.set_primary_key('primary_key', 'parent')
        metadata.set_primary_key('id', 'child')
        metadata.add_relationship(
            parent_primary_key='primary_key',
            parent_table_name='parent',
            child_foreign_key='user_id',
            child_table_name='child',
        )

        return parent_data, child_data, metadata

    def test_hma_custom_constraint(self):
        """Test an example of using a custom constraint."""
        # Setup
        parent_data, child_data, metadata = self.get_custom_constraint_data_and_metadata()
        synthesizer = HMASynthesizer(metadata)
        constraint = MyConstraint(column_names=['numerical_col'], table_name='parent')

        # Run
        synthesizer.add_constraints([constraint])
        processed_data = synthesizer.preprocess({'parent': parent_data, 'child': child_data})

        # Assert Processed Data
        np.testing.assert_equal(
            processed_data['parent']['numerical_col'].array,
            (parent_data['numerical_col'] ** 2.0).array,
        )

        # Run - Fit the model
        synthesizer.fit_processed_data(processed_data)

        # Run - sample
        sampled = synthesizer.sample(10)
        assert all(sampled['parent']['numerical_col'] > 1)

    def test_hma_custom_constraint_2_tables(self):
        """Test an example of using a custom constraint.

        Check that the same custom constraint can be applied to two different tables and columns.
        """
        # Setup
        parent_data, child_data, metadata = self.get_custom_constraint_data_and_metadata()
        synthesizer = HMASynthesizer(metadata)

        constraint_parent = MyConstraint(column_names=['numerical_col'], table_name='parent')
        constraint_child = MyConstraint(column_names=['numerical_col_2'], table_name='child')

        # Run
        synthesizer.add_constraints([constraint_parent, constraint_child])
        processed_data = synthesizer.preprocess({'parent': parent_data, 'child': child_data})

        # Assert Processed Data
        np.testing.assert_equal(
            processed_data['parent']['numerical_col'].array,
            (parent_data['numerical_col'] ** 2.0).array,
        )
        np.testing.assert_equal(
            processed_data['child']['numerical_col_2'].array,
            (child_data['numerical_col_2'] ** 2.0).array,
        )

        # Run - Fit the model
        synthesizer.fit_processed_data(processed_data)

        # Run - sample
        sampled = synthesizer.sample(10)
        assert all(sampled['parent']['numerical_col'] > 1)
        assert all(sampled['child']['numerical_col_2'] > 1)
        assert not all(sampled['child']['numerical_col'] > 1)

    def test_hma_with_inequality_constraint(self):
        """Test that when new columns are created by the constraint this still works."""
        # Setup
        parent_table = pd.DataFrame(
            data={'id': [1, 2, 3, 4, 5], 'column': [1.2, 2.1, 2.2, 2.1, 1.4]}
        )

        child_table = pd.DataFrame(
            data={
                'id': [1, 2, 3, 4, 5],
                'parent_id': [1, 1, 3, 2, 1],
                'low_column': [1, 3, 3, 1, 2],
                'high_column': [2, 4, 5, 2, 4],
            }
        )

        data = {'parent_table': parent_table, 'child_table': child_table}

        metadata = Metadata()
        metadata.detect_table_from_dataframe(table_name='parent_table', data=parent_table)
        metadata.update_column('id', 'parent_table', sdtype='id')
        metadata.detect_table_from_dataframe(table_name='child_table', data=child_table)
        metadata.update_column('id', 'child_table', sdtype='id')
        metadata.update_column('parent_id', 'child_table', sdtype='id')

        metadata.set_primary_key(table_name='parent_table', column_name='id')
        metadata.set_primary_key(table_name='child_table', column_name='id')

        metadata.add_relationship(
            parent_table_name='parent_table',
            child_table_name='child_table',
            parent_primary_key='id',
            child_foreign_key='parent_id',
        )
        constraint = Inequality(
            low_column_name='low_column',
            high_column_name='high_column',
            table_name='child_table',
            strict_boundaries=False,
        )
        synthesizer = HMASynthesizer(metadata)

        # Run
        synthesizer.add_constraints(constraints=[constraint])
        synthesizer.fit(data)
        sampled = synthesizer.sample(10)

        # Assert
        assert all(sampled['child_table']['low_column'] < sampled['child_table']['high_column'])

    def test_fit_processed_multiple_calls(self):
        """Test that ``fit_processed_data`` does not modify input data."""
        # Setup
        parent_data, child_data, metadata = self.get_custom_constraint_data_and_metadata()
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

    def test_save_and_load(self, tmp_path):
        """Test saving and loading a multi-table synthesizer."""
        # Setup
        _, _, metadata = self.get_custom_constraint_data_and_metadata()
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

    def test_hma_primary_key_and_foreign_key_only(self):
        """Test that ``HMASynthesizer`` can handle tables with primary and foreign keys only."""
        # Setup
        users = pd.DataFrame({'user_id': [1, 2, 3], 'user_name': ['John', 'Doe', 'Johanna']})
        sessions = pd.DataFrame({'session_id': ['a', 'b', 'c'], 'clicks': [10, 20, 30]})
        games = pd.DataFrame({
            'game_id': ['a1', 'b2', 'c3'],
            'session_id': ['a', 'b', 'c'],
            'user_id': [1, 2, 3],
        })

        data = {'users': users, 'sessions': sessions, 'games': games}

        metadata = Metadata()
        for table_name, table in data.items():
            metadata.detect_table_from_dataframe(table_name, table)

        metadata.update_column('user_id', 'users', sdtype='id')
        metadata.update_column('session_id', 'sessions', sdtype='id')
        metadata.update_column('game_id', 'games', sdtype='id')
        metadata.update_column('session_id', 'games', sdtype='id')
        metadata.update_column('user_id', 'games', sdtype='id')
        metadata.set_primary_key('user_id', 'users')
        metadata.set_primary_key('session_id', 'sessions')
        metadata.set_primary_key('game_id', 'games')
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

    def test_synthesize_multiple_tables_using_hma(self, tmp_path):
        """End to end test for multiple tables using ``HMASynthesizer``.

        The following functionalities are being tested:
            * Create, fit and sample from model
            * Anonymization
            * Evaluating synthetic data
            * Saving, loading and sampling from the loaded model
            * Using a custom configuration for the ``HMASynthesizer``
        """
        # Loading the demo data
        real_data, metadata = download_demo(modality='multi_table', dataset_name='fake_hotels')

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
        quality_report = evaluate_quality(real_data, synthetic_data, metadata, verbose=False)

        column_plot = get_column_plot(
            real_data=real_data,
            synthetic_data=synthetic_data,
            column_name='has_rewards',
            table_name='guests',
            metadata=metadata,
        )

        column_pair_plot = get_column_pair_plot(
            real_data=real_data,
            synthetic_data=synthetic_data,
            column_names=['room_rate', 'room_type'],
            table_name='guests',
            metadata=metadata,
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
        custom_synthesizer = HMASynthesizer(metadata)

        custom_synthesizer.set_table_parameters(
            table_name='hotels',
            table_parameters={
                'default_distribution': 'truncnorm',
            },
        )

        custom_synthesizer.fit(real_data)
        learned_distributions = custom_synthesizer.get_learned_distributions(table_name='hotels')

        # Assert
        assert list(learned_distributions['rating']['learned_parameters']) == [
            'a',
            'b',
            'loc',
            'scale',
        ]
        assert learned_distributions['rating']['distribution'] == 'truncnorm'

    def test_use_own_data_using_hma(self, tmp_path):
        """End to end test for loading and preparing data.

        Tests loading data from CSVs, auto-detecting metadata from the dataframes, updating the
        metadata with correct sdtypes, and finally fitting the synthesizer.
        """
        # Setup
        data_folder = tmp_path / 'datasets'
        download_demo(
            modality='multi_table', dataset_name='fake_hotels', output_folder_name=data_folder
        )

        # Run - load CSVs
        datasets = load_csvs(data_folder)

        # Assert - loaded CSVs correctly
        assert datasets.keys() == {'guests', 'hotels'}

        # Metadata
        metadata = Metadata()

        metadata.detect_table_from_dataframe(table_name='guests', data=datasets['guests'])
        metadata.detect_table_from_dataframe(table_name='hotels', data=datasets['hotels'])

        # Assert - detected metadata correctly
        for table in metadata.tables:
            assert metadata.tables[table].columns.keys() == set(datasets[table].columns)

        # Update metadata
        metadata.update_column(
            table_name='guests',
            column_name='checkin_date',
            sdtype='datetime',
            datetime_format='%d %b %Y',
        )
        metadata.update_column(
            table_name='guests',
            column_name='checkout_date',
            sdtype='datetime',
            datetime_format='%d %b %Y',
        )
        metadata.update_column(
            table_name='hotels', column_name='hotel_id', sdtype='id', regex_format='HID_[0-9]{3,4}'
        )
        metadata.update_column(
            table_name='guests', column_name='hotel_id', sdtype='id', regex_format='HID_[0-9]{3,4}'
        )
        metadata.update_column(
            table_name='hotels',
            column_name='city',
            sdtype='categorical',
        )
        metadata.update_column(
            table_name='hotels',
            column_name='state',
            sdtype='categorical',
        )
        metadata.update_column(
            table_name='hotels',
            column_name='classification',
            sdtype='categorical',
        )
        metadata.update_column(
            table_name='guests', column_name='guest_email', sdtype='email', pii=True
        )
        metadata.update_column(
            table_name='guests', column_name='billing_address', sdtype='address', pii=True
        )
        metadata.update_column(
            table_name='guests',
            column_name='credit_card_number',
            sdtype='credit_card_number',
            pii=True,
        )
        metadata.set_primary_key(table_name='hotels', column_name='hotel_id')
        metadata.set_primary_key(table_name='guests', column_name='guest_email')
        metadata.add_alternate_keys(table_name='guests', column_names=['credit_card_number'])
        metadata.add_relationship(
            parent_table_name='hotels',
            child_table_name='guests',
            parent_primary_key='hotel_id',
            child_foreign_key='hotel_id',
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
        loaded_metadata = Metadata.load_from_json(metadata_path)

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

    def test_progress_bar_print(self, capsys):
        """Test that the progress bar prints correctly."""
        # Setup
        data, metadata = download_demo('multi_table', 'got_families')
        hmasynthesizer = HMASynthesizer(metadata)

        key_phrases = [
            r'Preprocess Tables:',
            r'Learning relationships:',
            r"\(1/2\) Tables 'characters' and 'character_families' \('character_id'\):",
            r"\(2/2\) Tables 'families' and 'character_families' \('family_id'\):",
        ]

        # Run
        hmasynthesizer.fit(data)
        hmasynthesizer.sample(0.5)

        captured = capsys.readouterr()

        # Assert
        for constraint in key_phrases:
            match = re.search(constraint, captured.out + captured.err)
            assert match is not None

    def test_warning_message_too_many_cols(self, capsys):
        """Test that a warning appears if there are more than 1000 expected columns"""
        # Setup
        (_, metadata) = download_demo(modality='multi_table', dataset_name='NBA_v1')

        key_phrases = [
            r'PerformanceAlert:',
            r'large number of columns.',
            r'please visit datacebo.com and reach out to us for enterprise solutions.',
        ]

        # Run
        HMASynthesizer(metadata)

        captured = capsys.readouterr()

        # Assert
        for constraint in key_phrases:
            match = re.search(constraint, captured.out + captured.err)
            assert match is not None
        (_, small_metadata) = download_demo(modality='multi_table', dataset_name='trains_v1')

        # Run
        HMASynthesizer(small_metadata)

        captured = capsys.readouterr()

        # Assert that small amount of columns don't trigger the message
        for constraint in key_phrases:
            match = re.search(constraint, captured.out + captured.err)
            assert match is None

    def test_hma_three_linear_nodes(self):
        """Test it works on a simple 'grandparent-parent-child' dataset."""
        # Setup
        grandparent = pd.DataFrame(
            data={'grandparent_ID': [0, 1, 2, 3, 4], 'data': ['0', '1', '2', '3', '4']}
        )
        parent = pd.DataFrame(
            data={
                'parent_ID': ['a', 'b', 'c', 'd', 'e'],
                'grandparent_ID': [0, 0, 1, 1, 3],
                'data': [True, False, False, False, True],
            }
        )
        child = pd.DataFrame(
            data={
                'child_ID': ['00', '01', '02', '03', '04'],
                'parent_ID': ['b', 'b', 'a', 'e', 'e'],
                'data': ['Yes', 'Yes', 'Maybe', 'No', 'No'],
            }
        )
        data = {'grandparent': grandparent, 'parent': parent, 'child': child}
        metadata = Metadata.load_from_dict({
            'tables': {
                'grandparent': {
                    'primary_key': 'grandparent_ID',
                    'columns': {
                        'grandparent_ID': {'sdtype': 'id'},
                        'data': {'sdtype': 'categorical'},
                    },
                },
                'parent': {
                    'primary_key': 'parent_ID',
                    'columns': {
                        'parent_ID': {'sdtype': 'id'},
                        'grandparent_ID': {'sdtype': 'id'},
                        'data': {'sdtype': 'categorical'},
                    },
                },
                'child': {
                    'primary_key': 'child_ID',
                    'columns': {
                        'child_ID': {'sdtype': 'id'},
                        'parent_ID': {'sdtype': 'id'},
                        'data': {'sdtype': 'categorical'},
                    },
                },
            },
            'relationships': [
                {
                    'parent_table_name': 'grandparent',
                    'parent_primary_key': 'grandparent_ID',
                    'child_table_name': 'parent',
                    'child_foreign_key': 'grandparent_ID',
                },
                {
                    'parent_table_name': 'parent',
                    'parent_primary_key': 'parent_ID',
                    'child_table_name': 'child',
                    'child_foreign_key': 'parent_ID',
                },
            ],
        })
        synthesizer = HMASynthesizer(metadata)

        # Run
        synthesizer.fit(data)
        samples = synthesizer.sample(scale=1)

        # Assert tables are the same
        assert set(samples) == set(data)

        # Assert columns are the same
        for table_name, table in samples.items():
            assert set(table.columns) == set(data[table_name].columns)

        # Assert data values all exist in the original tables
        for table_name, table in samples.items():
            assert table['data'].isin(data[table_name]['data']).all()

    def test_hma_one_parent_two_children(self):
        """Test it works on a simple 'child-parent-child' dataset."""
        # Setup
        parent = pd.DataFrame(
            data={'parent_ID': [0, 1, 2, 3, 4], 'data': ['0', '1', '2', '3', '4']}
        )
        child1 = pd.DataFrame(
            data={
                'child_ID': ['a', 'b', 'c', 'd', 'e'],
                'parent_ID': [0, 0, 1, 1, 3],
                'data': [True, False, False, False, True],
            }
        )
        child2 = pd.DataFrame(
            data={
                'child_ID': ['00', '01', '02', '03', '04'],
                'parent_ID': [0, 1, 2, 3, 4],
                'data': ['Yes', 'Yes', 'Maybe', 'No', 'No'],
            }
        )
        data = {'parent': parent, 'child1': child1, 'child2': child2}
        metadata = Metadata.load_from_dict({
            'tables': {
                'parent': {
                    'primary_key': 'parent_ID',
                    'columns': {'parent_ID': {'sdtype': 'id'}, 'data': {'sdtype': 'categorical'}},
                },
                'child1': {
                    'primary_key': 'child_ID',
                    'columns': {
                        'child_ID': {'sdtype': 'id'},
                        'parent_ID': {'sdtype': 'id'},
                        'data': {'sdtype': 'categorical'},
                    },
                },
                'child2': {
                    'primary_key': 'child_ID',
                    'columns': {
                        'child_ID': {'sdtype': 'id'},
                        'parent_ID': {'sdtype': 'id'},
                        'data': {'sdtype': 'categorical'},
                    },
                },
            },
            'relationships': [
                {
                    'parent_table_name': 'parent',
                    'parent_primary_key': 'parent_ID',
                    'child_table_name': 'child1',
                    'child_foreign_key': 'parent_ID',
                },
                {
                    'parent_table_name': 'parent',
                    'parent_primary_key': 'parent_ID',
                    'child_table_name': 'child2',
                    'child_foreign_key': 'parent_ID',
                },
            ],
        })
        synthesizer = HMASynthesizer(metadata)

        # Run
        synthesizer.fit(data)
        samples = synthesizer.sample(scale=1)

        # Assert tables are the same
        assert set(samples) == set(data)

        # Assert columns are the same
        for table_name, table in samples.items():
            assert set(table.columns) == set(data[table_name].columns)

        # Assert data values all exist in the original tables
        for table_name, table in samples.items():
            assert table['data'].isin(data[table_name]['data']).all()

    def test_hma_two_parents_one_child(self):
        """Test it works on a simple 'parent-child-parent' dataset."""
        # Setup
        child = pd.DataFrame(
            data={
                'child_ID': ['a', 'b', 'c', 'd', 'e'],
                'parent_ID1': [0, 1, 2, 3, 3],
                'parent_ID2': [0, 1, 2, 3, 4],
                'data': ['0', '1', '2', '3', '4'],
            }
        )
        parent1 = pd.DataFrame(
            data={'parent_ID1': [0, 1, 2, 3, 4], 'data': [True, False, False, False, True]}
        )
        parent2 = pd.DataFrame(
            data={'parent_ID2': [0, 1, 2, 3, 4], 'data': ['Yes', 'Yes', 'Maybe', 'No', 'No']}
        )
        data = {'parent1': parent1, 'child': child, 'parent2': parent2}
        metadata = Metadata.load_from_dict({
            'tables': {
                'parent1': {
                    'primary_key': 'parent_ID1',
                    'columns': {'parent_ID1': {'sdtype': 'id'}, 'data': {'sdtype': 'categorical'}},
                },
                'parent2': {
                    'primary_key': 'parent_ID2',
                    'columns': {'parent_ID2': {'sdtype': 'id'}, 'data': {'sdtype': 'categorical'}},
                },
                'child': {
                    'primary_key': 'child_ID',
                    'columns': {
                        'child_ID': {'sdtype': 'id'},
                        'parent_ID1': {'sdtype': 'id'},
                        'parent_ID2': {'sdtype': 'id'},
                        'data': {'sdtype': 'categorical'},
                    },
                },
            },
            'relationships': [
                {
                    'parent_table_name': 'parent1',
                    'parent_primary_key': 'parent_ID1',
                    'child_table_name': 'child',
                    'child_foreign_key': 'parent_ID1',
                },
                {
                    'parent_table_name': 'parent2',
                    'parent_primary_key': 'parent_ID2',
                    'child_table_name': 'child',
                    'child_foreign_key': 'parent_ID2',
                },
            ],
        })
        synthesizer = HMASynthesizer(metadata)

        # Run
        synthesizer.fit(data)
        samples = synthesizer.sample(scale=1)

        # Assert tables are the same
        assert set(samples) == set(data)

        # Assert columns are the same
        for table_name, table in samples.items():
            assert set(table.columns) == set(data[table_name].columns)

        # Assert data values all exist in the original tables
        for table_name, table in samples.items():
            assert table['data'].isin(data[table_name]['data']).all()

    def test_hma_two_lineages_one_grandchild(self):
        """Test it works on a dataset where one grandchild comes from two lineages.

        Dataset has the shape:
        r1    r2
        \\    //
         c1  c2
          \\//
           gc
        """
        # Setup
        root1 = pd.DataFrame(
            data={'id': [0, 1, 2, 3, 4], 'data': [True, False, False, False, True]}
        )
        root2 = pd.DataFrame(
            data={'id': [0, 1, 2, 3, 4], 'data': [True, False, False, False, True]}
        )
        child1 = pd.DataFrame(
            data={
                'child_ID': ['a', 'b', 'c', 'd', 'e'],
                'root1_ID': [0, 1, 2, 3, 3],
                'data': [True, False, False, False, True],
            }
        )
        child2 = pd.DataFrame(
            data={
                'child_ID': ['a', 'b', 'c', 'd', 'e'],
                'root2_ID': [0, 1, 2, 3, 4],
                'data': [True, False, False, False, True],
            }
        )
        grandchild = pd.DataFrame(
            data={
                'grandchild_ID': ['a', 'b', 'c', 'd', 'e'],
                'child1_ID': ['a', 'b', 'c', 'd', 'e'],
                'child2_ID': ['a', 'b', 'c', 'd', 'e'],
                'data': [True, False, False, False, True],
            }
        )
        data = {
            'root1': root1,
            'root2': root2,
            'child1': child1,
            'child2': child2,
            'grandchild': grandchild,
        }
        metadata = Metadata.load_from_dict({
            'tables': {
                'root1': {
                    'primary_key': 'id',
                    'columns': {'id': {'sdtype': 'id'}, 'data': {'sdtype': 'categorical'}},
                },
                'root2': {
                    'primary_key': 'id',
                    'columns': {'id': {'sdtype': 'id'}, 'data': {'sdtype': 'categorical'}},
                },
                'child1': {
                    'primary_key': 'child_ID',
                    'columns': {
                        'child_ID': {'sdtype': 'id'},
                        'root1_ID': {'sdtype': 'id'},
                        'data': {'sdtype': 'categorical'},
                    },
                },
                'child2': {
                    'primary_key': 'child_ID',
                    'columns': {
                        'child_ID': {'sdtype': 'id'},
                        'root2_ID': {'sdtype': 'id'},
                        'data': {'sdtype': 'categorical'},
                    },
                },
                'grandchild': {
                    'primary_key': 'grandchild_ID',
                    'columns': {
                        'grandchild_ID': {'sdtype': 'id'},
                        'child1_ID': {'sdtype': 'id'},
                        'child2_ID': {'sdtype': 'id'},
                        'data': {'sdtype': 'categorical'},
                    },
                },
            },
            'relationships': [
                {
                    'parent_table_name': 'root1',
                    'parent_primary_key': 'id',
                    'child_table_name': 'child1',
                    'child_foreign_key': 'root1_ID',
                },
                {
                    'parent_table_name': 'root2',
                    'parent_primary_key': 'id',
                    'child_table_name': 'child2',
                    'child_foreign_key': 'root2_ID',
                },
                {
                    'parent_table_name': 'child1',
                    'parent_primary_key': 'child_ID',
                    'child_table_name': 'grandchild',
                    'child_foreign_key': 'child1_ID',
                },
                {
                    'parent_table_name': 'child2',
                    'parent_primary_key': 'child_ID',
                    'child_table_name': 'grandchild',
                    'child_foreign_key': 'child2_ID',
                },
            ],
        })
        synthesizer = HMASynthesizer(metadata)

        # Run
        synthesizer.fit(data)
        samples = synthesizer.sample(scale=1)

        # Assert tables are the same
        assert set(samples) == set(data)

        # Assert columns are the same
        for table_name, table in samples.items():
            assert set(table.columns) == set(data[table_name].columns)

        # Assert data values all exist in the original tables
        for table_name, table in samples.items():
            assert table['data'].isin(data[table_name]['data']).all()

    def test_hma_numerical_distributions(self):
        """Test it runs when 'numerical_distributions' is set (GH#1605)."""
        # Setup
        data, metadata = download_demo('multi_table', 'fake_hotels')
        synthesizer = HMASynthesizer(metadata)

        # Run
        synthesizer.set_table_parameters(
            table_name='guests',
            table_parameters={'numerical_distributions': {'amenities_fee': 'beta'}},
        )
        synthesizer.fit(data)
        samples = synthesizer.sample(scale=1)

        # Assert - check the data was actually generated
        assert data.keys() == samples.keys()
        for table_name, table in samples.items():
            assert set(data[table_name].columns) == set(table.columns)

    def test_get_learned_distributions_error_msg(self):
        """Ensure the error message is correct when calling ``get_learned_distributions``."""
        # Setup
        data, metadata = download_demo(modality='multi_table', dataset_name='fake_hotels')
        synth = HMASynthesizer(metadata)

        # Run
        synth.fit(data)

        # Assert
        error_msg = re.escape(
            "Learned distributions are not available for the 'guests' table. "
            'Please choose a table that does not have any parents.'
        )
        with pytest.raises(SynthesizerInputError, match=error_msg):
            synth.get_learned_distributions(table_name='guests')

    def test__get_likelihoods(self):
        """Test ``_get_likelihoods`` generates likelihoods for parents."""
        # Setup
        data, metadata = download_demo('multi_table', 'got_families')
        hmasynthesizer = HMASynthesizer(metadata)
        hmasynthesizer.fit(data)

        sampled_data = {}
        sampled_data['characters'] = hmasynthesizer._sample_rows(
            hmasynthesizer._table_synthesizers['characters'], len(data['characters'])
        )
        hmasynthesizer._sample_children('characters', sampled_data)

        # Run
        likelihoods = hmasynthesizer._get_likelihoods(
            sampled_data['character_families'],
            sampled_data['characters'].set_index('character_id'),
            'character_families',
            'character_id',
        )

        # Assert
        not_nan_cols = [1, 3, 6]
        nan_cols = [2, 4, 5, 7]
        assert set(likelihoods.columns) == {1, 2, 3, 4, 5, 6, 7}
        assert len(likelihoods) == len(sampled_data['character_families'])
        assert not any(likelihoods[not_nan_cols].isna().any())
        assert all(likelihoods[nan_cols].isna())

    def test__extract_parameters(self):
        """Test it when parameters are out of bounds."""
        # Setup
        parent_row = pd.Series({
            '__sessions__user_id__num_rows': 10,
            '__sessions__user_id__a': -1,
            '__sessions__user_id__b': 1000,
            '__sessions__user_id__loc': 0.5,
            '__sessions__user_id__scale': -0.25,
        })
        instance = HMASynthesizer(Metadata())
        instance.extended_columns = {
            'sessions': {
                '__sessions__user_id__num_rows': FloatFormatter(enforce_min_max_values=True),
                '__sessions__user_id__a': FloatFormatter(enforce_min_max_values=True),
                '__sessions__user_id__b': FloatFormatter(enforce_min_max_values=True),
                '__sessions__user_id__loc': FloatFormatter(enforce_min_max_values=True),
                '__sessions__user_id__scale': FloatFormatter(enforce_min_max_values=True),
            }
        }
        for col, float_formatter in instance.extended_columns['sessions'].items():
            float_formatter.fit(pd.DataFrame({col: [0.0, 100.0]}), col)

        instance._max_child_rows = {'__sessions__user_id__num_rows': 10}

        # Run
        result = instance._extract_parameters(parent_row, 'sessions', 'user_id')

        # Assert
        expected_result = {'a': 0.0, 'b': 100.0, 'loc': 0.5, 'num_rows': 10.0, 'scale': 0.0}
        assert result == expected_result

    def test__recreate_child_synthesizer_with_default_parameters(self):
        """Test HMA when sampled parameters invalid."""
        # Setup
        prefix = '__sessions__user_id__'
        parent_row = pd.Series({
            f'{prefix}num_rows': 10,
            f'{prefix}univariates__brand__a': 100,
            f'{prefix}univariates__brand__b': 10,
            f'{prefix}univariates__brand__loc': 0.5,
            f'{prefix}univariates__brand__scale': -0.25,
        })
        metadata = Metadata.load_from_dict({
            'tables': {
                'users': {'columns': {'user_id': {'sdtype': 'id'}}, 'primary_key': 'user_id'},
                'sessions': {
                    'columns': {'user_id': {'sdtype': 'id'}, 'brand': {'sdtype': 'categorical'}}
                },
            },
            'relationships': [
                {
                    'parent_table_name': 'users',
                    'child_table_name': 'sessions',
                    'parent_primary_key': 'user_id',
                    'child_foreign_key': 'user_id',
                }
            ],
        })
        instance = HMASynthesizer(metadata)
        instance.set_table_parameters('sessions', {'default_distribution': 'truncnorm'})
        instance._max_child_rows = {f'{prefix}num_rows': 10}
        instance.extended_columns = {
            'sessions': {
                f'{prefix}num_rows': FloatFormatter(enforce_min_max_values=True),
                f'{prefix}univariates__brand__a': FloatFormatter(enforce_min_max_values=True),
                f'{prefix}univariates__brand__b': FloatFormatter(enforce_min_max_values=True),
                f'{prefix}univariates__brand__loc': FloatFormatter(enforce_min_max_values=True),
                f'{prefix}univariates__brand__scale': FloatFormatter(enforce_min_max_values=True),
            }
        }
        for col, float_formatter in instance.extended_columns['sessions'].items():
            float_formatter.fit(pd.DataFrame({col: [0.0, 100.0]}), col)

        instance._default_parameters = {
            'sessions': {
                'univariates__brand__a': 5,
                'univariates__brand__b': 84,
                'univariates__brand__loc': 1,
                'univariates__brand__scale': 1,
            }
        }

        # Run
        child_synthesizer = instance._recreate_child_synthesizer('sessions', 'users', parent_row)

        # Assert
        expected_result = {
            'univariates__brand__a': 5,
            'univariates__brand__b': 84,
            'univariates__brand__loc': 1,
            'univariates__brand__scale': 1,
            'num_rows': 10,
        }
        assert child_synthesizer._get_parameters() == expected_result

    def test_metadata_updated_no_warning(self, tmp_path):
        """Test scenario where no warning about metadata should be raised.

        Run 1 - The medata is load from our demo datasets.
        Run 2 - The metadata uses ``detect_from_dataframes`` but is saved to a file
                before defining the syntheiszer.
        Run 3 - The metadata is updated with a new column after the synthesizer
                initialization, but is saved to a file before fitting.
        """
        # Setup
        data, multi_metadata = download_demo('multi_table', 'got_families')
        metadata = Metadata.load_from_dict(multi_metadata.to_dict())

        # Run 1
        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.filterwarnings(
                'ignore',
                message=".*The 'SingleTableMetadata' is deprecated.*",
                category=DeprecationWarning,
            )
            warnings.simplefilter('always')
            instance = HMASynthesizer(metadata)
            instance.fit(data)

        # Assert
        assert len(captured_warnings) == 0

        # Run 2
        metadata_detect = Metadata.detect_from_dataframes(data)

        metadata_detect.relationships = metadata.relationships
        for table_name, table_metadata in metadata.tables.items():
            metadata_detect.tables[table_name].columns = table_metadata.columns
            metadata_detect.tables[table_name].primary_key = table_metadata.primary_key

        file_name = tmp_path / 'multitable_1.json'
        metadata_detect.save_to_json(file_name)
        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter('always')
            instance = HMASynthesizer(metadata_detect)
            instance.fit(data)

        # Assert
        assert len(captured_warnings) == 0

        # Run 3
        instance = HMASynthesizer(metadata_detect)
        metadata_detect.update_column(
            table_name='characters', column_name='age', sdtype='categorical'
        )
        file_name = tmp_path / 'multitable_2.json'
        metadata_detect.save_to_json(file_name)
        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter('always')
            instance.fit(data)

        # Assert
        assert len(captured_warnings) == 0

    def test_metadata_updated_warning_detect(self):
        """Test that using ``detect_from_dataframes`` without saving the metadata raise a warning.

        The warning is expected to be raised only once during synthesizer initialization. It should
        not be raised again when calling ``fit``.
        """
        # Setup
        data, metadata = download_demo('multi_table', 'got_families')
        metadata_detect = Metadata.detect_from_dataframes(data)

        metadata_detect.relationships = metadata.relationships
        for table_name, table_metadata in metadata.tables.items():
            metadata_detect.tables[table_name].columns = table_metadata.columns
            metadata_detect.tables[table_name].primary_key = table_metadata.primary_key

        expected_message = re.escape(
            "We strongly recommend saving the metadata using 'save_to_json' for replicability"
            ' in future SDV versions.'
        )

        # Run
        with pytest.warns(UserWarning, match=expected_message) as record:
            instance = HMASynthesizer(metadata_detect)
            instance.fit(data)

        # Assert
        assert len(record) == 1

    def test_null_foreign_keys(self):
        """Test that the synthesizer does not crash when there are null foreign keys."""
        # Setup
        metadata = Metadata()
        metadata.add_table('parent_table1')
        metadata.add_column('id', 'parent_table1', sdtype='id')
        metadata.set_primary_key('id', 'parent_table1')

        metadata.add_table('parent_table2')
        metadata.add_column('id', 'parent_table2', sdtype='id')
        metadata.set_primary_key('id', 'parent_table2')

        metadata.add_table('child_table1')
        metadata.add_column('id', 'child_table1', sdtype='id')
        metadata.set_primary_key('id', 'child_table1')
        metadata.add_column('fk1', 'child_table1', sdtype='id')
        metadata.add_column('fk2', 'child_table1', sdtype='id')

        metadata.add_table('child_table2')
        metadata.add_column('id', 'child_table2', sdtype='id')
        metadata.set_primary_key('id', 'child_table2')
        metadata.add_column('fk1', 'child_table2', sdtype='id')
        metadata.add_column('fk2', 'child_table2', sdtype='id')
        metadata.add_column('cat_type', 'child_table2', sdtype='categorical')

        metadata.add_relationship(
            parent_table_name='parent_table1',
            child_table_name='child_table1',
            parent_primary_key='id',
            child_foreign_key='fk1',
        )

        metadata.add_relationship(
            parent_table_name='parent_table2',
            child_table_name='child_table1',
            parent_primary_key='id',
            child_foreign_key='fk2',
        )

        metadata.add_relationship(
            parent_table_name='parent_table1',
            child_table_name='child_table2',
            parent_primary_key='id',
            child_foreign_key='fk1',
        )

        metadata.add_relationship(
            parent_table_name='parent_table1',
            child_table_name='child_table2',
            parent_primary_key='id',
            child_foreign_key='fk2',
        )

        data = {
            'parent_table1': pd.DataFrame({'id': [1, 2, 3]}),
            'parent_table2': pd.DataFrame({'id': ['alpha', 'beta', 'gamma']}),
            'child_table1': pd.DataFrame({
                'id': [1, 2, 3],
                'fk1': pd.Series([np.nan, 2, np.nan], dtype='float64'),
                'fk2': pd.Series(['alpha', 'beta', np.nan], dtype='object'),
            }),
            'child_table2': pd.DataFrame({
                'id': [1, 2, 3],
                'fk1': [1, 2, np.nan],
                'fk2': pd.Series([1, np.nan, np.nan], dtype='float64'),
                'cat_type': pd.Series(['siamese', 'persian', 'american shorthair'], dtype='object'),
            }),
        }

        synthesizer = HMASynthesizer(metadata)

        # Run
        metadata.validate()
        metadata.validate_data(data)

        # Run
        synthesizer.fit(data)
        sampled = synthesizer.sample()

        # Assert
        assert len(sampled['parent_table1']) == 3
        assert len(sampled['parent_table2']) == 3
        assert sum(pd.isna(sampled['child_table1']['fk1'])) == 2
        assert sum(pd.isna(sampled['child_table1']['fk2'])) == 1
        assert sum(pd.isna(sampled['child_table2']['fk1'])) == 1
        assert sum(pd.isna(sampled['child_table2']['fk2'])) == 2

    def test_sampling_with_unknown_sdtype_numerical_column(self):
        """Test that if a numerical column is detected as unknown in the metadata,
        it does not fail and is handled as original detected value
        """
        # Setup
        fake = faker.Faker()

        table1 = pd.DataFrame({
            'name': [fake.name() for i in range(20)],
            'salary': np.random.randint(20_000, 250_000, 20),
            'age': np.random.randint(18, 70, 20),
            'address': [fake.address() for i in range(20)],
        })
        table2 = pd.DataFrame({
            'company': [fake.company() for i in range(20)],
            'employee_count': np.random.randint(15, 4000, 20),
            'revenue': np.random.randint(100_000, 1_000_000_000),
        })

        tables_dict = {'people': table1, 'company': table2}

        metadata = Metadata.detect_from_dataframes(tables_dict)

        # Run
        synth = HMASynthesizer(metadata)
        synth.fit(tables_dict)
        sample_data = synth.sample(1)

        # Assert
        people_sample = sample_data['people']
        company_sample = sample_data['company']

        # Since these values are inferred, windows and mac may have different int types
        # so check if it is numeric
        numeric_data = [
            people_sample['salary'],
            people_sample['age'],
            company_sample['employee_count'],
            company_sample['revenue'],
        ]
        object_data = [
            people_sample['name'].dtype,
            people_sample['address'].dtype,
            company_sample['company'].dtype,
        ]
        assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in numeric_data)
        assert all(dtype == 'object' for dtype in object_data)

    def test_large_integer_ids(self):
        """Test that HMASynthesizer can handle large integer IDs correctly GH#919."""
        # Setup
        table_1 = pd.DataFrame({
            'col_1': [1, 2, 3],
            'col_3': [7, 8, 9],
            'col_2': [4, 5, 6],
        })
        table_2 = pd.DataFrame({
            'col_A': [1, 1, 2],
            'col_B': ['d', 'e', 'f'],
            'col_C': ['g', 'h', 'i'],
        })
        metadata = Metadata.load_from_dict({
            'tables': {
                'table_1': {
                    'columns': {
                        'col_1': {'sdtype': 'id', 'regex_format': '[1-9]{17}'},
                        'col_2': {'sdtype': 'numerical'},
                        'col_3': {'sdtype': 'numerical'},
                    },
                    'primary_key': 'col_1',
                },
                'table_2': {
                    'columns': {
                        'col_A': {'sdtype': 'id', 'regex_format': '[1-9]{17}'},
                        'col_B': {'sdtype': 'categorical'},
                        'col_C': {'sdtype': 'categorical'},
                    },
                },
            },
            'relationships': [
                {
                    'parent_table_name': 'table_1',
                    'child_table_name': 'table_2',
                    'parent_primary_key': 'col_1',
                    'child_foreign_key': 'col_A',
                }
            ],
        })
        data = {
            'table_1': table_1,
            'table_2': table_2,
        }

        # Run
        synthesizer = HMASynthesizer(metadata, verbose=False)
        synthesizer.fit(data)
        synthetic_data = synthesizer.sample()

        # Assert
        # Check that IDs match the regex constraint
        for table_name, table in synthetic_data.items():
            for col in table.columns:
                if metadata.tables[table_name].columns[col].get('sdtype') == 'id':
                    values = table[col].astype(str)
                    assert all(len(str(v)) == 17 for v in values), (
                        f'ID length mismatch in {table_name}.{col}'
                    )
                    assert all(v.isdigit() for v in values), (
                        f'Non-digit characters in {table_name}.{col}'
                    )

        # Check relationships are preserved
        child_fks = set(synthetic_data['table_2']['col_A'])
        parent_pks = set(synthetic_data['table_1']['col_1'])
        assert child_fks.issubset(parent_pks), 'Foreign key constraint violated'

        # Check that the diagnostic report is 1.0
        report = DiagnosticReport()
        report.generate(data, synthetic_data, metadata.to_dict(), verbose=False)
        assert report.get_score() == 1.0

    def test_large_integer_ids_overflow(self):
        """Test that it overflows.

        When the real data primary key can fit in int64, ie has less than 19 digits,
        but the regex_format specifies data that can't fit in int64, ie over 20 digits,
        the synthetic data will raise an overflow warning and it will stay as object dtype.
        """
        # Setup
        table_1 = pd.DataFrame({
            'col_1': [1, 2, 3],
            'col_3': [7, 8, 9],
            'col_2': [4, 5, 6],
        })
        table_2 = pd.DataFrame({
            'col_A': [1, 1, 2],
            'col_B': ['d', 'e', 'f'],
            'col_C': ['g', 'h', 'i'],
        })
        metadata = Metadata.load_from_dict({
            'tables': {
                'table_1': {
                    'columns': {
                        'col_1': {'sdtype': 'id', 'regex_format': '[1-9]{21}'},
                        'col_2': {'sdtype': 'numerical'},
                        'col_3': {'sdtype': 'numerical'},
                    },
                    'primary_key': 'col_1',
                },
                'table_2': {
                    'columns': {
                        'col_A': {'sdtype': 'id', 'regex_format': '[1-9]{21}'},
                        'col_B': {'sdtype': 'categorical'},
                        'col_C': {'sdtype': 'categorical'},
                    },
                },
            },
            'relationships': [
                {
                    'parent_table_name': 'table_1',
                    'child_table_name': 'table_2',
                    'parent_primary_key': 'col_1',
                    'child_foreign_key': 'col_A',
                }
            ],
        })
        data = {
            'table_1': table_1,
            'table_2': table_2,
        }

        # Run
        synthesizer = HMASynthesizer(metadata)
        synthesizer.fit(data)
        with warnings.catch_warnings(record=True) as captured_warnings:
            synthetic_data = synthesizer.sample()

        # Assert
        # Check that IDs match the regex constraint
        for table_name, table in synthetic_data.items():
            for col in table.columns:
                if metadata.tables[table_name].columns[col].get('sdtype') == 'id':
                    values = table[col].astype(str)
                    assert all(len(str(v)) == 21 for v in values), (
                        f'ID length mismatch in {table_name}.{col}'
                    )
                    assert all(v.isdigit() for v in values), (
                        f'Non-digit characters in {table_name}.{col}'
                    )

        # Check relationships are preserved
        child_fks = set(synthetic_data['table_2']['col_A'])
        parent_pks = set(synthetic_data['table_1']['col_1'])
        assert child_fks.issubset(parent_pks), 'Foreign key constraint violated'

        # Check that a warning is raised
        assert len(captured_warnings) == 1
        assert str(captured_warnings[0].message) == (
            "The real data in 'table_1' and column 'col_1' was stored as 'int64' but the "
            'synthetic data overflowed when casting back to this type. If this is a problem, '
            'please check your input data and metadata settings.'
        )

    def test_large_integer_ids_overflow_three_tables(self):
        """Test that it overflows.

        When the real data primary key can fit in int64, ie has less than 19 digits,
        but the regex_format specifies data that can't fit in int64, ie over 20 digits,
        the synthetic data will raise an overflow warning and it will stay as object dtype.

        This should raise two warnings, one for each parent table with ids that overflow.
        """
        # Setup
        table_0 = pd.DataFrame({
            'col_0': [1, 2, 3],
        })
        table_1 = pd.DataFrame({
            'col_1': [1, 2, 3],
            'col_3': [7, 8, 9],
            'col_2': [4, 5, 6],
            'col_0': [1, 2, 2],
        })
        table_2 = pd.DataFrame({
            'col_A': [1, 2, 3],
            'col_B': ['d', 'e', 'f'],
            'col_C': ['g', 'h', 'i'],
        })
        metadata = Metadata.load_from_dict({
            'tables': {
                'table_0': {
                    'columns': {
                        'col_0': {'sdtype': 'id', 'regex_format': '[1-9]{20}'},
                    },
                    'primary_key': 'col_0',
                },
                'table_1': {
                    'columns': {
                        'col_1': {'sdtype': 'id', 'regex_format': '[1-9]{20}'},
                        'col_2': {'sdtype': 'numerical'},
                        'col_3': {'sdtype': 'numerical'},
                        'col_0': {'sdtype': 'id', 'regex_format': '[1-9]{20}'},
                    },
                    'primary_key': 'col_1',
                },
                'table_2': {
                    'columns': {
                        'col_A': {'sdtype': 'id', 'regex_format': '[1-9]{20}'},
                        'col_B': {'sdtype': 'categorical'},
                        'col_C': {'sdtype': 'categorical'},
                    },
                },
            },
            'relationships': [
                {
                    'parent_table_name': 'table_1',
                    'child_table_name': 'table_2',
                    'parent_primary_key': 'col_1',
                    'child_foreign_key': 'col_A',
                },
                {
                    'parent_table_name': 'table_0',
                    'child_table_name': 'table_1',
                    'parent_primary_key': 'col_0',
                    'child_foreign_key': 'col_0',
                },
            ],
        })
        data = {
            'table_0': table_0,
            'table_1': table_1,
            'table_2': table_2,
        }

        # Run
        synthesizer = HMASynthesizer(metadata)
        synthesizer.fit(data)
        with warnings.catch_warnings(record=True) as captured_warnings:
            synthetic_data = synthesizer.sample()

        # Assert
        # Check that IDs match the regex constraint
        for table_name, table in synthetic_data.items():
            for col in table.columns:
                if metadata.tables[table_name].columns[col].get('sdtype') == 'id':
                    values = table[col].astype(str)
                    assert all(len(str(v)) == 20 for v in values), (
                        f'ID length mismatch in {table_name}.{col}'
                    )
                    assert all(v.isdigit() for v in values), (
                        f'Non-digit characters in {table_name}.{col}'
                    )

        # Check relationships are preserved
        child_fks = set(synthetic_data['table_1']['col_0'])
        parent_pks = set(synthetic_data['table_0']['col_0'])
        assert child_fks.issubset(parent_pks), 'Foreign key constraint violated'

        child_fks = set(synthetic_data['table_2']['col_A'])
        parent_pks = set(synthetic_data['table_1']['col_1'])
        assert child_fks.issubset(parent_pks), 'Foreign key constraint violated'

        # Check that a warning is raised
        assert len(captured_warnings) == 2
        assert str(captured_warnings[0].message) == (
            "The real data in 'table_0' and column 'col_0' was stored as 'int64' but the "
            'synthetic data overflowed when casting back to this type. If this is a problem, '
            'please check your input data and metadata settings.'
        )
        assert str(captured_warnings[1].message) == (
            "The real data in 'table_1' and column 'col_1' was stored as 'int64' but the "
            'synthetic data overflowed when casting back to this type. If this is a problem, '
            'please check your input data and metadata settings.'
        )

    def test_ids_that_dont_fit_in_int64(self):
        """Test it when both real and synthetic data don't fit in int64."""
        # Setup
        table_1 = pd.DataFrame({
            'col_1': [99999999999999999990, 99999999999999999991, 99999999999999999992],  # len 20
            'col_3': [7, 8, 9],
            'col_2': [4, 5, 6],
        })
        table_2 = pd.DataFrame({
            'col_A': [99999999999999999990, 99999999999999999990, 99999999999999999991],  # len 20
            'col_B': ['d', 'e', 'f'],
            'col_C': ['g', 'h', 'i'],
        })
        metadata = Metadata.load_from_dict({
            'tables': {
                'table_1': {
                    'columns': {
                        'col_1': {'sdtype': 'id', 'regex_format': '[1-9]{20}'},
                        'col_2': {'sdtype': 'numerical'},
                        'col_3': {'sdtype': 'numerical'},
                    },
                    'primary_key': 'col_1',
                },
                'table_2': {
                    'columns': {
                        'col_A': {'sdtype': 'id', 'regex_format': '[1-9]{20}'},
                        'col_B': {'sdtype': 'categorical'},
                        'col_C': {'sdtype': 'categorical'},
                    },
                },
            },
            'relationships': [
                {
                    'parent_table_name': 'table_1',
                    'child_table_name': 'table_2',
                    'parent_primary_key': 'col_1',
                    'child_foreign_key': 'col_A',
                }
            ],
        })
        data = {
            'table_1': table_1,
            'table_2': table_2,
        }
        synthesizer = HMASynthesizer(metadata)
        synthesizer.fit(data)
        with warnings.catch_warnings(record=True) as captured_warnings:
            synthetic_data = synthesizer.sample()

        # Assert
        # Check that IDs match the regex constraint
        for table_name, table in synthetic_data.items():
            for col in table.columns:
                if metadata.tables[table_name].columns[col].get('sdtype') == 'id':
                    values = table[col].astype(str)
                    assert all(len(str(v)) == 20 for v in values), (
                        f'ID length mismatch in {table_name}.{col}'
                    )
                    assert all(v.isdigit() for v in values), (
                        f'Non-digit characters in {table_name}.{col}'
                    )

        # Check relationships are preserved
        child_fks = set(synthetic_data['table_2']['col_A'])
        parent_pks = set(synthetic_data['table_1']['col_1'])
        assert child_fks.issubset(parent_pks), 'Foreign key constraint violated'

        # No warnings should be raised
        assert len(captured_warnings) == 0

        # Check that the diagnostic report is 1.0
        report = DiagnosticReport()
        report.generate(data, synthetic_data, metadata.to_dict(), verbose=False)
        assert report.get_score() == 1.0

    def test_large_real_ids_small_synthetic_ids(self):
        """Test it when real data has more digits than synthetic data."""
        # Setup
        table_1 = pd.DataFrame({
            'col_1': [99999999999999999990, 99999999999999999991, 99999999999999999992],  # len 20
            'col_3': [7, 8, 9],
            'col_2': [4, 5, 6],
        })
        table_2 = pd.DataFrame({
            'col_A': [99999999999999999990, 99999999999999999990, 99999999999999999991],  # len 20
            'col_B': ['d', 'e', 'f'],
            'col_C': ['g', 'h', 'i'],
        })
        metadata = Metadata.load_from_dict({
            'tables': {
                'table_1': {
                    'columns': {
                        'col_1': {'sdtype': 'id', 'regex_format': '[1-9]{1}'},
                        'col_2': {'sdtype': 'numerical'},
                        'col_3': {'sdtype': 'numerical'},
                    },
                    'primary_key': 'col_1',
                },
                'table_2': {
                    'columns': {
                        'col_A': {'sdtype': 'id', 'regex_format': '[1-9]{1}'},
                        'col_B': {'sdtype': 'categorical'},
                        'col_C': {'sdtype': 'categorical'},
                    },
                },
            },
            'relationships': [
                {
                    'parent_table_name': 'table_1',
                    'child_table_name': 'table_2',
                    'parent_primary_key': 'col_1',
                    'child_foreign_key': 'col_A',
                }
            ],
        })
        data = {
            'table_1': table_1,
            'table_2': table_2,
        }
        synthesizer = HMASynthesizer(metadata)
        synthesizer.fit(data)
        with warnings.catch_warnings(record=True) as captured_warnings:
            synthetic_data = synthesizer.sample()

        # Assert
        # Check that IDs match the regex constraint
        for table_name, table in synthetic_data.items():
            for col in table.columns:
                if metadata.tables[table_name].columns[col].get('sdtype') == 'id':
                    values = table[col].astype(str)
                    assert all(len(str(v)) == 1 for v in values), (
                        f'ID length mismatch in {table_name}.{col}'
                    )
                    assert all(v.isdigit() for v in values), (
                        f'Non-digit characters in {table_name}.{col}'
                    )

        # Check relationships are preserved
        child_fks = set(synthetic_data['table_2']['col_A'])
        parent_pks = set(synthetic_data['table_1']['col_1'])
        assert child_fks.issubset(parent_pks), 'Foreign key constraint violated'

        # No warnings should be raised
        assert len(captured_warnings) == 0

        # Check that the diagnostic report is 1.0
        report = DiagnosticReport()
        report.generate(data, synthetic_data, metadata.to_dict(), verbose=False)
        assert report.get_score() == 1.0


@pytest.mark.parametrize('num_rows', [(10), (1000)])
def test_hma_0_1_child(num_rows):
    parent_table = pd.DataFrame(
        data={
            'id': list(range(num_rows)),
            'col_A': list(np.random.choice(['A', 'B', 'C', 'D', 'E'], size=num_rows)),
        }
    )
    child_table_data = {'parent_id': [], 'col_B': [], 'col_C': []}

    for i in range(num_rows):
        num_children = np.random.choice([0, 1, 10, 15], p=[0.4, 0.5, 0.05, 0.05])
        if num_children == 0:
            continue
        child_table_data['parent_id'].extend([i] * num_children)
        child_table_data['col_B'].extend([
            round(i, 2) for i in np.random.uniform(low=0, high=10, size=num_children)
        ])
        child_table_data['col_C'].extend(
            list(np.random.choice(['A', 'B', 'C', 'D', 'E'], size=num_children))
        )

    data = {'parent': parent_table, 'child': pd.DataFrame(data=child_table_data)}
    metadata = Metadata.load_from_dict({
        'tables': {
            'parent': {
                'primary_key': 'id',
                'columns': {'id': {'sdtype': 'id'}, 'col_A': {'sdtype': 'categorical'}},
            },
            'child': {
                'columns': {
                    'parent_id': {'sdtype': 'id'},
                    'col_B': {'sdtype': 'numerical'},
                    'col_C': {'sdtype': 'categorical'},
                }
            },
        },
        'relationships': [
            {
                'parent_table_name': 'parent',
                'child_table_name': 'child',
                'parent_primary_key': 'id',
                'child_foreign_key': 'parent_id',
            }
        ],
    })
    synthesizer = HMASynthesizer(metadata=metadata, verbose=False)
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample(scale=1)
    synthetic_child_df = synthetic_data['child']
    data_col_max = synthetic_child_df['col_B'].max()
    expected_constant_length = math.floor(len(synthetic_child_df) * 0.70)
    actual_constants = synthetic_child_df[synthetic_child_df['col_B'] == data_col_max]
    assert len(actual_constants) <= expected_constant_length
    assert synthetic_child_df['col_B'].max() <= synthetic_child_df['col_B'].max()
    assert synthetic_child_df['col_B'].min() >= synthetic_child_df['col_B'].min()


def test_hma_0_1_grandparent():
    grandparent = pd.DataFrame({'grandparent_id': [50, 51, 52]})
    parent = pd.DataFrame({
        'parent_id': [0, 1, 2, 3],
        'data': [1.5, 2.5, 5.9, 10.6],
        'grandparent_id': [50, 50, 50, 52],
    })
    child = pd.DataFrame({
        'child_id': [10, 11, 12],
        'parent_id': [0, 1, 2],
        'data': [1.8, 0.7, 2.5],
    })
    data = {'parent': parent, 'child': child, 'grandparent': grandparent}
    metadata_dict = {
        'tables': {
            'grandparent': {
                'primary_key': 'grandparent_id',
                'columns': {
                    'grandparent_id': {'sdtype': 'id'},
                },
            },
            'parent': {
                'primary_key': 'parent_id',
                'columns': {
                    'parent_id': {'sdtype': 'id'},
                    'data': {'sdtype': 'numerical'},
                    'grandparent_id': {'sdtype': 'id'},
                },
            },
            'child': {
                'primary_key': 'child_id',
                'columns': {
                    'child_id': {'sdtype': 'id'},
                    'parent_id': {'sdtype': 'id'},
                    'data': {'sdtype': 'numerical'},
                },
            },
        },
        'relationships': [
            {
                'parent_table_name': 'grandparent',
                'parent_primary_key': 'grandparent_id',
                'child_table_name': 'parent',
                'child_foreign_key': 'grandparent_id',
            },
            {
                'parent_table_name': 'parent',
                'parent_primary_key': 'parent_id',
                'child_table_name': 'child',
                'child_foreign_key': 'parent_id',
            },
        ],
    }
    metadata = Metadata().load_from_dict(metadata_dict)
    metadata.validate()
    metadata.validate_data(data)
    synthesizer = HMASynthesizer(metadata=metadata, verbose=False)
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample()
    child_df = synthetic_data['child']
    data_col_max = child_df['data'].max()
    data_col_min = child_df['data'].min()
    assert child_df[child_df['data'] == data_col_max].shape[0] == 2
    assert child_df[child_df['data'] == data_col_min].shape[0] == 1


parametrization = [
    ('update_column', {'table_name': 'departure', 'column_name': 'city', 'sdtype': 'categorical'}),
    ('set_primary_key', {'table_name': 'arrival', 'column_name': 'id_flight'}),
    (
        'add_column_relationship',
        {
            'table_name': 'departure',
            'relationship_type': 'address',
            'column_names': ['city', 'country'],
        },
    ),
    ('add_alternate_keys', {'table_name': 'departure', 'column_names': ['city', 'country']}),
    ('set_sequence_key', {'table_name': 'departure', 'column_name': 'city'}),
    (
        'add_column',
        {'table_name': 'departure', 'column_name': 'postal_code', 'sdtype': 'postal_code'},
    ),
]


@pytest.mark.parametrize(('method', 'kwargs'), parametrization)
def test_metadata_updated_warning(method, kwargs):
    """Test that modifying metadata without saving it raise a warning.

    The warning should be raised during synthesizer initialization.
    """
    metadata = Metadata().load_from_dict({
        'tables': {
            'departure': {
                'primary_key': 'id',
                'columns': {
                    'id': {'sdtype': 'id'},
                    'date': {'sdtype': 'datetime'},
                    'city': {'sdtype': 'city'},
                    'country': {'sdtype': 'country_code'},
                },
            },
            'arrival': {
                'columns': {
                    'id': {'sdtype': 'id'},
                    'date': {'sdtype': 'datetime'},
                    'city': {'sdtype': 'city'},
                    'country': {'sdtype': 'country'},
                    'id_flight': {'sdtype': 'id'},
                },
            },
        },
        'relationships': [
            {
                'parent_table_name': 'departure',
                'parent_primary_key': 'id',
                'child_table_name': 'arrival',
                'child_foreign_key': 'id',
            }
        ],
    })
    expected_message = re.escape(
        "We strongly recommend saving the metadata using 'save_to_json' for replicability"
        ' in future SDV versions.'
    )

    # Run
    metadata.__getattribute__(method)(**kwargs)
    with pytest.warns(UserWarning, match=expected_message):
        HMASynthesizer(metadata)

    # Assert
    assert metadata._multi_table_updated is False
    for table_name, table_metadata in metadata.tables.items():
        assert table_metadata._updated is False


def test_save_and_load_with_downgraded_version(tmp_path):
    """Test that synthesizers are raising errors if loaded on a downgraded version."""
    # Setup
    metadata = Metadata().load_from_dict({
        'tables': {
            'departure': {
                'primary_key': 'id',
                'columns': {
                    'id': {'sdtype': 'id'},
                    'date': {'sdtype': 'datetime'},
                    'city': {'sdtype': 'city'},
                    'country': {'sdtype': 'country'},
                },
            },
            'arrival': {
                'columns': {
                    'id': {'sdtype': 'id'},
                    'date': {'sdtype': 'datetime'},
                    'city': {'sdtype': 'city'},
                    'country': {'sdtype': 'country'},
                    'id_flight': {'sdtype': 'id'},
                },
            },
        },
        'relationships': [
            {
                'parent_table_name': 'departure',
                'parent_primary_key': 'id',
                'child_table_name': 'arrival',
                'child_foreign_key': 'id',
            }
        ],
    })

    instance = HMASynthesizer(metadata)
    instance._fitted = True
    instance._fitted_sdv_version = '10.0.0'
    synthesizer_path = tmp_path / 'synthesizer.pkl'
    instance.save(synthesizer_path)

    # Run and Assert
    error_msg = (
        f'You are currently on SDV version {version.community} but this '
        'synthesizer was created on version 10.0.0. '
        'Downgrading your SDV version is not supported.'
    )
    with pytest.raises(VersionError, match=error_msg):
        HMASynthesizer.load(synthesizer_path)


def test_fit_raises_version_error():
    """Test that a ``VersionError`` is being raised if the current version is newer."""
    # Setup
    metadata = Metadata().load_from_dict({
        'tables': {
            'departure': {
                'primary_key': 'id',
                'columns': {
                    'id': {'sdtype': 'id'},
                    'date': {'sdtype': 'datetime'},
                    'city': {'sdtype': 'city'},
                    'country': {'sdtype': 'country'},
                },
            },
            'arrival': {
                'columns': {
                    'id': {'sdtype': 'id'},
                    'date': {'sdtype': 'datetime'},
                    'city': {'sdtype': 'city'},
                    'country': {'sdtype': 'country'},
                    'id_flight': {'sdtype': 'id'},
                },
            },
        },
        'relationships': [
            {
                'parent_table_name': 'departure',
                'parent_primary_key': 'id',
                'child_table_name': 'arrival',
                'child_foreign_key': 'id',
            }
        ],
    })

    instance = HMASynthesizer(metadata)
    instance._fitted_sdv_version = '1.0.0'

    # Run and Assert
    expected_message = (
        f'You are currently on SDV version {version.community} but this synthesizer was created on '
        'version 1.0.0. Fitting this synthesizer again is not supported. Please create a new '
        'synthesizer.'
    )
    with pytest.raises(VersionError, match=expected_message):
        instance.fit({})


def test_hma_relationship_validity():
    """Test the quality of the HMA synthesizer GH#1834."""
    # Setup
    data, metadata = download_demo('multi_table', 'Dunur_v1')
    synthesizer = HMASynthesizer(metadata)
    report = DiagnosticReport()

    # Run
    synthesizer.fit(data)
    sample = synthesizer.sample()
    report.generate(data, sample, metadata.to_dict(), verbose=False)

    # Assert
    assert report.get_details('Relationship Validity')['Score'].mean() == 1.0


def test_hma_not_fit_raises_sampling_error():
    """Test that ``HMA`` will raise a ``SamplingError`` if it wasn't fit."""
    # Setup
    _data, metadata = download_demo('multi_table', 'Dunur_v1')
    synthesizer = HMASynthesizer(metadata)

    # Run and Assert
    error_msg = (
        'This synthesizer has not been fitted. Please fit your synthesizer first before '
        'sampling synthetic data.'
    )
    with pytest.raises(SamplingError, match=error_msg):
        synthesizer.sample(1)


def test_fit_and_sample_numerical_col_names():
    """Test fitting/sampling when column names are integers"""
    # Setup
    num_rows = 50
    num_cols = 10
    num_tables = 2
    data = {}
    for i in range(num_tables):
        values = {j: np.random.randint(0, 100, size=num_rows) for j in range(num_cols)}
        data[str(i)] = pd.DataFrame(values)

    primary_key = pd.DataFrame({1: range(num_rows)})
    primary_key_2 = pd.DataFrame({2: range(num_rows)})
    data['0'][1] = primary_key
    data['1'][1] = primary_key
    data['1'][2] = primary_key_2
    metadata = Metadata()
    metadata_dict = {'tables': {}}
    for table_idx in range(num_tables):
        metadata_dict['tables'][str(table_idx)] = {'columns': {}}
        for i in range(num_cols):
            metadata_dict['tables'][str(table_idx)]['columns'][i] = {'sdtype': 'numerical'}
    metadata_dict['tables']['0']['columns'][1] = {'sdtype': 'id'}
    metadata_dict['tables']['1']['columns'][2] = {'sdtype': 'id'}
    metadata_dict['relationships'] = [
        {
            'parent_table_name': '0',
            'parent_primary_key': 1,
            'child_table_name': '1',
            'child_foreign_key': 2,
        }
    ]
    metadata = Metadata.load_from_dict(metadata_dict)
    metadata.set_primary_key('1', '0')

    # Run
    synth = HMASynthesizer(metadata)
    synth.fit(data)
    first_sample = synth.sample()
    second_sample = synth.sample()
    assert first_sample['0'].columns.tolist() == data['0'].columns.tolist()
    assert first_sample['1'].columns.tolist() == data['1'].columns.tolist()
    assert second_sample['0'].columns.tolist() == data['0'].columns.tolist()
    assert second_sample['1'].columns.tolist() == data['1'].columns.tolist()

    # Assert
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(first_sample['0'], second_sample['0'])


def test_detect_from_dataframe_numerical_col():
    """Test that metadata detection of integer columns work."""
    # Setup
    parent_data = pd.DataFrame({
        1: [1000, 1001, 1002],
        2: [2, 3, 4],
        'categorical_col': ['a', 'b', 'a'],
    })
    child_data = pd.DataFrame({3: [1000, 1001, 1000], 4: [1, 2, 3]})
    data = {
        'parent_data': parent_data,
        'child_data': child_data,
    }
    metadata = Metadata()
    metadata.detect_table_from_dataframe('parent_data', parent_data)
    metadata.detect_table_from_dataframe('child_data', child_data)
    metadata.update_column('1', 'parent_data', sdtype='id')
    metadata.update_column('3', 'child_data', sdtype='id')
    metadata.update_column('4', 'child_data', sdtype='id')
    metadata.set_primary_key('1', 'parent_data')
    metadata.set_primary_key('4', 'child_data')
    metadata.add_relationship(
        parent_primary_key='1',
        parent_table_name='parent_data',
        child_foreign_key='3',
        child_table_name='child_data',
    )

    test_metadata = Metadata.detect_from_dataframes(data)
    test_metadata.update_column('1', 'parent_data', sdtype='id')
    test_metadata.update_column('3', 'child_data', sdtype='id')
    test_metadata.update_column('4', 'child_data', sdtype='id')
    test_metadata.set_primary_key('1', 'parent_data')
    test_metadata.set_primary_key('4', 'child_data')
    test_metadata.add_relationship(
        parent_primary_key='1',
        parent_table_name='parent_data',
        child_foreign_key='3',
        child_table_name='child_data',
    )

    # Run
    instance = HMASynthesizer(metadata)
    instance.fit(data)
    sample = instance.sample(5)

    # Assert
    assert test_metadata.to_dict() == metadata.to_dict()
    assert sample['parent_data'].columns.tolist() == data['parent_data'].columns.tolist()
    assert sample['child_data'].columns.tolist() == data['child_data'].columns.tolist()

    test_metadata = Metadata.detect_from_dataframes(data)


def test_table_name_logging(caplog):
    """Test the table name is correctly logged GH#1964."""
    # Setup
    parent_data = pd.DataFrame({
        'parent_id': [1, 2, 3, 4, 5, 6],
        'col': ['a', 'b', 'a', 'b', 'a', 'b'],
    })
    child_data = pd.DataFrame({'id': [1, 2, 3, 4, 5, 6], 'parent_id': [1, 2, 3, 4, 5, 6]})
    data = {
        'parent_data': parent_data,
        'child_data': child_data,
    }
    metadata = Metadata.detect_from_dataframes(data)
    instance = HMASynthesizer(metadata)

    # Run
    with catch_sdv_logs(caplog, logging.INFO, 'sdv.data_processing.data_processor'):
        instance.fit(data)

    # Assert
    for msg in caplog.messages:
        assert 'table parent_data' in msg or 'table child_data' in msg


def test_disjointed_tables():
    """Test to see if synthesizer works with disjointed tables."""
    # Setup
    real_data, metadata = download_demo('multi_table', 'Bupa_v1')

    # Delete Some Relationships to make it disjointed
    remove_some_dict = metadata.to_dict()
    half_list = remove_some_dict['relationships'][1::2]
    remove_some_dict['relationships'] = half_list
    disjoined_metadata = Metadata.load_from_dict(remove_some_dict)

    # Run
    disjoin_synthesizer = HMASynthesizer(disjoined_metadata)
    disjoin_synthesizer.fit(real_data)
    disjoin_synthetic_data = disjoin_synthesizer.sample(1.0)

    # Assert
    for table in real_data:
        assert list(real_data[table].columns) == list(disjoin_synthetic_data[table].columns)


def test_small_sample():
    """Test that the sample function still works with a small scale"""
    # Setup
    data, metadata = download_demo(modality='multi_table', dataset_name='fake_hotels')
    synthesizer = HMASynthesizer(metadata)
    synthesizer.fit(data)

    # Run and Assert
    warn_msg = re.escape(
        "The 'scale' parameter is too small. Some tables may have 1 row."
        ' For better quality data, please choose a larger scale.'
    )
    with pytest.warns(Warning, match=warn_msg):
        synthetic_data = synthesizer.sample(scale=0.01)

    assert len(synthetic_data['hotels']) == 1
    assert len(synthetic_data['guests']) >= len(data['guests']) * 0.01
    assert synthetic_data['hotels'].columns.tolist() == data['hotels'].columns.tolist()
    assert synthetic_data['guests'].columns.tolist() == data['guests'].columns.tolist()


def test_hma_synthesizer_with_fixed_combinations():
    """Tests that https://github.com/sdv-dev/SDV/issues/2087 does not occur."""
    # Creating the dataset
    data = {
        'users': pd.DataFrame({
            'user_id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'John'],
        }),
        'records': pd.DataFrame({
            'record_id': ['record_a', 'record_b', 'record_c', 'record_d'],
            'user_id': [1, 2, 2, 1],
            'score': [85, 92, 78, 88],
            'location_id': ['A', 'B', 'C', 'D'],
            'department': ['HR', 'IT', 'HR', 'Finance'],
            'office': ['Boston HQ', 'NYC Office', 'LA Office', 'Chicago HQ'],
        }),
        'locations': pd.DataFrame({
            'location_id': ['A', 'B', 'C', 'D'],
            'city': ['Boston', 'New York', 'Los Angeles', 'Chicago'],
            'country': ['USA', 'USA', 'USA', 'USA'],
        }),
    }

    # Creating metadata for the dataset
    metadata = Metadata.detect_from_dataframes(data)

    metadata.update_column('user_id', 'users', sdtype='id')
    metadata.update_column('record_id', 'records', sdtype='id')
    metadata.update_column('user_id', 'records', sdtype='id')
    metadata.update_column('location_id', 'records', sdtype='id')
    metadata.update_column('location_id', 'locations', sdtype='id')
    metadata.set_primary_key('user_id', 'users')
    metadata.set_primary_key('location_id', 'locations')

    # Adding FixedCombinations to HMASynthesizer
    constraint = FixedCombinations(
        table_name='records',
        column_names=['department', 'office'],
    )
    synthesizer = HMASynthesizer(metadata)
    synthesizer.add_constraints(constraints=[constraint])

    synthesizer.fit(data)
    sampled = synthesizer.sample(1)

    # Assert
    assert len(sampled['users']) > 1
    assert len(sampled['records']) > 1
    assert len(sampled['locations']) > 1


REGEXES = ['[0-9]{3,4}', '0HQ-[a-z]', '0+', r'\d', r'\d{1,5}', r'\w']


@pytest.mark.parametrize('regex', REGEXES)
def test_fit_int_primary_key_regex_includes_zero(regex):
    """Test that sdv errors if the primary key has a regex, is an int, and can start with 0."""
    # Setup
    parent_data = pd.DataFrame({
        'parent_id': [1, 2, 3, 4, 5, 6],
        'col': ['a', 'b', 'a', 'b', 'a', 'b'],
    })
    child_data = pd.DataFrame({'id': [1, 2, 3, 4, 5, 6], 'parent_id': [1, 2, 3, 4, 5, 6]})
    data = {
        'parent_data': parent_data,
        'child_data': child_data,
    }
    metadata = Metadata.detect_from_dataframes(data)
    metadata.update_column('parent_id', 'parent_data', sdtype='id', regex_format=regex)
    metadata.set_primary_key('parent_id', 'parent_data')

    # Run and Assert
    instance = HMASynthesizer(metadata)
    message = (
        'Primary key for table "parent_data" is stored as an int but the Regex allows it to start '
        'with "0". Please remove the Regex or update it to correspond to valid ints.'
    )
    with pytest.raises(InvalidDataError, match=message):
        instance.fit(data)


def test__estimate_num_columns_to_be_modeled_various_sdtypes():
    """Test the estimated number of columns is correct for various sdtypes.

    To check that the number columns is correct we Mock the ``_finalize`` method
    and compare its output with the estimated number of columns.

    The dataset used follows the structure below:
        R1 R2
        | /
        GP
        |
        P
    """
    # Setup
    root1 = pd.DataFrame({'R1': [0, 1, 2]})
    root2 = pd.DataFrame({'R2': [0, 1, 2], 'data': [0, 1, 2]})
    grandparent = pd.DataFrame({'GP': [0, 1, 2], 'R1': [0, 1, 2], 'R2': [0, 1, 2]})
    parent = pd.DataFrame({
        'P': [0, 1, 2],
        'GP': [0, 1, 2],
        'numerical': [0.1, 0.5, np.nan],
        'categorical': ['a', np.nan, 'c'],
        'datetime': [None, '2019-01-02', '2019-01-03'],
        'boolean': [float('nan'), False, True],
        'id': [0, 1, 2],
    })
    data = {
        'root1': root1,
        'root2': root2,
        'grandparent': grandparent,
        'parent': parent,
    }
    metadata = Metadata.load_from_dict({
        'tables': {
            'root1': {
                'primary_key': 'R1',
                'columns': {
                    'R1': {'sdtype': 'id'},
                },
            },
            'root2': {
                'primary_key': 'R2',
                'columns': {'R2': {'sdtype': 'id'}, 'data': {'sdtype': 'numerical'}},
            },
            'grandparent': {
                'primary_key': 'GP',
                'columns': {
                    'GP': {'sdtype': 'id'},
                    'R1': {'sdtype': 'id'},
                    'R2': {'sdtype': 'id'},
                },
            },
            'parent': {
                'primary_key': 'P',
                'columns': {
                    'P': {'sdtype': 'id'},
                    'GP': {'sdtype': 'id'},
                    'numerical': {'sdtype': 'numerical'},
                    'categorical': {'sdtype': 'categorical'},
                    'datetime': {'sdtype': 'datetime'},
                    'boolean': {'sdtype': 'boolean'},
                    'id': {'sdtype': 'id'},
                },
            },
        },
        'relationships': [
            {
                'parent_table_name': 'root1',
                'parent_primary_key': 'R1',
                'child_table_name': 'grandparent',
                'child_foreign_key': 'R1',
            },
            {
                'parent_table_name': 'root2',
                'parent_primary_key': 'R2',
                'child_table_name': 'grandparent',
                'child_foreign_key': 'R2',
            },
            {
                'parent_table_name': 'grandparent',
                'parent_primary_key': 'GP',
                'child_table_name': 'parent',
                'child_foreign_key': 'GP',
            },
        ],
    })
    synthesizer = HMASynthesizer(metadata)
    synthesizer._finalize = Mock(return_value=data)

    # Run estimation
    estimated_num_columns = synthesizer._estimate_num_columns(metadata)

    # Run actual modeling
    synthesizer.fit(data)
    synthesizer.sample()

    # Assert estimated number of columns is correct
    tables = synthesizer._finalize.call_args[0][0]
    for table_name, table in tables.items():
        # Subract all the id columns present in the data, as those are not estimated
        num_table_cols = len(table.columns)
        if table_name == 'grandparent':
            num_table_cols -= 3
        if table_name == 'parent':
            num_table_cols -= 2
        if table_name in {'root1', 'root2'}:
            num_table_cols -= 1

        assert num_table_cols == estimated_num_columns[table_name]


def test_column_order():
    """Test that the column order of the synthetic data is the one of the metadata."""
    # Setup
    table_1 = pd.DataFrame({
        'col_1': [1, 2, 3],
        'col_3': [7, 8, 9],
        'col_2': [4, 5, 6],
    })
    table_2 = pd.DataFrame({
        'col_A': ['a', 'b', 'c'],
        'col_B': ['d', 'e', 'f'],
        'col_C': ['g', 'h', 'i'],
    })
    metadata = Metadata.load_from_dict({
        'tables': {
            'table_1': {
                'columns': {
                    'col_1': {'sdtype': 'numerical'},
                    'col_2': {'sdtype': 'numerical'},
                    'col_3': {'sdtype': 'numerical'},
                },
            },
            'table_2': {
                'columns': {
                    'col_A': {'sdtype': 'categorical'},
                    'col_B': {'sdtype': 'categorical'},
                    'col_C': {'sdtype': 'categorical'},
                },
            },
        }
    })
    data = {
        'table_1': table_1,
        'table_2': table_2,
    }

    synthesizer = HMASynthesizer(metadata)
    synthesizer.fit(data)

    # Run
    synthetic_data = synthesizer.sample()

    # Assert
    table_1_column = list(synthetic_data['table_1'].columns)
    assert table_1_column != list(data['table_1'].columns)
    assert table_1_column == ['col_1', 'col_2', 'col_3']
    assert list(synthetic_data['table_2'].columns) == ['col_A', 'col_B', 'col_C']


def test_no_deprecation_warning_single_table_metadata_sampling():
    """Test that no single-table metadata deprecation warning raises with `MultiTableMetadata`."""
    # Setup
    data, _ = download_demo(modality='multi_table', dataset_name='fake_hotels')
    multi_metadata = MultiTableMetadata()
    multi_metadata.detect_from_dataframes(data)
    synthesizer = HMASynthesizer(multi_metadata)
    synthesizer.fit(data)

    # Run
    with warnings.catch_warnings(record=True) as captured_warnings:
        synthesizer.sample()

    # Assert
    assert len(captured_warnings) == 0


def test__unsupported_regex_format():
    """Test that ``HMA`` raises an error if the regex format is not supported."""
    # Setup
    metadata = Metadata.load_from_dict({
        'tables': {
            'table_1': {
                'columns': {
                    'id': {'sdtype': 'id', 'regex_format': '(10|20|30)[0-9]{4}'},
                    'A': {'sdtype': 'numerical'},
                }
            },
            'table_2': {
                'columns': {
                    'col_A': {'sdtype': 'numerical'},
                    'col_B': {'sdtype': 'categorical'},
                }
            },
        }
    })

    expected_error = re.escape(
        'SDV synthesizers do not currently support complex regex formats such as '
        "'(10|20|30)[0-9]{4}', which you have provided for table 'table_1', column 'id'. Please use"
        ' a simplified format or update to a different sdtype.'
    )

    # Run and Assert
    with pytest.raises(SynthesizerInputError, match=expected_error):
        HMASynthesizer(metadata)


def test_end_to_end_with_cags():
    """Test HMA with a single-table cag."""
    # Setup
    data, metadata = download_demo('multi_table', 'fake_hotels')
    data['guests']['amenities_lower'] = data['guests']['amenities_fee'] - np.random.rand(
        len(data['guests'])
    )
    metadata.add_column(
        table_name='guests',
        column_name='amenities_lower',
        sdtype='numerical',
    )
    synthesizer = HMASynthesizer(metadata)
    constraint = Inequality(
        low_column_name='amenities_lower',
        high_column_name='amenities_fee',
        strict_boundaries=False,
        table_name='guests',
    )
    synthesizer.add_constraints(constraints=[constraint])
    data_guests = data['guests']
    clean_data = data_guests[
        ~(data_guests[['amenities_lower', 'amenities_fee']].isna().any(axis=1))
    ]
    data_invalid = clean_data.copy()
    data_invalid.loc[0, 'amenities_lower'] = data_invalid.loc[0, 'amenities_fee'] + 1
    data['guests'] = clean_data
    invalid_data = data.copy()
    invalid_data['guests'] = data_invalid
    expected_error_msg = re.escape(
        "Data is not valid for the 'Inequality' constraint in table 'guests':\n   amenities_lower"
        '  amenities_fee\n0            38.89          37.89'
    )

    # Run
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample(scale=1.0)

    with pytest.raises(ConstraintNotMetError, match=expected_error_msg):
        synthesizer.fit(invalid_data)

    # Assert
    synthesizer.validate(synthetic_data)
