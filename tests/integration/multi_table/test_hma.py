import datetime
import importlib.metadata
import re
import warnings

import numpy as np
import pandas as pd
import pytest
from faker import Faker
from rdt.transformers import FloatFormatter
from sdmetrics.reports.multi_table import DiagnosticReport

from sdv import version
from sdv.datasets.demo import download_demo
from sdv.datasets.local import load_csvs
from sdv.errors import SamplingError, SynthesizerInputError, VersionError
from sdv.evaluation.multi_table import evaluate_quality, get_column_pair_plot, get_column_plot
from sdv.metadata.multi_table import MultiTableMetadata
from sdv.multi_table import HMASynthesizer
from tests.integration.single_table.custom_constraints import MyConstraint


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
            assert all(table.columns == data[table_name].columns)

        for normal_table, increased_table in zip(
                normal_sample.values(), increased_sample.values()):
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

    def test_get_info(self):
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
        version = importlib.metadata.version('sdv')
        assert info == {
            'class_name': 'HMASynthesizer',
            'creation_date': today,
            'is_fit': True,
            'last_fit_date': today,
            'fitted_sdv_version': version
        }

    def test_hma_set_table_parameters(self):
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
        character_params = hmasynthesizer.get_table_parameters('characters')
        assert character_params['synthesizer_name'] == 'GaussianCopulaSynthesizer'
        assert character_params['synthesizer_parameters'] == {
            'default_distribution': 'gamma',
            'enforce_min_max_values': True,
            'enforce_rounding': True,
            'locales': ['en_US'],
            'numerical_distributions': {}
        }
        families_params = hmasynthesizer.get_table_parameters('families')
        assert families_params['synthesizer_name'] == 'GaussianCopulaSynthesizer'
        assert families_params['synthesizer_parameters'] == {
            'default_distribution': 'uniform',
            'enforce_min_max_values': True,
            'enforce_rounding': True,
            'locales': ['en_US'],
            'numerical_distributions': {}
        }
        char_families_params = hmasynthesizer.get_table_parameters('character_families')
        assert char_families_params['synthesizer_name'] == 'GaussianCopulaSynthesizer'
        assert char_families_params['synthesizer_parameters'] == {
            'default_distribution': 'norm',
            'enforce_min_max_values': True,
            'enforce_rounding': True,
            'locales': ['en_US'],
            'numerical_distributions': {}
        }

        assert hmasynthesizer._table_synthesizers['characters'].default_distribution == 'gamma'
        assert hmasynthesizer._table_synthesizers['families'].default_distribution == 'uniform'
        assert hmasynthesizer._table_synthesizers['character_families'].default_distribution == \
            'norm'

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

    def test_hma_custom_constraint(self):
        """Test an example of using a custom constraint."""
        # Setup
        parent_data, child_data, metadata = self.get_custom_constraint_data_and_metadata()
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

    def test_hma_custom_constraint_2_tables(self):
        """Test an example of using a custom constraint.

        Check that the same custom constraint can be applied to two different tables and columns.
        """
        # Setup
        parent_data, child_data, metadata = self.get_custom_constraint_data_and_metadata()
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

    def test_hma_custom_constraint_loaded_from_file(self):
        """Test an example of using a custom constraint loaded from a file."""
        # Setup
        parent_data, child_data, metadata = self.get_custom_constraint_data_and_metadata()
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

    def test_hma_with_inequality_constraint(self):
        """Test that when new columns are created by the constraint this still works."""
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
                'default_distribution': 'truncnorm',
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

    def test_use_own_data_using_hma(self, tmp_path):
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

    def test_progress_bar_print(self, capsys):
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

    def test_warning_message_too_many_cols(self, capsys):
        """Test that a warning appears if there are more than 1000 expected columns"""
        # Setup
        (_, metadata) = download_demo(
            modality='multi_table',
            dataset_name='NBA_v1'
        )

        key_phrases = [
            r'PerformanceAlert:',
            r'large number of columns.',
            r'contact us at info@sdv.dev for enterprise solutions.'
        ]

        # Run
        HMASynthesizer(metadata)

        captured = capsys.readouterr()

        # Assert
        for pattern in key_phrases:
            match = re.search(pattern, captured.out + captured.err)
            assert match is not None
        (_, small_metadata) = download_demo(
            modality='multi_table',
            dataset_name='trains_v1'
        )

        # Run
        HMASynthesizer(small_metadata)

        captured = capsys.readouterr()

        # Assert that small amount of columns don't trigger the message
        for pattern in key_phrases:
            match = re.search(pattern, captured.out + captured.err)
            assert match is None

    def test_hma_three_linear_nodes(self):
        """Test it works on a simple 'grandparent-parent-child' dataset."""
        # Setup
        grandparent = pd.DataFrame(data={
            'grandparent_ID': [0, 1, 2, 3, 4],
            'data': ['0', '1', '2', '3', '4']
        })
        parent = pd.DataFrame(data={
            'parent_ID': ['a', 'b', 'c', 'd', 'e'],
            'grandparent_ID': [0, 0, 1, 1, 3],
            'data': [True, False, False, False, True]
        })
        child = pd.DataFrame(data={
            'child_ID': ['00', '01', '02', '03', '04'],
            'parent_ID': ['b', 'b', 'a', 'e', 'e'],
            'data': ['Yes', 'Yes', 'Maybe', 'No', 'No']
        })
        data = {'grandparent': grandparent, 'parent': parent, 'child': child}
        metadata = MultiTableMetadata.load_from_dict({
            'tables': {
                'grandparent': {
                    'primary_key': 'grandparent_ID',
                    'columns': {
                        'grandparent_ID': {'sdtype': 'id'},
                        'data': {'sdtype': 'categorical'}
                    }
                },
                'parent': {
                    'primary_key': 'parent_ID',
                    'columns': {
                        'parent_ID': {'sdtype': 'id'},
                        'grandparent_ID': {'sdtype': 'id'},
                        'data': {'sdtype': 'categorical'}
                    }
                },
                'child': {
                    'primary_key': 'child_ID',
                    'columns': {
                        'child_ID': {'sdtype': 'id'},
                        'parent_ID': {'sdtype': 'id'},
                        'data': {'sdtype': 'categorical'}
                    }
                }
            },
            'relationships': [
                {
                    'parent_table_name': 'grandparent',
                    'parent_primary_key': 'grandparent_ID',
                    'child_table_name': 'parent',
                    'child_foreign_key': 'grandparent_ID'
                },
                {
                    'parent_table_name': 'parent',
                    'parent_primary_key': 'parent_ID',
                    'child_table_name': 'child',
                    'child_foreign_key': 'parent_ID'
                }
            ]
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
        parent = pd.DataFrame(data={
            'parent_ID': [0, 1, 2, 3, 4],
            'data': ['0', '1', '2', '3', '4']
        })
        child1 = pd.DataFrame(data={
            'child_ID': ['a', 'b', 'c', 'd', 'e'],
            'parent_ID': [0, 0, 1, 1, 3],
            'data': [True, False, False, False, True]
        })
        child2 = pd.DataFrame(data={
            'child_ID': ['00', '01', '02', '03', '04'],
            'parent_ID': [0, 1, 2, 3, 4],
            'data': ['Yes', 'Yes', 'Maybe', 'No', 'No']
        })
        data = {'parent': parent, 'child1': child1, 'child2': child2}
        metadata = MultiTableMetadata.load_from_dict({
            'tables': {
                'parent': {
                    'primary_key': 'parent_ID',
                    'columns': {
                        'parent_ID': {'sdtype': 'id'},
                        'data': {'sdtype': 'categorical'}
                    }
                },
                'child1': {
                    'primary_key': 'child_ID',
                    'columns': {
                        'child_ID': {'sdtype': 'id'},
                        'parent_ID': {'sdtype': 'id'},
                        'data': {'sdtype': 'categorical'}
                    }
                },
                'child2': {
                    'primary_key': 'child_ID',
                    'columns': {
                        'child_ID': {'sdtype': 'id'},
                        'parent_ID': {'sdtype': 'id'},
                        'data': {'sdtype': 'categorical'}
                    }
                }
            },
            'relationships': [
                {
                    'parent_table_name': 'parent',
                    'parent_primary_key': 'parent_ID',
                    'child_table_name': 'child1',
                    'child_foreign_key': 'parent_ID'
                },
                {
                    'parent_table_name': 'parent',
                    'parent_primary_key': 'parent_ID',
                    'child_table_name': 'child2',
                    'child_foreign_key': 'parent_ID'
                },
            ]
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
        child = pd.DataFrame(data={
            'child_ID': ['a', 'b', 'c', 'd', 'e'],
            'parent_ID1': [0, 1, 2, 3, 3],
            'parent_ID2': [0, 1, 2, 3, 4],
            'data': ['0', '1', '2', '3', '4']
        })
        parent1 = pd.DataFrame(data={
            'parent_ID1': [0, 1, 2, 3, 4],
            'data': [True, False, False, False, True]
        })
        parent2 = pd.DataFrame(data={
            'parent_ID2': [0, 1, 2, 3, 4],
            'data': ['Yes', 'Yes', 'Maybe', 'No', 'No']
        })
        data = {'parent1': parent1, 'child': child, 'parent2': parent2}
        metadata = MultiTableMetadata.load_from_dict({
            'tables': {
                'parent1': {
                    'primary_key': 'parent_ID1',
                    'columns': {
                        'parent_ID1': {'sdtype': 'id'},
                        'data': {'sdtype': 'categorical'}
                    }
                },
                'parent2': {
                    'primary_key': 'parent_ID2',
                    'columns': {
                        'parent_ID2': {'sdtype': 'id'},
                        'data': {'sdtype': 'categorical'}
                    }
                },
                'child': {
                    'primary_key': 'child_ID',
                    'columns': {
                        'child_ID': {'sdtype': 'id'},
                        'parent_ID1': {'sdtype': 'id'},
                        'parent_ID2': {'sdtype': 'id'},
                        'data': {'sdtype': 'categorical'}
                    }
                },
            },
            'relationships': [
                {
                    'parent_table_name': 'parent1',
                    'parent_primary_key': 'parent_ID1',
                    'child_table_name': 'child',
                    'child_foreign_key': 'parent_ID1'
                },
                {
                    'parent_table_name': 'parent2',
                    'parent_primary_key': 'parent_ID2',
                    'child_table_name': 'child',
                    'child_foreign_key': 'parent_ID2'
                },
            ]
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
        root1 = pd.DataFrame(data={
            'id': [0, 1, 2, 3, 4],
            'data': [True, False, False, False, True]
        })
        root2 = pd.DataFrame(data={
            'id': [0, 1, 2, 3, 4],
            'data': [True, False, False, False, True]
        })
        child1 = pd.DataFrame(data={
            'child_ID': ['a', 'b', 'c', 'd', 'e'],
            'root1_ID': [0, 1, 2, 3, 3],
            'data': [True, False, False, False, True]
        })
        child2 = pd.DataFrame(data={
            'child_ID': ['a', 'b', 'c', 'd', 'e'],
            'root2_ID': [0, 1, 2, 3, 4],
            'data': [True, False, False, False, True]
        })
        grandchild = pd.DataFrame(data={
            'grandchild_ID': ['a', 'b', 'c', 'd', 'e'],
            'child1_ID': ['a', 'b', 'c', 'd', 'e'],
            'child2_ID': ['a', 'b', 'c', 'd', 'e'],
            'data': [True, False, False, False, True]
        })
        data = {
            'root1': root1,
            'root2': root2,
            'child1': child1,
            'child2': child2,
            'grandchild': grandchild
        }
        metadata = MultiTableMetadata.load_from_dict({
            'tables': {
                'root1': {
                    'primary_key': 'id',
                    'columns': {
                        'id': {'sdtype': 'id'},
                        'data': {'sdtype': 'categorical'}
                    }
                },
                'root2': {
                    'primary_key': 'id',
                    'columns': {
                        'id': {'sdtype': 'id'},
                        'data': {'sdtype': 'categorical'}
                    }
                },
                'child1': {
                    'primary_key': 'child_ID',
                    'columns': {
                        'child_ID': {'sdtype': 'id'},
                        'root1_ID': {'sdtype': 'id'},
                        'data': {'sdtype': 'categorical'}
                    }
                },
                'child2': {
                    'primary_key': 'child_ID',
                    'columns': {
                        'child_ID': {'sdtype': 'id'},
                        'root2_ID': {'sdtype': 'id'},
                        'data': {'sdtype': 'categorical'}
                    }
                },
                'grandchild': {
                    'primary_key': 'grandchild_ID',
                    'columns': {
                        'grandchild_ID': {'sdtype': 'id'},
                        'child1_ID': {'sdtype': 'id'},
                        'child2_ID': {'sdtype': 'id'},
                        'data': {'sdtype': 'categorical'}
                    }
                },
            },
            'relationships': [
                {
                    'parent_table_name': 'root1',
                    'parent_primary_key': 'id',
                    'child_table_name': 'child1',
                    'child_foreign_key': 'root1_ID'
                },
                {
                    'parent_table_name': 'root2',
                    'parent_primary_key': 'id',
                    'child_table_name': 'child2',
                    'child_foreign_key': 'root2_ID'
                },
                {
                    'parent_table_name': 'child1',
                    'parent_primary_key': 'child_ID',
                    'child_table_name': 'grandchild',
                    'child_foreign_key': 'child1_ID'
                },
                {
                    'parent_table_name': 'child2',
                    'parent_primary_key': 'child_ID',
                    'child_table_name': 'grandchild',
                    'child_foreign_key': 'child2_ID'
                },
            ]
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
            table_parameters={
                'numerical_distributions': {
                    'amenities_fee': 'beta'
                }
            }
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
        data, metadata = download_demo(
            modality='multi_table',
            dataset_name='fake_hotels'
        )
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
            hmasynthesizer._table_synthesizers['characters'],
            len(data['characters'])
        )
        hmasynthesizer._sample_children('characters', sampled_data)

        # Run
        likelihoods = hmasynthesizer._get_likelihoods(
            sampled_data['character_families'],
            sampled_data['characters'].set_index('character_id'),
            'character_families',
            'character_id'
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
            '__sessions__user_id__scale': -0.25
        })
        instance = HMASynthesizer(MultiTableMetadata())
        instance.extended_columns = {
            'sessions': {
                '__sessions__user_id__num_rows': FloatFormatter(enforce_min_max_values=True),
                '__sessions__user_id__a': FloatFormatter(enforce_min_max_values=True),
                '__sessions__user_id__b': FloatFormatter(enforce_min_max_values=True),
                '__sessions__user_id__loc': FloatFormatter(enforce_min_max_values=True),
                '__sessions__user_id__scale': FloatFormatter(enforce_min_max_values=True)
            }
        }
        for col, float_formatter in instance.extended_columns['sessions'].items():
            float_formatter.fit(pd.DataFrame({col: [0., 100.]}), col)

        instance._max_child_rows = {'__sessions__user_id__num_rows': 10}

        # Run
        result = instance._extract_parameters(parent_row, 'sessions', 'user_id')

        # Assert
        expected_result = {
            'a': 0.,
            'b': 100.,
            'loc': 0.5,
            'num_rows': 10.,
            'scale': 0.
        }
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
            f'{prefix}univariates__brand__scale': -0.25
        })
        metadata = MultiTableMetadata.load_from_dict({
            'tables': {
                'users': {
                    'columns': {
                        'user_id': {'sdtype': 'id'}
                    },
                    'primary_key': 'user_id'
                },
                'sessions': {
                    'columns': {
                        'user_id': {'sdtype': 'id'},
                        'brand': {'sdtype': 'categorical'}
                    }
                }
            },
            'relationships': [
                {
                    'parent_table_name': 'users',
                    'child_table_name': 'sessions',
                    'parent_primary_key': 'user_id',
                    'child_foreign_key': 'user_id'
                }
            ]
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
                f'{prefix}univariates__brand__scale': FloatFormatter(enforce_min_max_values=True)
            }
        }
        for col, float_formatter in instance.extended_columns['sessions'].items():
            float_formatter.fit(pd.DataFrame({col: [0., 100.]}), col)

        instance._default_parameters = {
            'sessions': {
                'univariates__brand__a': 5,
                'univariates__brand__b': 84,
                'univariates__brand__loc': 1,
                'univariates__brand__scale': 1
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
            'num_rows': 10
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
        data, metadata = download_demo('multi_table', 'got_families')

        # Run 1
        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter('always')
            instance = HMASynthesizer(metadata)
            instance.fit(data)

        # Assert
        assert len(captured_warnings) == 0

        # Run 2
        metadata_detect = MultiTableMetadata()
        metadata_detect.detect_from_dataframes(data)

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
            table_name='characters', column_name='age', sdtype='categorical')
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
        metadata_detect = MultiTableMetadata()
        metadata_detect.detect_from_dataframes(data)

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
        """Test that the synthesizer crashes when there are null foreign keys."""
        # Setup
        metadata = MultiTableMetadata()
        metadata.add_table('parent_table')
        metadata.add_column('parent_table', 'id', sdtype='id')
        metadata.set_primary_key('parent_table', 'id')

        metadata.add_table('child_table1')
        metadata.add_column('child_table1', 'id', sdtype='id')
        metadata.set_primary_key('child_table1', 'id')
        metadata.add_column('child_table1', 'fk', sdtype='id')

        metadata.add_table('child_table2')
        metadata.add_column('child_table2', 'id', sdtype='id')
        metadata.set_primary_key('child_table2', 'id')
        metadata.add_column('child_table2', 'fk1', sdtype='id')
        metadata.add_column('child_table2', 'fk2', sdtype='id')

        metadata.add_relationship(
            parent_table_name='parent_table',
            child_table_name='child_table1',
            parent_primary_key='id',
            child_foreign_key='fk'
        )

        metadata.add_relationship(
            parent_table_name='parent_table',
            child_table_name='child_table2',
            parent_primary_key='id',
            child_foreign_key='fk1'
        )

        metadata.add_relationship(
            parent_table_name='parent_table',
            child_table_name='child_table2',
            parent_primary_key='id',
            child_foreign_key='fk2'
        )

        data = {
            'parent_table': pd.DataFrame({
                'id': [1, 2, 3]
            }),
            'child_table1': pd.DataFrame({
                'id': [1, 2, 3],
                'fk': [1, 2, np.nan]
            }),
            'child_table2': pd.DataFrame({
                'id': [1, 2, 3],
                'fk1': [1, 2, np.nan],
                'fk2': [1, 2, np.nan]
            })
        }

        synthesizer = HMASynthesizer(metadata)

        # Run
        metadata.validate()
        metadata.validate_data(data)

        # Run and Assert
        err_msg = re.escape(
            'The data contains null values in foreign key columns. This feature is currently '
            'unsupported. Please remove null values to fit the synthesizer.\n'
            '\n'
            'Affected columns:\n'
            "Table 'child_table1', column(s) ['fk']\n"
            "Table 'child_table2', column(s) ['fk1', 'fk2']\n"
        )
        with pytest.raises(SynthesizerInputError, match=err_msg):
            synthesizer.fit(data)


parametrization = [
    ('update_column', {
        'table_name': 'departure', 'column_name': 'city', 'sdtype': 'categorical'
    }),
    ('set_primary_key', {'table_name': 'arrival', 'column_name': 'id_flight'}),
    (
        'add_column_relationship', {
            'table_name': 'departure',
            'relationship_type': 'address',
            'column_names': ['city', 'country']
        }
    ),
    ('add_alternate_keys', {'table_name': 'departure', 'column_names': ['city', 'country']}),
    ('set_sequence_key', {'table_name': 'departure', 'column_name': 'city'}),
    ('add_column', {
        'table_name': 'departure', 'column_name': 'postal_code', 'sdtype': 'postal_code'
    }),
]


@pytest.mark.parametrize(('method', 'kwargs'), parametrization)
def test_metadata_updated_warning(method, kwargs):
    """Test that modifying metadata without saving it raise a warning.

    The warning should be raised during synthesizer initialization.
    """
    metadata = MultiTableMetadata().load_from_dict({
        'tables': {
            'departure': {
                'primary_key': 'id',
                'columns': {
                    'id': {'sdtype': 'id'},
                    'date': {'sdtype': 'datetime'},
                    'city': {'sdtype': 'city'},
                    'country': {'sdtype': 'country_code'}
                },
            },
            'arrival': {
                'foreign_key': 'id',
                'columns': {
                    'id': {'sdtype': 'id'},
                    'date': {'sdtype': 'datetime'},
                    'city': {'sdtype': 'city'},
                    'country': {'sdtype': 'country'},
                    'id_flight': {'sdtype': 'id'}
                },
            },
        },
        'relationships': [
            {
                'parent_table_name': 'departure',
                'parent_primary_key': 'id',
                'child_table_name': 'arrival',
                'child_foreign_key': 'id'
            }
        ]
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
    metadata = MultiTableMetadata().load_from_dict({
        'tables': {
            'departure': {
                'primary_key': 'id',
                'columns': {
                    'id': {'sdtype': 'id'},
                    'date': {'sdtype': 'datetime'},
                    'city': {'sdtype': 'city'},
                    'country': {'sdtype': 'country'}
                },
            },
            'arrival': {
                'foreign_key': 'id',
                'columns': {
                    'id': {'sdtype': 'id'},
                    'date': {'sdtype': 'datetime'},
                    'city': {'sdtype': 'city'},
                    'country': {'sdtype': 'country'},
                    'id_flight': {'sdtype': 'id'}
                },
            },
        },
        'relationships': [
            {
                'parent_table_name': 'departure',
                'parent_primary_key': 'id',
                'child_table_name': 'arrival',
                'child_foreign_key': 'id'
            }
        ]
    })

    instance = HMASynthesizer(metadata)
    instance._fitted = True
    instance._fitted_sdv_version = '10.0.0'
    synthesizer_path = tmp_path / 'synthesizer.pkl'
    instance.save(synthesizer_path)

    # Run and Assert
    error_msg = (
        f'You are currently on SDV version {version.public} but this '
        'synthesizer was created on version 10.0.0. '
        'Downgrading your SDV version is not supported.'
    )
    with pytest.raises(VersionError, match=error_msg):
        HMASynthesizer.load(synthesizer_path)


def test_fit_raises_version_error():
    """Test that a ``VersionError`` is being raised if the current version is newer."""
    # Setup
    metadata = MultiTableMetadata().load_from_dict({
        'tables': {
            'departure': {
                'primary_key': 'id',
                'columns': {
                    'id': {'sdtype': 'id'},
                    'date': {'sdtype': 'datetime'},
                    'city': {'sdtype': 'city'},
                    'country': {'sdtype': 'country'}
                },
            },
            'arrival': {
                'foreign_key': 'id',
                'columns': {
                    'id': {'sdtype': 'id'},
                    'date': {'sdtype': 'datetime'},
                    'city': {'sdtype': 'city'},
                    'country': {'sdtype': 'country'},
                    'id_flight': {'sdtype': 'id'}
                },
            },
        },
        'relationships': [
            {
                'parent_table_name': 'departure',
                'parent_primary_key': 'id',
                'child_table_name': 'arrival',
                'child_foreign_key': 'id'
            }
        ]
    })

    instance = HMASynthesizer(metadata)
    instance._fitted_sdv_version = '1.0.0'

    # Run and Assert
    expected_message = (
        f'You are currently on SDV version {version.public} but this synthesizer was created on '
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
    data, metadata = download_demo('multi_table', 'Dunur_v1')
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
    metadata = MultiTableMetadata()
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
            'child_foreign_key': 2
        }
    ]
    metadata = MultiTableMetadata.load_from_dict(metadata_dict)
    metadata.set_primary_key('0', '1')

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
