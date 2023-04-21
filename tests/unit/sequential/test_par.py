import re
from unittest.mock import ANY, Mock, mock_open, patch

import numpy as np
import pandas as pd
import pytest
from rdt.transformers import FloatFormatter, UnixTimestampEncoder

from sdv.data_processing.data_processor import DataProcessor
from sdv.errors import SamplingError, SynthesizerInputError
from sdv.metadata.single_table import SingleTableMetadata
from sdv.sequential.par import PARSynthesizer
from sdv.single_table.copulas import GaussianCopulaSynthesizer
from sdv.single_table.errors import InvalidDataError
from tests.utils import DataFrameMatcher


class TestPARSynthesizer:

    def get_metadata(self, add_sequence_key=True):
        metadata = SingleTableMetadata()
        metadata.add_column('time', sdtype='datetime')
        metadata.add_column('gender', sdtype='categorical')
        metadata.add_column('name', sdtype='id')
        metadata.add_column('measurement', sdtype='numerical')
        if add_sequence_key:
            metadata.set_sequence_key('name')

        return metadata

    def get_data(self):
        data = pd.DataFrame({
            'time': [1, 2, 3],
            'gender': ['F', 'M', 'M'],
            'name': ['Jane', 'John', 'Doe'],
            'measurement': [55, 60, 65]
        })
        return data

    def test___init__(self):
        """Test that the parameters are set correctly.

        The parameters passed in the ``__init__`` should be set on the instance. Additionally,
        a context synthesizer should be created with the correct metadata and parameters.
        """
        # Setup
        metadata = self.get_metadata()

        # Run
        synthesizer = PARSynthesizer(
            metadata=metadata,
            enforce_min_max_values=True,
            enforce_rounding=True,
            context_columns=['gender'],
            segment_size=10,
            epochs=10,
            sample_size=5,
            cuda=False,
            verbose=False
        )

        # Assert
        assert synthesizer.context_columns == ['gender']
        assert synthesizer._sequence_key == ['name']
        assert synthesizer.enforce_min_max_values is True
        assert synthesizer.enforce_rounding is True
        assert synthesizer.segment_size == 10
        assert synthesizer._model_kwargs == {
            'epochs': 10,
            'sample_size': 5,
            'cuda': False,
            'verbose': False
        }
        assert isinstance(synthesizer._data_processor, DataProcessor)
        assert synthesizer._data_processor.metadata == metadata
        assert isinstance(synthesizer._context_synthesizer, GaussianCopulaSynthesizer)
        assert synthesizer._context_synthesizer.metadata.columns == {
            'gender': {'sdtype': 'categorical'},
            'name': {'sdtype': 'id'}
        }

    def test___init___context_columns_no_sequence_key(self):
        """Test when there are context columns but no sequence keys.

        If there are context columns and no sequence keys then an error should be raised.
        """
        # Setup
        metadata = self.get_metadata(add_sequence_key=False)

        # Run and Assert
        error_message = (
            "No 'sequence_keys' are specified in the metadata. The PARSynthesizer cannot "
            "model 'context_columns' in this case."
        )
        with pytest.raises(SynthesizerInputError, match=error_message):
            PARSynthesizer(
                metadata=metadata,
                enforce_min_max_values=True,
                enforce_rounding=True,
                context_columns=['gender'],
                segment_size=10,
                epochs=10,
                sample_size=5,
                cuda=False,
                verbose=False
            )

    @patch('sdv.sequential.par.warnings')
    def test_add_constraints(self, warnings_mock):
        """Test that if constraints are being added, a warning is raised."""
        # Setup
        metadata = self.get_metadata()
        synthesizer = PARSynthesizer(metadata=metadata)

        # Run
        synthesizer.add_constraints([object()])

        # Assert
        warning_message = (
            'The PARSynthesizer does not yet support constraints. This model will ignore any '
            'constraints in the metadata.'
        )
        warnings_mock.warn.assert_called_once_with(warning_message)
        assert synthesizer._data_processor._constraints == []
        assert synthesizer._data_processor._constraints_list == []

    def test_get_parameters(self):
        """Test that it returns every ``init`` parameter without the ``metadata``."""
        # Setup
        metadata = SingleTableMetadata()
        instance = PARSynthesizer(
            metadata=metadata,
            enforce_min_max_values=True,
            enforce_rounding=True,
            locales='en_CA',
            context_columns=None,
            segment_size=10,
            epochs=10,
            sample_size=5,
            cuda=False,
            verbose=False
        )

        # Run
        parameters = instance.get_parameters()

        # Assert
        assert 'metadata' not in parameters
        assert parameters == {
            'enforce_min_max_values': True,
            'enforce_rounding': True,
            'locales': 'en_CA',
            'context_columns': [],
            'segment_size': 10,
            'epochs': 10,
            'sample_size': 5,
            'cuda': False,
            'verbose': False
        }

    def test_get_metadata(self):
        """Test that it returns the ``metadata`` object."""
        # Setup
        metadata = SingleTableMetadata()
        instance = PARSynthesizer(
            metadata=metadata,
            enforce_min_max_values=True,
            enforce_rounding=True,
            context_columns=None,
            segment_size=10,
            epochs=10,
            sample_size=5,
            cuda=False,
            verbose=False
        )

        # Run
        result = instance.get_metadata()

        # Assert
        assert result == metadata

    def test_validate_context_columns_unique_per_sequence_key(self):
        """Test error is raised if context column values vary for each tuple of sequence keys.

        Setup:
            A ``SingleTableMetadata`` instance where the context columns vary for different
            combinations of values of the sequence keys.
        """
        # Setup
        data = pd.DataFrame({
            'sk_col1': [1, 1, 2, 2, 2],
            'sk_col2': [1, 1, 2, 2, 3],
            'ct_col1': [1, 2, 2, 3, 2],
            'ct_col2': [3, 3, 4, 3, 2],
        })
        metadata = SingleTableMetadata()
        metadata.add_column('sk_col1', sdtype='id')
        metadata.add_column('sk_col2', sdtype='id')
        metadata.add_column('ct_col1', sdtype='numerical')
        metadata.add_column('ct_col2', sdtype='numerical')
        metadata.set_sequence_key(('sk_col1', 'sk_col2'))
        instance = PARSynthesizer(
            metadata=metadata,
            context_columns=['ct_col1', 'ct_col2']
        )

        # Run and Assert
        err_msg = re.escape(
            'The provided data does not match the metadata:'
            "\nContext column 'ct_col1' is changing inside "
            "sequence (['sk_col1', 'sk_col2']=(1, 1))."
            '\n'
            "\nContext column 'ct_col1' is changing inside "
            "sequence (['sk_col1', 'sk_col2']=(2, 2))."
            '\n'
            "\nContext column 'ct_col2' is changing inside "
            "sequence (['sk_col1', 'sk_col2']=(2, 2))."
        )
        with pytest.raises(InvalidDataError, match=err_msg):
            instance.validate(data)

    @patch('sdv.sequential.par.BaseSynthesizer._preprocess')
    def test_preprocess_transformers_not_assigned(self, base_preprocess_mock):
        """Test that the method auto assigns the transformers if not already done.

        If the transformers in the ``DataProcessor`` haven't been assigned, then this method
        should do it so that it can overwrite the transformers for all the sequence key columns.
        """
        # Setup
        metadata = self.get_metadata()
        par = PARSynthesizer(
            metadata=metadata
        )
        par.auto_assign_transformers = Mock()
        par.update_transformers = Mock()
        data = self.get_data()

        # Run
        par.preprocess(data)

        # Assert
        expected_transformers = {'name': None}
        par.auto_assign_transformers.assert_called_once_with(data)
        par.update_transformers.assert_called_once_with(expected_transformers)
        base_preprocess_mock.assert_called_once_with(data)

    @patch('sdv.sequential.par.BaseSynthesizer._preprocess')
    def test_preprocess(self, base_preprocess_mock):
        """Test that the method does not auto assign the transformers if it's already been done.

        To test this, we set the hyper transformer's ``_prepared_for_fitting`` to True.
        """
        # Setup
        metadata = self.get_metadata()
        par = PARSynthesizer(
            metadata=metadata
        )
        par.auto_assign_transformers = Mock()
        par.update_transformers = Mock()
        par._data_processor._prepared_for_fitting = True
        data = self.get_data()

        # Run
        par.preprocess(data)

        # Assert
        expected_transformers = {'name': None}
        par.auto_assign_transformers.assert_not_called()
        par.update_transformers.assert_called_once_with(expected_transformers)
        base_preprocess_mock.assert_called_once_with(data)

    def test_update_transformers(self):
        """Test that it updates the transfomer correctly."""
        # Setup
        metadata = self.get_metadata()
        data = self.get_data()
        instance = PARSynthesizer(metadata)
        transformer = UnixTimestampEncoder()
        column_name_to_transformer = {'time': transformer}

        # Run
        instance.auto_assign_transformers(data)
        instance.update_transformers(column_name_to_transformer)

        # Assert
        assert instance.get_transformers()['time'] == transformer

    def test_update_transformers_context_column(self):
        """Test it errors out when a column name is a context column."""
        # Setup
        metadata = self.get_metadata()
        data = self.get_data()
        instance = PARSynthesizer(metadata, context_columns=['time'])

        # Run and Assert
        instance.auto_assign_transformers(data)
        err_msg = 'Transformers for context columns are not allowed to be updated.'
        with pytest.raises(SynthesizerInputError, match=err_msg):
            instance.update_transformers({'time': FloatFormatter()})

    def test__fit_context_model_with_context_columns(self):
        """Test that the method fits a synthesizer to the context columns.

        If there are context columns, the method should create a new DataFrame that groups
        the data by the sequence_key and only contains the context columns. Then a synthesizer
        should be fit to this new data.
        """
        # Setup
        metadata = self.get_metadata()
        data = self.get_data()
        par = PARSynthesizer(metadata, context_columns=['gender'])
        par._context_synthesizer = Mock()

        # Run
        par._fit_context_model(data)

        # Assert
        fitted_data = par._context_synthesizer.fit.mock_calls[0][1][0]
        expected_fitted_data = pd.DataFrame({
            'name': ['Doe', 'Jane', 'John'],
            'gender': ['M', 'F', 'M']
        })
        pd.testing.assert_frame_equal(fitted_data.sort_values(by='name'), expected_fitted_data)

    @patch('sdv.sequential.par.uuid')
    def test__fit_context_model_without_context_columns(self, uuid_mock):
        """Test that the method fits a synthesizer to a constant column.

        If there are no context columns, the method should create a constant column and
        group that by the sequence key. Then a synthesizer should be fit to this new data.
        """
        # Setup
        metadata = self.get_metadata()
        data = self.get_data()
        par = PARSynthesizer(metadata)
        par._context_synthesizer = Mock()
        uuid_mock.uuid4.return_value = 'abc'

        # Run
        par._fit_context_model(data)

        # Assert
        fitted_data = par._context_synthesizer.fit.mock_calls[0][1][0]
        expected_fitted_data = pd.DataFrame({
            'name': ['Doe', 'Jane', 'John'],
            'abc': [0, 0, 0]
        })
        pd.testing.assert_frame_equal(fitted_data.sort_values(by='name'), expected_fitted_data)

    @patch('sdv.sequential.par.PARModel')
    @patch('sdv.sequential.par.assemble_sequences')
    def test__fit_sequence_columns(self, assemble_sequences_mock, model_mock):
        """Test that the method assembles sequences properly and fits the ``PARModel`` to them.

        The method should use the ``assemble_sequences`` method to create a list of sequences
        that the model can fit to. It also needs to extract the data types for the context
        and non-context columns.
        """
        # Setup
        data = self.get_data()
        metadata = self.get_metadata()
        par = PARSynthesizer(
            metadata=metadata,
            context_columns=['gender']
        )
        sequences = [
            {'context': np.array(['M'], dtype=object), 'data': [[3], [65]]},
            {'context': np.array(['F'], dtype=object), 'data': [[1], [55]]},
            {'context': np.array(['M'], dtype=object), 'data': [[2], [60]]}
        ]
        assemble_sequences_mock.return_value = sequences

        # Run
        par._fit_sequence_columns(data)

        # Assert
        assemble_sequences_mock.assert_called_once_with(
            data,
            ['name'],
            ['gender'],
            None,
            None,
            drop_sequence_index=False
        )
        model_mock.assert_called_once_with(epochs=128, sample_size=1, cuda=True, verbose=False)
        model_mock.return_value.fit_sequences.assert_called_once_with(
            sequences,
            ['categorical'],
            ['continuous', 'continuous']
        )

    @patch('sdv.sequential.par.PARModel')
    @patch('sdv.sequential.par.assemble_sequences')
    def test__fit_sequence_columns_with_sequence_index(self, assemble_sequences_mock, model_mock):
        """Test the method when a sequence_index is present.

        If there is a sequence_index, the method should transform it by taking the sequence index
        and turning into to columns: one that is a list of diffs between each consecutive value in
        the sequence index, and another that is the starting value for the sequence index.
        """
        # Setup
        data = pd.DataFrame({
            'time': [1, 2, 3, 5, 8],
            'gender': ['F', 'F', 'M', 'M', 'M'],
            'name': ['Jane', 'Jane', 'John', 'John', 'John'],
            'measurement': [55, 60, 65, 65, 70]
        })
        metadata = self.get_metadata()
        metadata.set_sequence_index('time')
        par = PARSynthesizer(
            metadata=metadata,
            context_columns=['gender']
        )
        sequences = [
            {'context': np.array(['F'], dtype=object), 'data': [[1, 2], [55, 60]]},
            {'context': np.array(['M'], dtype=object), 'data': [[3, 5, 8], [65, 65, 70]]},
        ]
        assemble_sequences_mock.return_value = sequences

        # Run
        par._fit_sequence_columns(data)

        # Assert
        assemble_call_args_list = assemble_sequences_mock.call_args_list
        pd.testing.assert_frame_equal(assemble_call_args_list[0][0][0], data)
        assert assemble_call_args_list[0][0][1] == ['name']
        assert assemble_call_args_list[0][0][2] == ['gender']
        assert assemble_call_args_list[0][0][3] is None
        assert assemble_call_args_list[0][0][4] == 'time'
        assert assemble_call_args_list[0][1] == {'drop_sequence_index': False}
        expected_sequences = [
            {'context': np.array(['F'], dtype=object), 'data': [[1, 1], [55, 60], [1, 1]]},
            {
                'context': np.array(['M'], dtype=object),
                'data': [[2, 2, 3], [65, 65, 70], [3, 3, 3]]
            }
        ]
        model_mock.assert_called_once_with(epochs=128, sample_size=1, cuda=True, verbose=False)
        model_mock.return_value.fit_sequences.assert_called_once_with(
            expected_sequences,
            ['categorical'],
            ['continuous', 'continuous', 'continuous']
        )

    @patch('sdv.sequential.par.PARModel')
    @patch('sdv.sequential.par.assemble_sequences')
    def test__fit_sequence_columns_bad_dtype(self, assemble_sequences_mock, model_mock):
        """Test the method when a column has an unsupported dtype."""
        # Setup
        datetime = pd.Series(
            [pd.to_datetime('1/1/1999'), pd.to_datetime('1/2/1999'), '1/3/1999'],
            dtype='<M8[ns]')
        data = pd.DataFrame({
            'time': datetime,
            'gender': ['F', 'M', 'M'],
            'name': ['Jane', 'John', 'Doe'],
            'measurement': [55, 60, 65]
        })
        metadata = self.get_metadata()
        par = PARSynthesizer(
            metadata=metadata,
            context_columns=['gender']
        )

        # Run and Assert
        with pytest.raises(ValueError, match=re.escape('Unsupported dtype datetime64[ns]')):
            par._fit_sequence_columns(data)

    def test__fit_with_sequence_key(self):
        """Test that the method fits the context columns if there is a sequence key.

        When a sequence key is present, the context columns should be fitted before the rest of
        the columns.
        """
        # Setup
        metadata = self.get_metadata()
        par = PARSynthesizer(metadata=metadata)
        data = self.get_data()
        par._fit_context_model = Mock()
        par._fit_sequence_columns = Mock()

        # Run
        par._fit(data)

        # Assert
        par._fit_context_model.assert_called_once_with(data)
        par._fit_sequence_columns.assert_called_once_with(data)

    def test__fit_without_sequence_key(self):
        """Test that the method doesn't fit the context synthesizer if there are no sequence keys.

        If there are no sequence keys, then only the ``PARModel`` needs to be fit.
        """
        # Setup
        metadata = self.get_metadata(add_sequence_key=False)
        par = PARSynthesizer(metadata=metadata)
        data = self.get_data()
        par._fit_context_model = Mock()
        par._fit_sequence_columns = Mock()

        # Run
        par._fit(data)

        # Assert
        par._fit_context_model.assert_not_called()
        par._fit_sequence_columns.assert_called_once_with(data)

    @patch('sdv.sequential.par.tqdm')
    def test__sample_from_par(self, tqdm_mock):
        """Test that the method properly samples from the underlying ``PAR`` model.

        This method should sample from ``PAR``, set any context columns to be what was
        provided, set the sequence key to be what was provided and return the sampled
        sequences in a ``pandas.DataFrame``.
        """
        # Setup
        metadata = self.get_metadata(add_sequence_key=False)
        par = PARSynthesizer(metadata=metadata)
        model_mock = Mock()
        par._model = model_mock
        par._data_columns = ['time', 'gender', 'name', 'measurement']
        par._output_columns = ['time', 'gender', 'name', 'measurement']
        model_mock.sample_sequence.return_value = [
            [18000, 20000, 22000],
            [1, 1, 1],
            [.4, .7, .1],
            [55, 60, 65]
        ]
        context_columns = pd.DataFrame(index=range(1))
        tqdm_mock.tqdm.return_value = context_columns.iterrows()

        # Run
        sampled = par._sample_from_par(context_columns, 3)

        # Assert
        arg_list, kwargs = tqdm_mock.tqdm.call_args
        called_context_iterator_list = list(arg_list[0])
        assert kwargs['disable'] is True
        assert kwargs['total'] == 1
        for i, row in enumerate(context_columns.iterrows()):
            called_row = called_context_iterator_list[i]
            pd.testing.assert_series_equal(row[1], called_row[1])

        expected_output = pd.DataFrame({
            'time': [18000, 20000, 22000],
            'gender': [1, 1, 1],
            'name': [.4, .7, .1],
            'measurement': [55, 60, 65]
        })
        pd.testing.assert_frame_equal(sampled, expected_output)

    @patch('sdv.sequential.par.tqdm')
    def test__sample_from_par_with_sequence_key(self, tqdm_mock):
        """Test that the method handles the sequence key properly.

        If there are sequence keys, this method should set them as the index for the context
        columns. Both sequence keys and context columns should still be added back to the final
        returned data.
        """
        # Setup
        metadata = self.get_metadata()
        par = PARSynthesizer(
            metadata=metadata,
            context_columns=['gender']
        )
        model_mock = Mock()
        par._model = model_mock
        par._data_columns = ['time', 'measurement']
        par._output_columns = ['time', 'gender', 'name', 'measurement']
        model_mock.sample_sequence.return_value = [
            [18000, 20000, 22000],
            [55, 60, 65]
        ]
        context_columns = pd.DataFrame({'name': ['John'], 'gender': ['M']})
        tqdm_mock.tqdm.return_value = context_columns.set_index('name').iterrows()

        # Run
        sampled = par._sample_from_par(context_columns, 3)

        # Assert
        arg_list, kwargs = tqdm_mock.tqdm.call_args
        called_context_iterator_list = list(arg_list[0])
        assert kwargs['disable'] is True
        assert kwargs['total'] == 1
        for i, row in enumerate(context_columns.set_index('name').iterrows()):
            called_row = called_context_iterator_list[i]
            pd.testing.assert_series_equal(row[1], called_row[1])

        expected_output = pd.DataFrame({
            'time': [18000, 20000, 22000],
            'gender': ['M', 'M', 'M'],
            'name': ['John', 'John', 'John'],
            'measurement': [55, 60, 65]
        })
        pd.testing.assert_frame_equal(sampled, expected_output)

    @patch('sdv.sequential.par.tqdm')
    def test__sample_from_par_with_sequence_index(self, tqdm_mock):
        """Test that the method handles the sequence index properly.

        If there is a sequence index, then this method should recreate it from the
        sampled sequence. To do this, we have to mock the underlying model to return
        an extra column for the starting point of the sequence index.
        """
        # Setup
        metadata = self.get_metadata()
        metadata.set_sequence_index('time')
        par = PARSynthesizer(
            metadata=metadata,
            context_columns=['gender']
        )
        model_mock = Mock()
        par._model = model_mock
        par._data_columns = ['time', 'measurement']
        par._output_columns = ['time', 'gender', 'name', 'measurement']
        model_mock.sample_sequence.return_value = [
            [1000, 2000, 2000],
            [55, 60, 65],
            [18000, 18000, 18000]
        ]
        context_columns = pd.DataFrame({'name': ['John'], 'gender': ['M']})
        tqdm_mock.tqdm.return_value = context_columns.set_index('name').iterrows()

        # Run
        sampled = par._sample_from_par(context_columns, 3)

        # Assert
        expected_output = pd.DataFrame({
            'time': [18000, 20000, 22000],
            'gender': ['M', 'M', 'M'],
            'name': ['John', 'John', 'John'],
            'measurement': [55, 60, 65]
        })
        pd.testing.assert_frame_equal(sampled, expected_output, check_dtype=False)

    def test__sample(self):
        """This method should sample from par and reverse transform the data."""
        # Setup
        par = PARSynthesizer(metadata=self.get_metadata())
        par._sample_from_par = Mock()
        fake_sampled = pd.DataFrame()
        par._sample_from_par.return_value = fake_sampled
        par._data_processor = Mock()
        context_columns = pd.DataFrame({'gender': ['M']})

        # Run
        par._sample(context_columns=context_columns, sequence_length=5)

        # Assert
        par._sample_from_par.assert_called_once_with(context_columns, 5)
        par._data_processor.reverse_transform.assert_called_once_with(fake_sampled)

    def test_sample(self):
        """Test that the method samples the context columns and uses them to sample from PAR."""
        # Setup
        metadata = self.get_metadata()
        par = PARSynthesizer(
            metadata=metadata,
            context_columns=['gender']
        )
        par._context_synthesizer = Mock()
        context_columns = pd.DataFrame({
            'name': ['John', 'John', 'Jane'],
            'gender': ['M', 'M', 'F']
        })
        par._context_synthesizer._sample_with_progress_bar.return_value = context_columns
        par._sample = Mock()

        # Run
        par.sample(3, 2)

        # Assert
        par._context_synthesizer._sample_with_progress_bar.assert_called_once_with(
            3, output_file_path='disable', show_progress_bar=False)
        par._sample.assert_called_once_with(context_columns, 2)

    def test_sample_sequence_key_needs_to_be_filled_in(self):
        """Test that the method adds the sequence key to the context columns if necessary."""
        # Setup
        metadata = self.get_metadata()
        par = PARSynthesizer(
            metadata=metadata,
            context_columns=['gender']
        )
        par._context_synthesizer = Mock()
        context_columns = pd.DataFrame({
            'gender': ['M', 'M', 'F']
        })
        par._context_synthesizer._sample_with_progress_bar.return_value = context_columns
        par._sample = Mock()

        # Run
        par.sample(3, 2)

        # Assert
        par._context_synthesizer._sample_with_progress_bar.assert_called_once_with(
            3, output_file_path='disable', show_progress_bar=False)
        par._sample.assert_called_once_with(context_columns, 2)
        expected_context_columns = pd.DataFrame({
            'gender': ['M', 'M', 'F'],
            'name': [0, 1, 2]
        })
        pd.testing.assert_frame_equal(context_columns, expected_context_columns, check_dtype=False)

    def test_sample_no_sequence_key(self):
        """Test that if there is no sequence key, a column is made to substitute context."""
        # Setup
        metadata = self.get_metadata(add_sequence_key=False)
        par = PARSynthesizer(
            metadata=metadata
        )
        par._context_synthesizer = Mock()
        par._sample = Mock()

        # Run
        par.sample(3, 2)

        # Assert
        par._sample.assert_called_once_with(DataFrameMatcher(pd.DataFrame(index=range(3))), 2)

    def test_sample_sequential_columns(self):
        """Test that the method uses the provided context columns to sample."""
        # Setup
        par = PARSynthesizer(
            metadata=self.get_metadata(),
            context_columns=['gender']
        )
        par._sample = Mock()
        context_columns = pd.DataFrame({
            'gender': ['M', 'M', 'F']
        })

        # Run
        par.sample_sequential_columns(context_columns, 5)

        # Assert
        call_args, _ = par._sample.call_args
        pd.testing.assert_frame_equal(call_args[0], context_columns)
        assert call_args[1] == 5

    def test_sample_sequential_columns_no_context_columns(self):
        """Test that the method raises an error if the synthesizer has no context columns.

        If the synthesizer was not initialized with context columns, then this method cannot be
        used.
        """
        # Setup
        par = PARSynthesizer(metadata=self.get_metadata(add_sequence_key=False))
        par._sample = Mock()
        context_columns = pd.DataFrame({
            'gender': ['M', 'M', 'F']
        })

        # Run and Assert
        error_message = re.escape(
            "This synthesizer does not have any context columns. Please use 'sample()' "
            'to sample new sequences.'
        )
        with pytest.raises(SamplingError, match=error_message):
            par.sample_sequential_columns(context_columns, 5)

    @patch('sdv.single_table.base.cloudpickle')
    def test_save(self, cloudpickle_mock):
        """Test that the synthesizer is saved correctly."""
        # Setup
        synthesizer = Mock()

        # Run
        PARSynthesizer.save(synthesizer, 'output.pkl')

        # Assert
        cloudpickle_mock.dump.assert_called_once_with(synthesizer, ANY)

    @patch('sdv.single_table.base.cloudpickle')
    @patch('builtins.open', new_callable=mock_open)
    def test_load(self, mock_file, cloudpickle_mock):
        """Test that the ``load`` method loads a stored synthesizer."""
        # Setup
        synthesizer_mock = Mock()
        cloudpickle_mock.load.return_value = synthesizer_mock

        # Run
        loaded_instance = PARSynthesizer.load('synth.pkl')

        # Assert
        mock_file.assert_called_once_with('synth.pkl', 'rb')
        cloudpickle_mock.load.assert_called_once_with(mock_file.return_value)
        assert loaded_instance == synthesizer_mock
