import re
from unittest.mock import ANY, Mock, mock_open, patch

import numpy as np
import pandas as pd
import pytest
from rdt.transformers import FloatFormatter, UnixTimestampEncoder

from sdv.data_processing.data_processor import DataProcessor
from sdv.data_processing.errors import InvalidConstraintsError
from sdv.errors import InvalidDataError, NotFittedError, SamplingError, SynthesizerInputError
from sdv.metadata.single_table import SingleTableMetadata
from sdv.sampling import Condition
from sdv.sequential.par import PARSynthesizer
from sdv.single_table.copulas import GaussianCopulaSynthesizer


class TestPARSynthesizer:
    def get_metadata(self, add_sequence_key=True, add_sequence_index=False):
        metadata = SingleTableMetadata()
        metadata.add_column('time', sdtype='datetime')
        metadata.add_column('gender', sdtype='categorical')
        metadata.add_column('name', sdtype='id')
        metadata.add_column('measurement', sdtype='numerical')
        if add_sequence_key:
            metadata.set_sequence_key('name')

        if add_sequence_index:
            metadata.set_sequence_index('time')

        return metadata

    def get_data(self):
        data = pd.DataFrame({
            'time': ['2020-01-01', '2020-01-02', '2020-01-03'],
            'gender': ['F', 'M', 'M'],
            'name': ['Jane', 'John', 'Doe'],
            'measurement': [55, 60, 65],
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
            verbose=False,
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
            'verbose': False,
        }
        assert isinstance(synthesizer._data_processor, DataProcessor)
        assert synthesizer._data_processor.metadata == metadata
        assert isinstance(synthesizer._context_synthesizer, GaussianCopulaSynthesizer)
        assert synthesizer._context_synthesizer.metadata.columns == {
            'gender': {'sdtype': 'categorical'},
            'name': {'sdtype': 'id'},
        }

    def test___init___no_sequence_key(self):
        """Test when there are no sequence keys.

        If there are no sequence keys then an error should be raised.
        """
        # Setup
        metadata = self.get_metadata(add_sequence_key=False)

        # Run and Assert
        error_message = (
            'The PARSythesizer is designed for multi-sequence data, identifiable through a '
            'sequence key. Your metadata does not include a sequence key.'
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
                verbose=False,
            )

    def test_add_constraints(self):
        """Test that that only simple constraints can be added to PARSynthesizer."""
        # Setup
        metadata = self.get_metadata(add_sequence_key=True)
        synthesizer = PARSynthesizer(metadata=metadata, context_columns=['gender', 'measurement'])
        measurement_constraint = {
            'constraint_class': 'Mock',
            'constraint_parameters': {'column_name': 'measurement'},
        }
        gender_constraint = {
            'constraint_class': 'Mock',
            'constraint_parameters': {'column_name': 'gender'},
        }
        time_constraint = {
            'constraint_class': 'Mock',
            'constraint_parameters': {'column_name': 'time'},
        }
        multi_constraint = {
            'constraint_class': 'Mock',
            'constraint_parameters': {'column_names': ['gender', 'time']},
        }
        overlapping_error_msg = re.escape(
            'The PARSynthesizer cannot accommodate multiple constraints '
            'that overlap on the same columns.'
        )
        mixed_constraint_error_msg = re.escape(
            'The PARSynthesizer cannot accommodate constraints '
            'with a mix of context and non-context columns.'
        )

        # Run and Assert
        with pytest.raises(SynthesizerInputError, match=mixed_constraint_error_msg):
            synthesizer.add_constraints([time_constraint, gender_constraint])

        with pytest.raises(SynthesizerInputError, match=mixed_constraint_error_msg):
            synthesizer.add_constraints([time_constraint, measurement_constraint])

        with pytest.raises(SynthesizerInputError, match=mixed_constraint_error_msg):
            synthesizer.add_constraints([multi_constraint])

        with pytest.raises(SynthesizerInputError, match=overlapping_error_msg):
            synthesizer.add_constraints([multi_constraint, time_constraint])

        with pytest.raises(SynthesizerInputError, match=overlapping_error_msg):
            synthesizer.add_constraints([multi_constraint, multi_constraint])

        with pytest.raises(SynthesizerInputError, match=overlapping_error_msg):
            synthesizer.add_constraints([gender_constraint, gender_constraint])

        # Custom constraint will not be found
        with pytest.raises(InvalidConstraintsError):
            synthesizer.add_constraints([gender_constraint])

    def test_load_custom_constraint_classes(self):
        """Test that if custom constraint is being added, an error is raised."""
        # Setup
        metadata = self.get_metadata()
        synthesizer = PARSynthesizer(metadata=metadata)

        # Run and Assert
        error_message = re.escape('The PARSynthesizer cannot accommodate custom constraints.')
        with pytest.raises(SynthesizerInputError, match=error_message):
            synthesizer.load_custom_constraint_classes(filepath='test', class_names=[])

    def test_add_custom_constraint_class(self):
        """Test that if custom constraint is being added, an error is raised."""
        # Setup
        metadata = self.get_metadata()
        synthesizer = PARSynthesizer(metadata=metadata)

        # Run and Assert
        error_message = re.escape('The PARSynthesizer cannot accommodate custom constraints.')
        with pytest.raises(SynthesizerInputError, match=error_message):
            synthesizer.add_custom_constraint_class(Mock(), class_name='Mock')

    def test_get_parameters(self):
        """Test that it returns every ``init`` parameter without the ``metadata``."""
        # Setup
        metadata = self.get_metadata()
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
            verbose=False,
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
            'verbose': False,
        }

    def test_get_metadata(self):
        """Test that it returns the ``metadata`` object."""
        # Setup
        metadata = self.get_metadata()
        instance = PARSynthesizer(
            metadata=metadata,
            enforce_min_max_values=True,
            enforce_rounding=True,
            context_columns=None,
            segment_size=10,
            epochs=10,
            sample_size=5,
            cuda=False,
            verbose=False,
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
        metadata.set_sequence_key('sk_col1')
        instance = PARSynthesizer(metadata=metadata, context_columns=['ct_col1', 'ct_col2'])

        # Run and Assert
        err_msg = re.escape(
            'The provided data does not match the metadata:'
            "\nContext column 'ct_col1' is changing inside "
            "sequence (['sk_col1']=1)."
            '\n'
            "\nContext column 'ct_col1' is changing inside "
            "sequence (['sk_col1']=2)."
            '\n'
            "\nContext column 'ct_col2' is changing inside "
            "sequence (['sk_col1']=2)."
        )
        with pytest.raises(InvalidDataError, match=err_msg):
            instance.validate(data)

    def test__transform_sequence(self):
        # Setup
        metadata = self.get_metadata(add_sequence_index=True)
        par = PARSynthesizer(metadata=metadata)
        data = pd.DataFrame({
            'time': [1, 2, 4, 5],
            'gender': ['F', 'M', 'M', 'M'],
            'name': ['Jane', 'John', 'John', 'John'],
            'measurement': [55, 60, 65, 68],
        })

        # Run
        transformed_data = par._transform_sequence_index(data)

        # Assert
        expected = pd.DataFrame({
            'time': [1.0, 2.0, 2.0, 1.0],
            'gender': ['F', 'M', 'M', 'M'],
            'name': ['Jane', 'John', 'John', 'John'],
            'measurement': [55, 60, 65, 68],
            'time.context': [1, 2, 2, 2],
        })
        pd.testing.assert_frame_equal(transformed_data, expected)
        assert par._extra_context_columns == {'time.context': {'sdtype': 'numerical'}}
        assert list(par.extended_columns.keys()) == ['time']
        assert par.extended_columns['time'].enforce_min_max_values is True

    def test__transform_sequence_index_single_instances(self):
        # Setup
        metadata = self.get_metadata(add_sequence_index=True)
        par = PARSynthesizer(metadata=metadata)
        data = self.get_data()

        # Run
        transformed_data = par._transform_sequence_index(data)

        # Assert
        expected = pd.DataFrame({
            'time': [0.0, 0.0, 0.0],
            'gender': ['F', 'M', 'M'],
            'name': ['Jane', 'John', 'Doe'],
            'measurement': [55, 60, 65],
            'time.context': ['2020-01-01', '2020-01-02', '2020-01-03'],
        })
        pd.testing.assert_frame_equal(transformed_data, expected)
        assert par._extra_context_columns == {'time.context': {'sdtype': 'numerical'}}
        assert list(par.extended_columns.keys()) == ['time']
        assert par.extended_columns['time'].enforce_min_max_values is True

    def test__transform_sequence_index_non_unique_sequence_key(self):
        # Setup
        metadata = self.get_metadata(add_sequence_index=True)
        par = PARSynthesizer(metadata=metadata)
        data = self.get_data()
        data = data[data['name'] == 'John'].reset_index(drop=True)

        # Run
        transformed_data = par._transform_sequence_index(data)

        # Assert
        expected = pd.DataFrame({
            'time': [0.0],
            'gender': ['M'],
            'name': ['John'],
            'measurement': [60],
            'time.context': ['2020-01-02'],
        })
        pd.testing.assert_frame_equal(transformed_data, expected)
        assert par._extra_context_columns == {'time.context': {'sdtype': 'numerical'}}
        assert list(par.extended_columns.keys()) == ['time']
        assert par.extended_columns['time'].enforce_min_max_values is True

    @patch('sdv.sequential.par.BaseSynthesizer._preprocess')
    def test_preprocess_transformers_not_assigned(self, base_preprocess_mock):
        """Test that the method auto assigns the transformers if not already done.

        If the transformers in the ``DataProcessor`` haven't been assigned, then this method
        should do it so that it can overwrite the transformers for all the sequence key columns.
        """
        # Setup
        metadata = self.get_metadata()
        par = PARSynthesizer(metadata=metadata)
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
        metadata = self.get_metadata(add_sequence_index=True)
        par = PARSynthesizer(metadata=metadata)
        par._transform_sequence_index = Mock()
        par.auto_assign_transformers = Mock()
        par.update_transformers = Mock()
        get_transform_mock = Mock()
        get_transform_mock.return_value = {'time': Mock()}
        par.get_transformers = get_transform_mock
        par._data_processor._prepared_for_fitting = True
        data = self.get_data()

        # Run
        par.preprocess(data)

        # Assert
        expected_transformers = {'name': None}
        par.auto_assign_transformers.assert_not_called()
        par.update_transformers.assert_called_once_with(expected_transformers)
        base_preprocess_mock.assert_called_once_with(data)
        par._transform_sequence_index.assert_called_once_with(base_preprocess_mock.return_value)

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

    @patch('sdv.sequential.par.GaussianCopulaSynthesizer')
    def test__fit_context_model_with_context_columns(self, gaussian_copula_mock):
        """Test that the method fits a synthesizer to the context columns.

        If there are context columns, the method should create a new DataFrame that groups
        the data by the sequence_key and only contains the context columns. Then a synthesizer
        should be fit to this new data.
        """
        # Setup
        metadata = self.get_metadata()
        data = self.get_data()
        par = PARSynthesizer(metadata, context_columns=['gender'])
        initial_synthesizer = Mock()
        context_metadata = SingleTableMetadata.load_from_dict({
            'columns': {'gender': {'sdtype': 'categorical'}, 'name': {'sdtype': 'id'}}
        })
        par._context_synthesizer = initial_synthesizer
        par._get_context_metadata = Mock()
        par._get_context_metadata.return_value = context_metadata

        # Run
        par._fit_context_model(data)

        # Assert
        gaussian_copula_mock.assert_called_with(
            context_metadata,
            enforce_min_max_values=initial_synthesizer.enforce_min_max_values,
            enforce_rounding=initial_synthesizer.enforce_rounding,
        )
        fitted_data = gaussian_copula_mock().fit.mock_calls[0][1][0]
        expected_fitted_data = pd.DataFrame({
            'name': ['Doe', 'Jane', 'John'],
            'gender': ['M', 'F', 'M'],
        })
        pd.testing.assert_frame_equal(fitted_data.sort_values(by='name'), expected_fitted_data)

    @patch('sdv.sequential.par.PARSynthesizer.get_transformers')
    def test_auto_assign_transformers_without_enforce_min_max(self, mock_get_transfomers):
        """Test to see if auto_assign_transformers does not add enforce_min_max_values
        if the transformer does not contain it already
        """
        # Setup
        datetime = pd.Series(
            [pd.to_datetime('1/1/1999'), pd.to_datetime('1/2/1999'), '1/3/1999'], dtype='<M8[ns]'
        )
        data = pd.DataFrame({
            'time': datetime,
            'gender': ['F', 'F', 'M'],
            'name': ['Jane', 'Jane', 'John'],
            'measurement': [55, 60, 65],
        })
        metadata = self.get_metadata()
        metadata.set_sequence_index('time')
        mock_get_transfomers.return_value = {'time': FloatFormatter}

        # Run
        par = PARSynthesizer(metadata=metadata, context_columns=['gender'])
        par.auto_assign_transformers(data)

        # Assert
        assert (
            hasattr(par.get_transformers()[par._sequence_index], 'enforce_min_max_values') is False
        )

    @patch('sdv.sequential.par.GaussianCopulaSynthesizer')
    @patch('sdv.sequential.par.uuid')
    def test__fit_context_model_without_context_columns(self, uuid_mock, gaussian_copula_mock):
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
        expected_fitted_data = pd.DataFrame({'name': ['Doe', 'Jane', 'John'], 'abc': [0, 0, 0]})
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
        par = PARSynthesizer(metadata=metadata, context_columns=['gender'])
        sequences = [
            {'context': np.array(['M'], dtype=object), 'data': [['2020-01-03'], [65]]},
            {'context': np.array(['F'], dtype=object), 'data': [['2020-01-01'], [55]]},
            {'context': np.array(['M'], dtype=object), 'data': [['2020-01-02'], [60]]},
        ]
        assemble_sequences_mock.return_value = sequences

        # Run
        par._fit_sequence_columns(data)

        # Assert
        assemble_sequences_mock.assert_called_once_with(
            data, ['name'], ['gender'], None, None, drop_sequence_index=False
        )
        model_mock.assert_called_once_with(epochs=128, sample_size=1, cuda=True, verbose=False)
        model_mock.return_value.fit_sequences.assert_called_once_with(
            sequences, ['categorical'], ['categorical', 'continuous']
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
            'measurement': [55, 60, 65, 65, 70],
        })
        metadata = self.get_metadata()
        metadata.set_sequence_index('time')
        par = PARSynthesizer(metadata=metadata, context_columns=['gender'])
        sequences = [
            {'context': np.array(['F'], dtype=object), 'data': [[1, 1], [55, 60], [1, 1]]},
            {
                'context': np.array(['M'], dtype=object),
                'data': [[2, 2, 3], [65, 65, 70], [3, 3, 3]],
            },
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
                'data': [[2, 2, 3], [65, 65, 70], [3, 3, 3]],
            },
        ]
        model_mock.assert_called_once_with(epochs=128, sample_size=1, cuda=True, verbose=False)
        model_mock.return_value.fit_sequences.assert_called_once_with(
            expected_sequences, ['categorical'], ['continuous', 'continuous']
        )

    @patch('sdv.sequential.par.PARModel')
    @patch('sdv.sequential.par.assemble_sequences')
    def test__fit_sequence_columns_bad_dtype(self, assemble_sequences_mock, model_mock):
        """Test the method when a column has an unsupported dtype."""
        # Setup
        datetime = pd.Series(
            [pd.to_datetime('1/1/1999'), pd.to_datetime('1/2/1999'), '1/3/1999'], dtype='<M8[ns]'
        )
        data = pd.DataFrame({
            'time': datetime,
            'gender': ['F', 'M', 'M'],
            'name': ['Jane', 'John', 'Doe'],
            'measurement': [55, 60, 65],
        })
        metadata = self.get_metadata()
        par = PARSynthesizer(metadata=metadata, context_columns=['gender'])

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

    def test_get_loss_values(self):
        """Test the ``get_loss_values`` method from ``PARSynthesizer."""
        # Setup
        mock_model = Mock()
        loss_values = pd.DataFrame({'Epoch': [0, 1, 2], 'Loss': [0.8, 0.6, 0.5]})
        mock_model.loss_values = loss_values
        metadata = self.get_metadata()
        instance = PARSynthesizer(metadata)
        instance._model = mock_model
        instance._fitted = True

        # Run
        actual_loss_values = instance.get_loss_values()

        # Assert
        pd.testing.assert_frame_equal(actual_loss_values, loss_values)

    def test_get_loss_values_error(self):
        """Test the ``get_loss_values`` errors if synthesizer has not been fitted."""
        # Setup
        metadata = self.get_metadata()
        instance = PARSynthesizer(metadata)

        # Run / Assert
        msg = 'Loss values are not available yet. Please fit your synthesizer first.'
        with pytest.raises(NotFittedError, match=msg):
            instance.get_loss_values()

    @patch('sdv.sequential.par.tqdm')
    def test__sample_from_par(self, tqdm_mock):
        """Test that the method properly samples from the underlying ``PAR`` model.

        This method should sample from ``PAR``, set any context columns to be what was
        provided, set the sequence key to be what was provided and return the sampled
        sequences in a ``pandas.DataFrame``.
        """
        # Setup
        metadata = self.get_metadata()
        par = PARSynthesizer(metadata=metadata)
        model_mock = Mock()
        par._model = model_mock
        par._data_columns = ['time', 'gender', 'measurement']
        par._output_columns = ['time', 'gender', 'name', 'measurement']
        model_mock.sample_sequence.return_value = [[18000, 20000, 22000], [1, 1, 1], [55, 60, 65]]
        context_columns = pd.DataFrame({'name': ['John Doe']})
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
            'gender': [1, 1, 1],
            'name': ['John Doe', 'John Doe', 'John Doe'],
            'measurement': [55, 60, 65],
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
        par = PARSynthesizer(metadata=metadata, context_columns=['gender'])
        model_mock = Mock()
        par._model = model_mock
        par._data_columns = ['time', 'measurement']
        par._output_columns = ['time', 'gender', 'name', 'measurement']
        model_mock.sample_sequence.return_value = [[18000, 20000, 22000], [55, 60, 65]]
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
            'measurement': [55, 60, 65],
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
        par = PARSynthesizer(metadata=metadata, context_columns=['gender'])
        model_mock = Mock()
        par._model = model_mock
        mock_transformer = Mock()
        mock_transformer.reverse_transform.return_value = pd.DataFrame({'time': [1000, 2000, 2000]})
        par.extended_columns = {'time': mock_transformer}
        par._data_columns = ['time', 'measurement']
        par._output_columns = ['time', 'gender', 'name', 'measurement']
        par._extra_context_columns = {'time.context': {'sdtype': 'numerical'}}
        model_mock.sample_sequence.return_value = [[1000, 2000, 2000], [55, 60, 65]]
        context_columns = pd.DataFrame({'name': ['John'], 'gender': ['M'], 'time.context': [18000]})
        tqdm_mock.tqdm.return_value = context_columns.set_index('name').iterrows()

        # Run
        sampled = par._sample_from_par(context_columns, 3)

        # Assert
        expected_output = pd.DataFrame({
            'time': [18000, 20000, 22000],
            'gender': ['M', 'M', 'M'],
            'name': ['John', 'John', 'John'],
            'measurement': [55, 60, 65],
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
        par = PARSynthesizer(metadata=metadata, context_columns=['gender'])
        par._context_synthesizer = Mock()
        context_columns = pd.DataFrame({
            'name': ['John', 'John', 'Jane'],
            'gender': ['M', 'M', 'F'],
        })
        par._context_synthesizer._sample_with_progress_bar.return_value = context_columns
        par._sample = Mock()

        # Run
        par.sample(3, 2)

        # Assert
        par._context_synthesizer._sample_with_progress_bar.assert_called_once_with(
            3, output_file_path='disable', show_progress_bar=False
        )
        par._sample.assert_called_once_with(context_columns, 2)

    def test_sample_sequence_key_needs_to_be_filled_in(self):
        """Test that the method adds the sequence key to the context columns if necessary."""
        # Setup
        metadata = self.get_metadata()
        par = PARSynthesizer(metadata=metadata, context_columns=['gender'])
        par._context_synthesizer = Mock()
        context_columns = pd.DataFrame({'gender': ['M', 'M', 'F']})
        par._context_synthesizer._sample_with_progress_bar.return_value = context_columns
        par._sample = Mock()

        # Run
        par.sample(3, 2)

        # Assert
        par._context_synthesizer._sample_with_progress_bar.assert_called_once_with(
            3, output_file_path='disable', show_progress_bar=False
        )
        par._sample.assert_called_once_with(context_columns, 2)
        expected_context_columns = pd.DataFrame({'gender': ['M', 'M', 'F'], 'name': [0, 1, 2]})
        pd.testing.assert_frame_equal(context_columns, expected_context_columns, check_dtype=False)

    def test_sample_sequential_columns(self):
        """Test that the method uses the provided context columns to sample."""
        # Setup
        par = PARSynthesizer(metadata=self.get_metadata(), context_columns=['gender'])
        par._context_synthesizer = Mock()
        par._context_synthesizer._model.columns = ['gender', 'extra_col']
        par._context_synthesizer.sample_from_conditions.return_value = pd.DataFrame({
            'id_col': ['A', 'A', 'A'],
            'gender': ['M', 'M', 'F'],
            'extra_col': [0, 1, 1],
        })
        par._sample = Mock()
        context_columns = pd.DataFrame({
            'id_col': ['ID-1', 'ID-2', 'ID-3'],
            'gender': ['M', 'M', 'F'],
        })

        # Run
        par.sample_sequential_columns(context_columns, 5)

        # Assert
        call_args, _ = par._context_synthesizer.sample_from_conditions.call_args
        expected_conditions = [
            Condition({'gender': 'M'}),
            Condition({'gender': 'M'}),
            Condition({'gender': 'F'}),
        ]
        assert len(call_args[0]) == len(expected_conditions)
        for arg, expected in zip(call_args[0], expected_conditions):
            assert arg.column_values == expected.column_values
            assert arg.num_rows == expected.num_rows

        expected_call_arg = pd.DataFrame({
            'id_col': ['ID-1', 'ID-2', 'ID-3'],
            'gender': ['M', 'M', 'F'],
            'extra_col': [0, 1, 1],
        })
        call_args, _ = par._sample.call_args
        pd.testing.assert_frame_equal(call_args[0], expected_call_arg)
        assert call_args[1] == 5

    def test_sample_sequential_columns_no_context_columns(self):
        """Test that the method raises an error if the synthesizer has no context columns.

        If the synthesizer was not initialized with context columns, then this method cannot be
        used.
        """
        # Setup
        par = PARSynthesizer(metadata=self.get_metadata())
        par._sample = Mock()
        context_columns = pd.DataFrame({'gender': ['M', 'M', 'F']})

        # Run and Assert
        error_message = re.escape(
            "This synthesizer does not have any context columns. Please use 'sample()' "
            'to sample new sequences.'
        )
        with pytest.raises(SamplingError, match=error_message):
            par.sample_sequential_columns(context_columns, 5)

    @patch('sdv.single_table.base.cloudpickle')
    def test_save(self, cloudpickle_mock, tmp_path):
        """Test that the synthesizer is saved correctly."""
        # Setup
        synthesizer = Mock()

        # Run
        filepath = tmp_path / 'output.pkl'
        PARSynthesizer.save(synthesizer, filepath)

        # Assert
        cloudpickle_mock.dump.assert_called_once_with(synthesizer, ANY)

    @patch('sdv.single_table.base.cloudpickle')
    @patch('builtins.open', new_callable=mock_open)
    def test_load(self, mock_file, cloudpickle_mock):
        """Test that the ``load`` method loads a stored synthesizer."""
        # Setup
        synthesizer_mock = Mock(
            _fitted_sdv_version=None,
            _fitted_sdv_enterprise_version=None,
        )
        cloudpickle_mock.load.return_value = synthesizer_mock

        # Run
        loaded_instance = PARSynthesizer.load('synth.pkl')

        # Assert
        mock_file.assert_called_once_with('synth.pkl', 'rb')
        cloudpickle_mock.load.assert_called_once_with(mock_file.return_value)
        assert loaded_instance == synthesizer_mock

    def test__par_error_on_context_columns(self):
        metadata = self.get_metadata(add_sequence_key=True)
        sequence_key_context_column_error_msg = re.escape(
            "The sequence key ['name'] cannot be a context column. "
            'To proceed, please remove the sequence key from the context_columns parameter.'
        )
        with pytest.raises(SynthesizerInputError, match=sequence_key_context_column_error_msg):
            PARSynthesizer(
                metadata=metadata,
                context_columns=['name'],
            )
