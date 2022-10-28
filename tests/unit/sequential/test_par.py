import pandas as pd
from unittest.mock import patch, Mock

from sdv.data_processing.data_processor import DataProcessor
from sdv.metadata.single_table import SingleTableMetadata
from sdv.sequential.par import PARSynthesizer
from sdv.single_table.copulas import GaussianCopulaSynthesizer


class TestPARSynthesizer:

    def get_metadata(self):
        metadata = SingleTableMetadata()
        metadata.add_column('time', sdtype='datetime')
        metadata.add_column('gender', sdtype='categorical')
        metadata.add_column('name', sdtype='text')
        metadata.add_column('measurement', sdtype='numerical')
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
        assert synthesizer._context_synthesizer.metadata._columns == {
            'gender': {'sdtype': 'categorical'},
            'name': {'sdtype': 'text'}
        }

    def test_get_parameters(self):
        """Test that it returns every ``init`` parameter without the ``metadata``."""
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
        parameters = instance.get_parameters()

        # Assert
        assert 'metadata' not in parameters
        assert parameters == {
            'enforce_min_max_values': True,
            'enforce_rounding': True,
            'context_columns': None,
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

    @patch('sdv.sequential.par.BaseSynthesizer.preprocess')
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

    @patch('sdv.sequential.par.BaseSynthesizer.preprocess')
    def test_preprocess(self, base_preprocess_mock):
        """Test that the method does not auto assign the transformers if it's already been done.

        To test this, we set the hyper transformer to have its ``field_transformers`` set.
        """
        # Setup
        metadata = self.get_metadata()
        par = PARSynthesizer(
            metadata=metadata
        )
        par.auto_assign_transformers = Mock()
        par.update_transformers = Mock()
        par._data_processor._hyper_transformer.field_transformers = {
            'time': None,
            'gender': None,
            'name': None,
            'measurement': None
        }
        data = self.get_data()

        # Run
        par.preprocess(data)

        # Assert
        expected_transformers = {'name': None}
        par.auto_assign_transformers.assert_not_called()
        par.update_transformers.assert_called_once_with(expected_transformers)
        base_preprocess_mock.assert_called_once_with(data)

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
            'name': [ 'Doe','Jane', 'John'],
            'gender': ['M', 'F', 'M']
        })
        pd.testing.assert_frame_equal(fitted_data.sort_values(by='name'), expected_fitted_data)
