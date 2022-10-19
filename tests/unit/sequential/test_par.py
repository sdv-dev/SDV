from sdv.data_processing.data_processor import DataProcessor
from sdv.metadata.single_table import SingleTableMetadata
from sdv.sequential.par import PARSynthesizer
from sdv.single_table.copulas import GaussianCopulaSynthesizer


class TestPARSynthesizer:

    def test___init__(self):
        """Test that the parameters are set correctly.

        The parameters passed in the ``__init__`` should be set on the instance. Additionally,
        a context synthesizer should be created with the correct metadata and parameters.
        """
        # Setup
        metadata = SingleTableMetadata()
        metadata.add_column('time', sdtype='datetime')
        metadata.add_column('gender', sdtype='categorical')
        metadata.add_column('name', sdtype='text')
        metadata.add_column('measurement', sdtype='numerical')

        # Run
        synthesizer = PARSynthesizer(
            metadata=metadata,
            enforce_min_max_values=True,
            enforce_rounding=True,
            context_columns=['gender', 'name'],
            segment_size=10,
            epochs=10,
            sample_size=5,
            cuda=False,
            verbose=False
        )

        # Assert
        assert synthesizer.context_columns == ['gender', 'name']
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
