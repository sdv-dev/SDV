from unittest.mock import Mock, patch

from sdv.single_table.base import BaseSynthesizer


class TestBaseSynthesizer:

    @patch('sdv.single_table.base.DataProcessor')
    def test___init__(self, mock_data_processor):
        """Test instantiating with default values."""
        # Setup
        metadata = Mock()

        # Run
        instance = BaseSynthesizer(metadata)

        # Assert
        assert instance.enforce_min_max_values is True
        assert instance.enforce_rounding is True
        assert instance._data_processor == mock_data_processor.return_value
        mock_data_processor.assert_called_once_with(metadata)

    @patch('sdv.single_table.base.DataProcessor')
    def test___init__custom(self, mock_data_processor):
        """Test that instantiating with custom parameters are properly stored in the instance."""
        # Setup
        metadata = Mock()

        # Run
        instance = BaseSynthesizer(metadata, enforce_min_max_values=False, enforce_rounding=False)

        # Assert
        assert instance.enforce_min_max_values is False
        assert instance.enforce_rounding is False
        assert instance._data_processor == mock_data_processor.return_value
        mock_data_processor.assert_called_once_with(metadata)

    @patch('sdv.single_table.base.DataProcessor')
    def test_get_parameters(self, mock_data_processor):
        """Test that returns every ``init`` parameter without the ``metadata``."""
        # Setup
        metadata = Mock()
        instance = BaseSynthesizer(metadata, enforce_min_max_values=False, enforce_rounding=False)

        # Run
        parameters = instance.get_parameters()

        # Assert
        assert 'metadata' not in parameters
        assert parameters == {'enforce_min_max_values': False, 'enforce_rounding': False}

    @patch('sdv.single_table.base.DataProcessor')
    def test_get_metadata(self, mock_data_processor):
        """Test that returns the ``metadata`` object."""
        # Setup
        metadata = Mock()
        instance = BaseSynthesizer(metadata, enforce_min_max_values=False, enforce_rounding=False)

        # Run
        result = instance.get_metadata()

        # Assert
        assert result == metadata
