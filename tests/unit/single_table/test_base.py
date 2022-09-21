from sdv.single_table.base import BaseSynthesizer


class TestBaseSynthesizer:

    @patch('sdv.single_table.base.DataProcessor')
    def test___init__(self, mocK_data_processor):
        """Test that instantiating parameters are properly stored in the instance."""
        # Run
        instance = BaseSynthesizer(metadata)

        # Assert
        assert instance.enforce_min_max_values
        assert instance.enforce_rounding
        assert instance._data_processor == mocK_data_processor.return_value
        mocK_data_processor.assert_called_once_with(metadata)
