from collections import defaultdict
from unittest.mock import Mock, call

from sdv.multi_table.base import BaseMultiTableSynthesizer
from sdv.single_table.copulas import GaussianCopulaSynthesizer
from tests.utils import get_multi_table_metadata


class TestBaseMultiTableSynthesizer:

    def test__initialize_models(self):
        """Test that this method initializes the ``self._synthezier`` for each table.

        This tests that we iterate over the tables within the metadata and if there are
        ``table_parameters`` we use those to initialize the model.
        """
        # Setup
        instance = Mock()
        instance._table_synthesizers = {}
        instance._table_parameters = {
            'nesreca': {
                'default_distribution': 'gamma'
            }
        }
        instance.metadata = get_multi_table_metadata()

        # Run
        BaseMultiTableSynthesizer._initialize_models(instance)

        # Assert
        assert instance._table_synthesizers == {
            'nesreca': instance._synthesizer.return_value,
            'oseba': instance._synthesizer.return_value,
            'upravna_enota': instance._synthesizer.return_value
        }
        instance._synthesizer.assert_has_calls([
            call(metadata=instance.metadata._tables['nesreca'], default_distribution='gamma'),
            call(metadata=instance.metadata._tables['oseba']),
            call(metadata=instance.metadata._tables['upravna_enota'])
        ])

    def test___init__(self):
        """Test that when creating a new instance this sets the defaults.

        Test that the metadata object is being stored and also being validated. Afterwards, this
        calls the ``self._initialize_models`` which creates the initial instances of those.
        """
        # Setup
        metadata = get_multi_table_metadata()
        metadata.validate = Mock()
        # Run
        instance = BaseMultiTableSynthesizer(metadata)

        # Assert
        assert instance.metadata == metadata
        assert isinstance(instance._table_synthesizers['nesreca'], GaussianCopulaSynthesizer)
        assert isinstance(instance._table_synthesizers['oseba'], GaussianCopulaSynthesizer)
        assert isinstance(instance._table_synthesizers['upravna_enota'], GaussianCopulaSynthesizer)
        assert instance._table_parameters == defaultdict(dict)
        instance.metadata.validate.assert_called_once_with()

    def test_get_table_parameters_empty(self):
        """Test that this method returns an empty dictionary when there are no parameters."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)

        # Run
        result = instance.get_table_parameters('oseba')

        # Assert
        assert result == {}

    def test_get_table_parameters_has_parameters(self):
        """Test that this method returns a dictionary with the parameters."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)
        instance._table_parameters['oseba'] = {'default_distribution': 'gamma'}

        # Run
        result = instance.get_table_parameters('oseba')

        # Assert
        assert result == {'default_distribution': 'gamma'}

    def test_get_parameters(self):
        """Test that the table's synthesizer parameters are being returned."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)

        # Run
        result = instance.get_parameters('oseba')

        # Assert
        assert result == {
            'default_distribution': 'beta',
            'enforce_min_max_values': True,
            'enforce_rounding': True,
            'numerical_distributions': {}
        }

    def test_update_table_parameters(self):
        """Test that the table's parameters are being updated.

        This test should ensure that the ``self._table_parameters`` for the given table
        are being updated with the given parameters, and also that the model is re-created
        and updated it's parameters as well.
        """
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)

        # Run
        instance.update_table_parameters('oseba', {'default_distribution': 'gamma'})

        # Assert
        assert instance._table_parameters['oseba'] == {'default_distribution': 'gamma'}
        assert instance.get_parameters('oseba') == {
            'default_distribution': 'gamma',
            'enforce_min_max_values': True,
            'enforce_rounding': True,
            'numerical_distributions': {}
        }

    def test_get_metadata(self):
        """Test that the metadata object is returned."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)

        # Run
        result = instance.get_metadata()

        # Assert
        assert metadata == result
