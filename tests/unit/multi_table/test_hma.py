from unittest.mock import Mock

from sdv.multi_table.hma import HMASynthesizer
from sdv.single_table.copulas import GaussianCopulaSynthesizer
from tests.utils import get_multi_table_metadata


class TestHMASynthesizer:

    def test___init__(self):
        # Run
        metadata = get_multi_table_metadata()
        metadata.validate = Mock()
        instance = HMASynthesizer(metadata)

        # Assert
        assert instance.metadata == metadata
        assert isinstance(instance._table_synthesizers['nesreca'], GaussianCopulaSynthesizer)
        assert isinstance(instance._table_synthesizers['oseba'], GaussianCopulaSynthesizer)
        assert isinstance(instance._table_synthesizers['upravna_enota'], GaussianCopulaSynthesizer)
        assert instance._table_parameters == {
            'nesreca': {'default_distribution': 'norm'},
            'oseba': {'default_distribution': 'norm'},
            'upravna_enota': {'default_distribution': 'norm'},
        }
        instance.metadata.validate.assert_called_once_with()
