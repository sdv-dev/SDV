"""Hierarchical Modeling Algorithms."""

from sdv.single_table.copulas import GaussianCopulaSynthesizer
from sdv.multi_table.base import BaseMultiTableSynthesizer



class HMASynthesizer(BaseMultiTableSynthesizer):
    """Hierarchical Modeling Algorithm One.

    Args:
        metadata (dict, str or Metadata):
            Metadata dict, path to the metadata JSON file or Metadata instance itself.
    """

    DEFAULT_SYNTHESIZER_KWARGS = {
        'default_distribution': 'gaussian',
    }

    def __init__(self, metadata, synthesizer_kwargs=None):
        super().__init__(metadata)
        self._synthesizer_kwargs = model_kwargs or self.DEFAULT_SYNTHESIZER_KWARGS
        self._table_sizes = {}
        self._max_child_rows = {}
        for table_name in self.metadata._tables:
            self.update_table_parameters(table_name, self._synthesizer_kwargs)
