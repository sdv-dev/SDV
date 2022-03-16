"""Base class for tabular model presets."""

import logging
import sys
import warnings

from sdv.tabular import GaussianCopula
from sdv.tabular.base import BaseTabularModel

LOGGER = logging.getLogger(__name__)

SPEED_PRESET = 'SPEED'
PRESETS = {
    SPEED_PRESET: 'Use this preset to minimize the time needed to create a synthetic data model.',
}


class TabularPreset(BaseTabularModel):
    """Class for all tabular model presets.

    Args:
        optimize_for (str):
            The preset to use.
        metadata (dict or metadata.Table):
            Table metadata instance or dict representation.
    """

    _model = None

    def __init__(self, optimize_for=None, metadata=None):
        if optimize_for is None:
            raise ValueError('You must provide the name of a preset using the `optimize_for` '
                             'parameter. Use `TabularPreset.list_available_presets()` to browse '
                             'through the options.')
        if optimize_for not in PRESETS:
            raise ValueError(f'`optimize_for` must be one of {PRESETS}.')
        if metadata is None:
            warnings.warn('No metadata provided. Metadata will be automatically '
                          'detected from your data. This process may not be accurate. '
                          'We recommend writing metadata to ensure correct data handling.')

        if optimize_for == SPEED_PRESET:
            self._model = GaussianCopula(
                table_metadata=metadata,
                categorical_transformer='categorical',
                default_distribution='gaussian',
                rounding=None,
            )
            print('This config optimizes the modeling speed above all else.\n\n'
                  'Your exact runtime is dependent on the data. Benchmarks:\n'
                  '100K rows and 100 columns may take around 1 minute.\n'
                  '1M rows and 250 columns may take around 30 minutes.')

    def fit(self, data):
        """Fit this model to the data."""
        self._model.fit(data)

    def sample(self, num_rows):
        """Sample rows from this table."""
        return self._model.sample(num_rows)

    @classmethod
    def list_available_presets(cls, out=sys.stdout):
        """List the available presets and their descriptions."""
        out.write(f'Available presets:\n{PRESETS}\n\n'
                  'Supply the desired preset using the `opimize_for` parameter.\n\n'
                  'Have any requests for custom presets? Contact the SDV team to learn '
                  'more an SDV Premium license.\n')
