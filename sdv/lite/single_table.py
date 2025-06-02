"""Base class for single table model presets."""

import inspect
import logging
import sys
import warnings

import cloudpickle

from sdv.metadata.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

LOGGER = logging.getLogger(__name__)

FAST_ML_PRESET = 'FAST_ML'
PRESETS = {
    FAST_ML_PRESET: 'Use this preset to minimize the time needed to create a synthetic data model.'
}
DEPRECATION_MSG = (
    "The 'SingleTablePreset' is deprecated. For equivalent Fast ML "
    "functionality, please use the 'GaussianCopulaSynthesizer'."
)

META_DEPRECATION_MSG = (
    "The 'SingleTableMetadata' is deprecated. Please use the new 'Metadata' class for synthesizers."
)


class SingleTablePreset:
    """Class for all single table synthesizer presets.

    Args:
        metadata (sdv.metadata.Metadata):
            ``Metadata`` instance.
            * sdv.metadata.SingleTableMetadata can be used but will be deprecated.
        name (str):
            The preset to use.
        locales (list or str):
            The default locale(s) to use for AnonymizedFaker transformers.
            Defaults to ``['en_US']``.
    """

    _synthesizer = None
    _default_synthesizer = GaussianCopulaSynthesizer

    def _setup_fast_preset(self, metadata, locales):
        self._synthesizer = GaussianCopulaSynthesizer(
            metadata=metadata, default_distribution='norm', enforce_rounding=False, locales=locales
        )

    def __init__(self, metadata, name, locales=['en_US']):
        warnings.warn(DEPRECATION_MSG, FutureWarning)
        self.locales = locales
        if name not in PRESETS:
            raise ValueError(f"'name' must be one of {PRESETS}.")

        self.name = name
        if isinstance(metadata, Metadata):
            metadata = metadata._convert_to_single_table()
        else:
            warnings.warn(META_DEPRECATION_MSG, FutureWarning)
        if name == FAST_ML_PRESET:
            self._setup_fast_preset(metadata, self.locales)

    def get_metadata(self):
        """Return the ``Metadata`` for this synthesizer."""
        warnings.warn(DEPRECATION_MSG, FutureWarning)
        return self._synthesizer.get_metadata()

    def get_parameters(self):
        """Return the parameters used to instantiate the synthesizer."""
        warnings.warn(DEPRECATION_MSG, FutureWarning)
        parameters = inspect.signature(self.__init__).parameters
        instantiated_parameters = {}
        for parameter_name in parameters:
            if parameter_name != 'metadata':
                instantiated_parameters[parameter_name] = self.__dict__.get(parameter_name)

        return instantiated_parameters

    def fit(self, data):
        """Fit this model to the data.

        Args:
            data (pandas.DataFrame):
                Data to fit the model to.
        """
        warnings.warn(DEPRECATION_MSG, FutureWarning)
        self._synthesizer.fit(data)

    def sample(self, num_rows, max_tries_per_batch=100, batch_size=None, output_file_path=None):
        """Sample rows from this table.

        Args:
            num_rows (int):
                Number of rows to sample. This parameter is required.
            max_tries_per_batch (int):
                Number of times to try sampling discarded rows. Defaults to 100.
            batch_size (int or None):
                The batch size to sample. Defaults to `num_rows`, if None.
            output_file_path (str or None):
                The file to periodically write sampled rows to. If None, does not
                write rows anywhere.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        warnings.warn(DEPRECATION_MSG, FutureWarning)
        sampled = self._synthesizer.sample(
            num_rows,
            max_tries_per_batch,
            batch_size,
            output_file_path,
        )

        return sampled

    def sample_from_conditions(
        self, conditions, max_tries_per_batch=100, batch_size=None, output_file_path=None
    ):
        """Sample rows from this table with the given conditions.

        Args:
            conditions (list[sdv.sampling.Condition]):
                A list of sdv.sampling.Condition objects, which specify the column
                values in a condition, along with the number of rows for that
                condition.
            max_tries_per_batch (int):
                Number of times to try sampling discarded rows. Defaults to 100.
            batch_size (int):
                The batch size to use per attempt at sampling. Defaults to 10 times
                the number of rows.
            output_file_path (str or None):
                The file to periodically write sampled rows to. Defaults to None.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        warnings.warn(DEPRECATION_MSG, FutureWarning)
        sampled = self._synthesizer.sample_from_conditions(
            conditions, max_tries_per_batch, batch_size, output_file_path
        )

        return sampled

    def sample_remaining_columns(
        self, known_columns, max_tries_per_batch=100, batch_size=None, output_file_path=None
    ):
        """Sample rows from this table.

        Args:
            known_columns (pandas.DataFrame):
                A pandas.DataFrame with the columns that are already known. The output
                is a DataFrame such that each row in the output is sampled
                conditionally on the corresponding row in the input.
            max_tries_per_batch (int):
                Number of times to try sampling discarded rows. Defaults to 100.
            batch_size (int):
                The batch size to use per attempt at sampling. Defaults to 10 times
                the number of rows.
            output_file_path (str or None):
                The file to periodically write sampled rows to. Defaults to None.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        warnings.warn(DEPRECATION_MSG, FutureWarning)
        sampled = self._synthesizer.sample_remaining_columns(
            known_columns, max_tries_per_batch, batch_size, output_file_path
        )

        return sampled

    def save(self, filepath):
        """Save this model instance to the given path using cloudpickle.

        Args:
            filepath (str):
                Path where the SDV instance will be serialized.
        """
        warnings.warn(DEPRECATION_MSG, FutureWarning)
        with open(filepath, 'wb') as output:
            cloudpickle.dump(self, output)

    @classmethod
    def load(cls, filepath):
        """Load a SingleTableSynthesizer instance from a given path.

        Args:
            filepath (str):
                Path from which to load the instance.

        Returns:
            SingleTableSynthesizer:
                The loaded synthesizer.
        """
        warnings.warn(DEPRECATION_MSG, FutureWarning)
        with open(filepath, 'rb') as f:
            model = cloudpickle.load(f)
            return model

    @classmethod
    def list_available_presets(cls, out=sys.stdout):
        """List the available presets and their descriptions."""
        warnings.warn(DEPRECATION_MSG, FutureWarning)
        out.write(
            f'Available presets:\n{PRESETS}\n\n'
            'Supply the desired preset using the `name` parameter.\n\n'
            'Have any requests for custom presets? Contact the SDV team to learn '
            'more an SDV Premium license.\n'
        )

    def __repr__(self):
        """Represent single table preset instance as text.

        Returns:
            str
        """
        return f'SingleTablePreset(name={self.name})'
