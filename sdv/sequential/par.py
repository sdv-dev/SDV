"""PAR Synthesizer class."""

import inspect

from sdv.data_processing import DataProcessor
from sdv.metadata.single_table import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer


class PARSynthesizer:
    """Synthesizer for sequential data.

    This synthesizer uses the ``deepecho.models.par.PARModel`` class as the core model.
    Additionally, it uses a separate synthesizer to model and sample the context columns
    to be passed into PAR.

    Args:
        metadata (sdv.metadata.SingleTableMetadata):
            Single table metadata representing the data that this synthesizer will be used for.
        enforce_min_max_values (bool):
            Specify whether or not to clip the data returned by ``reverse_transform`` of
            the numerical transformer, ``FloatFormatter``, to the min and max values seen
            during ``fit``. Defaults to ``True``.
        enforce_rounding (bool):
            Define rounding scheme for ``numerical`` columns. If ``True``, the data returned
            by ``reverse_transform`` will be rounded as in the original data. Defaults to ``True``.
        context_columns (list[str]):
            A list of strings, representing the columns that do not vary in a sequence.
        segment_size (int):
            If specified, cut each training sequence in several segments of
            the indicated size. The size can be passed as an integer
            value, which will interpreted as the number of data points to
            put on each segment.
        epochs (int):
            The number of epochs to train for. Defaults to 128.
        sample_size (int):
            The number of times to sample (before choosing and
            returning the sample which maximizes the likelihood).
            Defaults to 1.
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
        verbose (bool):
            Whether to print progress to console or not.
    """

    def _get_context_metadata(self):
        context_columns_dict = {}
        context_columns = self.context_columns or []
        for column in context_columns:
            context_columns_dict[column] = self.metadata._columns[column]

        context_metadata_dict = {'columns': context_columns_dict}
        return SingleTableMetadata._load_from_dict(context_metadata_dict)

    def __init__(self, metadata, enforce_min_max_values, enforce_rounding, context_columns=None,
                 segment_size=None, epochs=128, sample_size=1, cuda=True, verbose=False):
        self.metadata = metadata
        self.enforce_min_max_values = enforce_min_max_values
        self.enforce_rounding = enforce_rounding
        self._data_processor = DataProcessor(metadata)
        self.context_columns = context_columns
        self.segment_size = segment_size
        self._model_kwargs = {
            'epochs': epochs,
            'sample_size': sample_size,
            'cuda': cuda,
            'verbose': verbose,
        }
        context_metadata = self._get_context_metadata()
        self._context_synthesizer = GaussianCopulaSynthesizer(
            metadata=context_metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding
        )

    def get_parameters(self):
        """Return the parameters used to instantiate the synthesizer."""
        parameters = inspect.signature(self.__init__).parameters
        instantiated_parameters = {}
        for parameter_name in parameters:
            if parameter_name != 'metadata':
                instantiated_parameters[parameter_name] = self.__dict__.get(parameter_name)

        for parameter_name, value in self._model_kwargs.items():
            instantiated_parameters[parameter_name] = value

        return instantiated_parameters

    def get_metadata(self):
        """Return the ``SingleTableMetadata`` for this synthesizer."""
        return self.metadata
