"""Base Synthesizer class."""

import inspect

from sdv.data_processing.data_processor import DataProcessor


class BaseSynthesizer:
    """Base class for all ``Synthesizers``.

    The ``BaseSynthesizer`` class defines the common API that all the
    ``Synthesizers`` need to implement, as well as common functionality.

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
    """

    def __init__(self, metadata, enforce_min_max_values=True, enforce_rounding=True):
        self.metadata = metadata
        self.enforce_min_max_values = enforce_min_max_values
        self.enforce_rounding = enforce_rounding
        self._data_processor = DataProcessor(metadata)
        self._fitted = False

    def get_parameters(self):
        """Return the parameters used to instantiate the synthesizer."""
        parameters = inspect.signature(self.__init__).parameters
        instantiated_parameters = {}
        for parameter_name in parameters:
            if parameter_name != 'metadata':
                instantiated_parameters[parameter_name] = self.__dict__.get(parameter_name)

        return instantiated_parameters

    def get_metadata(self):
        """Return the ``SingleTableMetadata`` for this synthesizer."""
        return self.metadata

    def preprocess(self, data):
        """Transform the raw data to numerical space."""
        pass

    def _fit(self, processed_data):
        """Fit the model to the table.

        Args:
            processed_data (pandas.DataFrame):
                Data to be learned.
        """
        raise NotImplementedError()

    def fit_processed_data(self, processed_data):
        """Fit this model to the transformed data.

        Args:
            processed_data (pandas.DataFrame):
                The transformed data used to fit the model to.
        """
        self._fit(processed_data)

    def fit(self, data):
        """Fit this model to the original data.

        Args:
            data (pandas.DataFrame):
                The raw data (before any transformations) to fit the model to.
        """
        processed_data = self.preprocess(data)
        self.fit_processed_data(processed_data)