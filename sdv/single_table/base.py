"""Base Synthesizer class."""

import inspect
import pandas as pd
from sdv.constraints.utils import cast_to_datetime64

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
    
    @staticmethod
    def _update_invalid_values(invalid_values):
        if len(invalid_values) > 3:
            return invalid_values[:3] + [f'+ {len(invalid_values) - 3} more']
        return invalid_values
    
    @staticmethod
    def _castable_to_datetime(value):
        try:
            # TODO: this can't cast None values.
            # Also, it converts integers and the string 10-10-10-10...
            pd.to_datetime(value).to_datetime64().astype('datetime64[ns]')
            return True
        except:
            return False
        
    def validate(self, data):
        """Validate stuff."""
        errors = []
        for column in data:
            sdtype = self.metadata._columns[column]['sdtype']
            if sdtype == 'numerical':
                # What do we mean by number? Are booleans number? Is a string of a number? Is None a missing value?
                valid = data[column].apply(lambda x: pd.isna(x) | pd.api.types.is_float(x) | pd.api.types.is_integer(x))
                invalid_values = list(set(data[column][~valid]))
                if invalid_values:
                    invalid_values = self._update_invalid_values(invalid_values)
                    errors.append(
                        f"Invalid values found for numerical column '{column}': {invalid_values}"
                    )
            
            if sdtype == 'datetime':
                valid = data[column].apply(self._castable_to_datetime)
                invalid_values = list(set(data[column][~valid]))
                if invalid_values:
                    invalid_values = self._update_invalid_values(invalid_values)
                    errors.append(
                        f"Invalid values found for datetime column '{column}': {invalid_values}"
                    )

        if errors:
            print(errors)
            raise ValueError



        # boolean is True, False, None: Invalid values found for boolean column 'is_subscribed': (0.0, 30.0, 4.4, +more)

        # pk, fk, alternat_key, sequence_key cant have missing: Key column 'user_id' contains missing values

        # pk, alternate_key should have unique values: Primary key column 'user_id' contains repeating values: ('UID_000', 'UID_001', 'UID_002', +more)

        # Context column 'patient_address' is changing inside sequence ('Patient_ID'='ID_004').
