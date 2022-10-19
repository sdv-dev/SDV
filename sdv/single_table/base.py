"""Base Synthesizer class."""

import inspect
import warnings

import pandas as pd

from sdv.data_processing.data_processor import DataProcessor
from sdv.errors import InvalidPreprocessingError
from sdv.single_table.errors import InvalidDataError
from sdv.utils import is_boolean_type, is_datetime_type, is_numerical_type


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

    _model_sdtype_transformers = None

    def _update_default_transformers(self):
        if self._model_sdtype_transformers is not None:
            for sdtype, transformer in self._model_sdtype_transformers.items():
                self._data_processor._update_transformers_by_sdtypes(sdtype, transformer)

    def __init__(self, metadata, enforce_min_max_values=True, enforce_rounding=True):
        self.metadata = metadata
        self.enforce_min_max_values = enforce_min_max_values
        self.enforce_rounding = enforce_rounding
        self._data_processor = DataProcessor(metadata)
        self._update_default_transformers()
        self._fitted = False

    def _validate_metadata_matches_data(self, columns):
        errors = []
        metadata_columns = self.metadata._columns or []
        missing_data_columns = set(columns).difference(metadata_columns)
        if missing_data_columns:
            errors.append(
                f'The columns {sorted(missing_data_columns)} are not present in the metadata.')

        missing_metadata_columns = set(metadata_columns).difference(columns)
        if missing_metadata_columns:
            errors.append(
                f'The metadata columns {sorted(missing_metadata_columns)} '
                'are not present in the data.'
            )

        if errors:
            raise InvalidDataError(errors)

    def _get_primary_and_alternate_keys(self):
        keys = set(self.metadata._alternate_keys)
        if self.metadata._primary_key:
            keys.update({self.metadata._primary_key})

        return keys

    def _get_set_of_sequence_keys(self):
        if isinstance(self.metadata._sequence_key, tuple):
            return set(self.metadata._sequence_key)

        if isinstance(self.metadata._sequence_key, str):
            return {self.metadata._sequence_key}

        return set()

    def _validate_keys_dont_have_missing_values(self, data):
        errors = []
        keys = self._get_primary_and_alternate_keys()
        keys.update(self._get_set_of_sequence_keys())
        for key in sorted(keys):
            if pd.isna(data[key]).any():
                errors.append(f"Key column '{key}' contains missing values.")

        return errors

    @staticmethod
    def _format_invalid_values_string(invalid_values):
        invalid_values = sorted(invalid_values, key=lambda x: str(x))
        if len(invalid_values) > 3:
            return invalid_values[:3] + [f'+ {len(invalid_values) - 3} more']

        return invalid_values

    def _validate_key_values_are_unique(self, data):
        errors = []
        keys = self._get_primary_and_alternate_keys()
        for key in sorted(keys):
            repeated_values = set(data[key][data[key].duplicated()])
            if repeated_values:
                repeated_values = self._format_invalid_values_string(repeated_values)
                errors.append(f"Key column '{key}' contains repeating values: {repeated_values}")

        return errors

    def _validate_context_columns(self, data):
        # NOTE: move method to PARSynthesizer when it has been implemented
        errors = []
        context_column_names = self._data_processor._model_kwargs.get('context_columns')
        if context_column_names:
            sequence_key_names = sorted(self._get_set_of_sequence_keys())
            for sequence_key, group in data.groupby(sequence_key_names):
                if len(group.groupby(context_column_names)) != 1:
                    invalid_context_cols = {}
                    for context_column_name in context_column_names:
                        values = set(group[context_column_name])
                        if len(values) != 1:
                            invalid_context_cols[context_column_name] = values

                    errors.append(
                        f'Context column(s) {invalid_context_cols} are changing '
                        f'inside the sequence keys ({sequence_key_names}: {sequence_key}).'
                    )

        return errors

    def _validate_sdtype(self, sdtype, column, validation):
        valid = column.apply(validation)
        invalid_values = set(column[~valid])
        if invalid_values:
            invalid_values = self._format_invalid_values_string(invalid_values)
            return [f"Invalid values found for {sdtype} column '{column.name}': {invalid_values}."]

        return []

    def _validate_column(self, column):
        """Validate values of the column satisfy its sdtype properties."""
        errors = []
        sdtype = self.metadata._columns[column.name]['sdtype']

        # boolean values must be True/False, None or missing values
        # int/str are not allowed
        if sdtype == 'boolean':
            errors += self._validate_sdtype(sdtype, column, is_boolean_type)

        # numerical values must be int/float, None or missing values
        # str/bool are not allowed
        if sdtype == 'numerical':
            errors += self._validate_sdtype(sdtype, column, is_numerical_type)

        # datetime values must be castable to datetime, None or missing values
        if sdtype == 'datetime':
            errors += self._validate_sdtype(
                sdtype, column, lambda x: pd.isna(x) | is_datetime_type(x))

        return errors

    def validate(self, data):
        """Validate data.

        Args:
            data (pd.DataFrame):
                The data to validate.

        Raises:
            ValueError:
                Raised when data is not of type pd.DataFrame.
            InvalidDataError:
                Raised if:
                    * data columns don't match metadata
                    * keys have missing values
                    * primary or alternate keys are not unique
                    * context columns vary for a sequence key
                    * values of a column don't satisfy their sdtype
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f'Data must be a DataFrame, not a {type(data)}.')

        # Both metadata and data must have the same set of columns
        self._validate_metadata_matches_data(data.columns)

        errors = []
        # Primary, sequence and alternate keys can't have missing values
        errors += self._validate_keys_dont_have_missing_values(data)

        # Primary and alternate key values must be unique
        errors += self._validate_key_values_are_unique(data)

        # Context column values must be the same for each tuple of sequence keys
        errors += self._validate_context_columns(data)

        # Every column must satisfy the properties of their sdtypes
        for column in data:
            errors += self._validate_column(data[column])

        if errors:
            raise InvalidDataError(errors)

    def _validate_transformers(self, column_name_to_transformer):
        keys = self._get_primary_and_alternate_keys() | self._get_set_of_sequence_keys()
        for column, transformer in column_name_to_transformer.items():
            if column in keys and not transformer.is_generator():
                raise InvalidPreprocessingError(
                    f"Column '{column}' is a key. It cannot be preprocessed using "
                    f"the '{type(transformer).__name__}' transformer."
                )

            # If columns were set, the transformer was fitted
            if transformer.columns:
                raise InvalidPreprocessingError(
                    f"Transformer for column '{column}' has already been fit on data.")

    def _warn_for_update_transformers(self, column_name_to_transformer):
        """Raise warnings for update_transformers.

        Args:
            column_name_to_transformer (dict):
                Dict mapping column names to transformers to be used for that column.
        """
        for column in column_name_to_transformer:
            sdtype = self.metadata._columns[column]['sdtype']
            if sdtype in {'categorical', 'boolean'}:
                warnings.warn(
                    f"Replacing the default transformer for column '{column}' "
                    'might impact the quality of your synthetic data.'
                )

    def update_transformers(self, column_name_to_transformer):
        """Update any of the transformers assigned to each of the column names.

        Args:
            column_name_to_transformer (dict):
                Dict mapping column names to transformers to be used for that column.
        """
        self._validate_transformers(column_name_to_transformer)
        self._warn_for_update_transformers(column_name_to_transformer)
        self._data_processor.update_transformers(column_name_to_transformer)

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
        self.validate(data)
        if self._fitted:
            warnings.warn(
                'This model has already been fitted. To use the new preprocessed data, '
                "please refit the model using 'fit' or 'fit_processed_data'."
            )

        self._data_processor.fit(data)
        return self._data_processor.transform(data)

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
        # Reset fit status
        self._fitted = False
        self._fit(processed_data)
        self._fitted = True

    def fit(self, data):
        """Fit this model to the original data.

        Args:
            data (pandas.DataFrame):
                The raw data (before any transformations) to fit the model to.
        """
        processed_data = self.preprocess(data)
        self.fit_processed_data(processed_data)
