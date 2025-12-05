"""Base Synthesizer class."""

import datetime
import functools
import inspect
import logging
import math
import operator
import os
import uuid
import warnings
from collections import defaultdict
from copy import deepcopy

import cloudpickle
import copulas
import numpy as np
import pandas as pd
import tqdm
from copulas.multivariate import GaussianMultivariate
from pandas.api.types import is_float_dtype

from sdv import version
from sdv._utils import (
    _check_regex_format,
    _groupby_list,
    _validate_correct_synthesizer_loading,
    check_sdv_versions_and_warn,
    check_synthesizer_version,
    generate_synthesizer_id,
    warn_load_deprecated,
)
from sdv.cag._errors import ConstraintNotMetError
from sdv.cag._utils import (
    _convert_to_snake_case,
    _get_invalid_rows,
    _validate_constraints_single_table,
)
from sdv.cag.programmable_constraint import ProgrammableConstraint, ProgrammableConstraintHarness
from sdv.data_processing.data_processor import DataProcessor
from sdv.errors import (
    ConstraintsNotMetError,
    InvalidDataError,
    RefitWarning,
    SamplingError,
    SynthesizerInputError,
)
from sdv.logging import get_sdv_logger
from sdv.metadata.errors import InvalidMetadataError
from sdv.metadata.metadata import Metadata
from sdv.metadata.single_table import SingleTableMetadata
from sdv.sampling import Condition, DataFrameCondition
from sdv.single_table.utils import check_num_rows, handle_sampling_error, validate_file_path

LOGGER = logging.getLogger(__name__)

SYNTHESIZER_LOGGER = get_sdv_logger('SingleTableSynthesizer')

COND_IDX = str(uuid.uuid4())
FIXED_RNG_SEED = 73251
DEPRECATION_MSG = (
    "The 'SingleTableMetadata' is deprecated. Please use the new 'Metadata' class for synthesizers."
)


class BaseSynthesizer:
    """Base class for all ``Synthesizers``.

    The ``BaseSynthesizer`` class defines the common API that all the
    ``Synthesizers`` need to implement, as well as common functionality.

    Args:
        metadata (sdv.metadata.Metadata):
            Single table metadata representing the data that this synthesizer will be used for.
            * sdv.metadata.SingleTableMetadata can be used but will be deprecated.
        enforce_min_max_values (bool):
            Specify whether or not to clip the data returned by ``reverse_transform`` of
            the numerical transformer, ``FloatFormatter``, to the min and max values seen
            during ``fit``. Defaults to ``True``.
        enforce_rounding (bool):
            Define rounding scheme for ``numerical`` columns. If ``True``, the data returned
            by ``reverse_transform`` will be rounded as in the original data. Defaults to ``True``.
        locales (list or str):
            The default locale(s) to use for AnonymizedFaker transformers.
            Defaults to ``['en_US']``.

    """

    _model_sdtype_transformers = None
    _model = None

    def _validate_inputs(self, enforce_min_max_values, enforce_rounding):
        if not isinstance(enforce_min_max_values, bool):
            raise SynthesizerInputError(
                f"Invalid value '{enforce_min_max_values}' for parameter 'enforce_min_max_values'."
                ' Please provide True or False.'
            )

        if not isinstance(enforce_rounding, bool):
            raise SynthesizerInputError(
                f"Invalid value '{enforce_rounding}' for parameter 'enforce_rounding'."
                ' Please provide True or False.'
            )

    def _update_default_transformers(self):
        if self._model_sdtype_transformers is not None:
            for sdtype, transformer in self._model_sdtype_transformers.items():
                self._data_processor._update_transformers_by_sdtypes(sdtype, transformer)

    def _check_input_metadata_updated(self):
        if not hasattr(self, '_input_metadata'):
            if hasattr(self, '_original_metadata'):
                unified_metadata = Metadata.load_from_dict(self._original_metadata.to_dict())
            else:
                unified_metadata = Metadata.load_from_dict(self.metadata.to_dict())

            setattr(self, '_input_metadata', unified_metadata)

        if isinstance(self._input_metadata, Metadata):
            metadata = self._input_metadata._convert_to_single_table()

        else:
            metadata = self._input_metadata

        if metadata._updated:
            warnings.warn(
                'Your metadata has been modified. Metadata modifications cannot be applied to an '
                'existing synthesizer. Please create a new synthesizer with the modified metadata.'
            )

    def _check_metadata_updated(self):
        if self.metadata._check_updated_flag():
            self.metadata._reset_updated_flag()
            warnings.warn(
                "We strongly recommend saving the metadata using 'save_to_json' for replicability"
                ' in future SDV versions.'
            )
            if hasattr(self, '_input_metadata'):
                if hasattr(self._input_metadata, '_reset_updated_flag'):
                    self._input_metadata._reset_updated_flag()
                else:
                    self._input_metadata._updated = False

    def _validate_regex_format(self):
        if self.metadata.tables:
            id_columns = self.metadata.get_column_names(table_name=self._table_name, sdtype='id')
            for column_name in id_columns:
                regex = (
                    self.metadata.tables[self._table_name].columns[column_name].get('regex_format')
                )
                _check_regex_format(self._table_name, column_name, regex)

    def __init__(
        self, metadata, enforce_min_max_values=True, enforce_rounding=True, locales=['en_US']
    ):
        self._validate_inputs(enforce_min_max_values, enforce_rounding)

        # Points to the input metadata object and allows us to check if user has changed it
        self._input_metadata = metadata

        # Points to a dynamic metadata object that could be modified by constraints
        self.metadata = metadata

        self._table_name = Metadata.DEFAULT_SINGLE_TABLE_NAME
        if isinstance(metadata, Metadata):
            self._table_name = metadata._get_single_table_name()
        else:
            warnings.warn(DEPRECATION_MSG, FutureWarning)
            self._table_name = Metadata.DEFAULT_SINGLE_TABLE_NAME
            self.metadata = Metadata.load_from_dict(metadata.to_dict(), self._table_name)
            self.metadata.tables[self._table_name]._updated = metadata._updated

        self.metadata.validate()
        self._check_metadata_updated()

        # Points to a metadata object that conserves the initialized status of the synthesizer
        self._original_metadata = deepcopy(self.metadata)

        self.enforce_min_max_values = enforce_min_max_values
        self.enforce_rounding = enforce_rounding
        self.locales = locales
        self._data_processor = DataProcessor(
            metadata=self.metadata._convert_to_single_table(),
            enforce_rounding=self.enforce_rounding,
            enforce_min_max_values=self.enforce_min_max_values,
            locales=self.locales,
        )
        self._validate_regex_format()
        self._original_columns = pd.Index([])
        self._fitted = False
        self._random_state_set = False
        self._update_default_transformers()
        self._creation_date = datetime.datetime.today().strftime('%Y-%m-%d')
        self._fitted_date = None
        self._fitted_sdv_version = None
        self._fitted_sdv_enterprise_version = None
        self._chained_constraints = []  # chain of constraints used to preprocess the data
        self._reject_sampling_constraints = []  # constraints used only for reject sampling
        self._constraints_fitted = False
        self._synthesizer_id = generate_synthesizer_id(self)
        SYNTHESIZER_LOGGER.info({
            'EVENT': 'Instance',
            'TIMESTAMP': datetime.datetime.now(),
            'SYNTHESIZER CLASS NAME': self.__class__.__name__,
            'SYNTHESIZER ID': self._synthesizer_id,
        })

    def set_address_columns(self, column_names, anonymization_level='full'):
        """Set the address multi-column transformer."""
        warnings.warn(
            '`set_address_columns` is deprecated. Please add these columns directly to your'
            ' metadata using `add_column_relationship`.',
            FutureWarning,
        )

    def _validate_metadata(self, data):
        """Validate that the data follows the metadata."""
        errors = []
        try:
            if isinstance(self.metadata, Metadata):
                self.metadata.validate_data({self._table_name: data})
            else:
                self.metadata.validate_data(data)
        except InvalidDataError as error:
            errors += error.errors

        if errors:
            raise InvalidDataError(errors)

    def _validate(self, data):
        """Validate any rules that only apply to specific synthesizers.

        This method should be overridden by subclasses.
        """
        return []

    def _get_table_metadata(self):
        if isinstance(self.metadata, Metadata):
            return self.metadata.tables.get(self._table_name, SingleTableMetadata())

        return self.metadata

    def _validate_transformers(self, column_name_to_transformer):
        primary_and_alternate_keys = self._get_table_metadata()._get_primary_and_alternate_keys()
        sequence_keys = self._get_table_metadata()._get_set_of_sequence_keys()
        keys = primary_and_alternate_keys | sequence_keys
        for column, transformer in column_name_to_transformer.items():
            if transformer is None:
                continue

            if column in keys and not transformer.is_generator():
                raise SynthesizerInputError(
                    f"Column '{column}' is a key. It cannot be preprocessed using "
                    f"the '{type(transformer).__name__}' transformer."
                )

            # If columns were set, the transformer was fitted
            if transformer.columns:
                raise SynthesizerInputError(
                    f"Transformer for column '{column}' has already been fit on data."
                )

    def _warn_quality_and_performance(self, column_name_to_transformer):
        """Raise warning if the quality/performance may be impacted.

        Args:
            column_name_to_transformer (dict):
                Dict mapping column names to transformers to be used for that column.
        """
        table_metadata = self._get_table_metadata()
        for column in column_name_to_transformer:
            sdtype = table_metadata.columns.get(column, {}).get('sdtype')
            if sdtype in {'categorical', 'boolean'}:
                warnings.warn(
                    f"Replacing the default transformer for column '{column}' "
                    'might impact the quality of your synthetic data.'
                )

    def _warn_unable_to_enforce_rounding(self, column_name_to_transformer):
        if self.enforce_rounding:
            invalid_columns = []
            for column, transformer in column_name_to_transformer.items():
                if (
                    hasattr(transformer, 'learn_rounding_scheme')
                    and not transformer.learn_rounding_scheme
                ):
                    invalid_columns.append(column)

            if invalid_columns:
                warnings.warn(
                    f'Unable to turn off rounding scheme for column(s) {invalid_columns}, '
                    'because the overall synthesizer is enforcing rounding. We '
                    "recommend setting the synthesizer's 'enforce_rounding' "
                    'parameter to False.'
                )

    def update_transformers(self, column_name_to_transformer):
        """Update any of the transformers assigned to each of the column names.

        Args:
            column_name_to_transformer (dict):
                Dict mapping column names to transformers to be used for that column.
        """
        self._validate_transformers(column_name_to_transformer)
        self._warn_quality_and_performance(column_name_to_transformer)
        self._warn_unable_to_enforce_rounding(column_name_to_transformer)
        self._data_processor.update_transformers(column_name_to_transformer)
        if self._fitted:
            msg = 'For this change to take effect, please refit the synthesizer using `fit`.'
            warnings.warn(msg, RefitWarning)

    def _resolve_gpu_parameters(self, parameters):
        if parameters.get('cuda') is not None and parameters.get('enable_gpu') is None:
            parameters.pop('enable_gpu', None)  # Ensure backward-compatibility
        elif 'cuda' in parameters:  # Removed because deprecated
            del parameters['cuda']

        return parameters

    def get_parameters(self):
        """Return the parameters used to instantiate the synthesizer."""
        parameters = inspect.signature(self.__init__).parameters
        instantiated_parameters = {}
        for parameter_name in parameters:
            if parameter_name not in ['metadata']:
                instantiated_parameters[parameter_name] = self.__dict__.get(parameter_name)

        return self._resolve_gpu_parameters(instantiated_parameters)

    def get_metadata(self, version='original'):
        """Get the metadata, either original or modified after applying constraints.

        Args:
            version (str, optional):
                The version of metadata to return, must be one of 'original' or 'modified'. If
                'original', will return the original metadata used to instantiate the
                synthesizer. If 'modified', will return the modified metadata after applying this
                synthesizer's constraints. Defaults to 'original'.
        """
        if version not in ('original', 'modified'):
            raise ValueError(
                f"Unrecognized version '{version}', please use 'original' or 'modified'."
            )

        table_name = getattr(self, '_table_name', None)
        if hasattr(self, '_original_metadata') and version == 'original':
            return Metadata.load_from_dict(self._original_metadata.to_dict(), table_name)

        return Metadata.load_from_dict(self.metadata.to_dict(), table_name)

    def load_custom_constraint_classes(self, filepath, class_names):
        """Load a custom constraint class for the current synthesizer.

        Args:
            filepath (str):
                String representing the absolute or relative path to the python file where
                the custom constraints are declared.
            class_names (list):
                A list of custom constraint classes to be imported.
        """
        self._data_processor.load_custom_constraint_classes(filepath, class_names)

    def auto_assign_transformers(self, data):
        """Automatically assign the required transformers for the given data and constraints.

        This method will automatically set a configuration to the ``rdt.HyperTransformer``
        with the required transformers for the current data.

        Args:
            data (pandas.DataFrame):
                The raw data (before any transformations) that will be used to fit the model.
        """
        self.validate(data)
        data = self._validate_transform_constraints(data)
        self._data_processor.prepare_for_fitting(data)

    def get_transformers(self):
        """Get a dictionary mapping of ``column_name``  and ``rdt.transformers``.

        A dictionary representing the column names and the transformers that will be used
        to transform the data.

        Returns:
            dict:
                A dictionary mapping with column names and transformers.
        """
        field_transformers = self._data_processor._hyper_transformer.field_transformers
        if field_transformers == {}:
            raise ValueError(
                "No transformers were returned in 'get_transformers'. "
                "Use 'auto_assign_transformers' or 'fit' to create them."
            )

        # Order the output to match metadata
        table_metadata = self._get_table_metadata()
        ordered_field_transformers = {
            column_name: field_transformers.get(column_name)
            for column_name in table_metadata.columns
            if column_name in field_transformers
        }

        # Add missing columns created by the constraints
        ordered_field_transformers.update(field_transformers)

        return ordered_field_transformers

    def get_info(self):
        """Get dictionary with information regarding the synthesizer.

        Return:
            dict:
                * ``class_name``: synthesizer class name.
                * ``creation_date``: date of creation.
                * ``is_fit``: whether or not the synthesizer has been fit.
                * ``last_fit_date``: date for the last time it was fit.
                * ``fitted_sdv_version``: version of sdv it was on when fitted.
                * ``fitted_sdv_enterprise_version``: version of sdv enterprsie if available.
        """
        info = {
            'class_name': self.__class__.__name__,
            'creation_date': self._creation_date,
            'is_fit': self._fitted,
            'last_fit_date': self._fitted_date,
            'fitted_sdv_version': self._fitted_sdv_version,
        }
        if self._fitted_sdv_enterprise_version is not None:
            info['fitted_sdv_enterprise_version'] = self._fitted_sdv_enterprise_version

        return info

    def _preprocess(self, data):
        if not self.metadata.tables:
            raise InvalidMetadataError(
                'The metadata is empty. Please add at least one table to the metadata.'
            )

        self._data_processor.fit(data)
        return self._data_processor.transform(data)

    def _store_and_convert_original_cols(self, data):
        # Transform in place to avoid possible large copy of data
        for column in data.columns:
            if isinstance(column, int):
                self._original_columns = data.columns
                data.columns = data.columns.astype(str)
                return True

        return False

    def add_constraints(self, constraints):
        """Add the list of constraint-augmented generation constraints to the synthesizer.

        Args:
            constraints (list):
                A list of constraints to apply to the synthesizer.
        """
        constraints = _validate_constraints_single_table(constraints, self._fitted)
        for constraint in constraints:
            if isinstance(constraint, ProgrammableConstraint):
                constraint = ProgrammableConstraintHarness(constraint)

            try:
                self.metadata = constraint.get_updated_metadata(self.metadata)
                self._chained_constraints.append(constraint)
                self._constraints_fitted = False
            except ConstraintNotMetError as e:
                LOGGER.info(
                    'Enforcing constraint %s using reject sampling.', constraint.__class__.__name__
                )

                try:
                    constraint.get_updated_metadata(self._original_metadata)
                    self._reject_sampling_constraints.append(constraint)
                except ConstraintNotMetError:
                    raise e

        self._data_processor = DataProcessor(
            metadata=self.metadata._convert_to_single_table(),
            enforce_rounding=self.enforce_rounding,
            enforce_min_max_values=self.enforce_min_max_values,
            locales=self.locales,
        )

    def get_constraints(self):
        """Get a list of constraint-augmented generation constraints applied to the synthesizer."""
        constraints = []
        for constraint in self._chained_constraints + self._reject_sampling_constraints:
            if isinstance(constraint, ProgrammableConstraintHarness):
                constraints.append(deepcopy(constraint.programmable_constraint))
            else:
                constraints.append(deepcopy(constraint))

        return constraints

    def validate_constraints(self, synthetic_data):
        """Validate synthetic_data against the constraints.

        Args:
            synthetic_data (pd.DataFrame): The synthetic data to validate

        Raises:
            ConstraintNotMetError:
                Raised if synthetic data does not match constraints.
        """
        transformed_data = synthetic_data
        for attribute in ['_reject_sampling_constraints', '_chained_constraints']:
            for constraint in getattr(self, attribute, []):
                if attribute == '_reject_sampling_constraints':
                    valid = constraint.is_valid(data=synthetic_data)
                else:
                    valid = constraint.is_valid(data=transformed_data)

                if not valid.all():
                    invalid_rows_str = _get_invalid_rows(valid)
                    constraint_name = _convert_to_snake_case(constraint.__class__.__name__)
                    constraint_name = constraint_name.replace('_', ' ')
                    msg = f'The {constraint_name} requirement is not met '
                    msg += f'for row indices: {invalid_rows_str}.'
                    raise ConstraintNotMetError(msg)
                elif attribute == '_chained_constraints':
                    transformed_data = constraint.transform(data=transformed_data)

    def _validate_transform_constraints(self, data, enforce_constraint_fitting=False):
        """Validate the data against the constraints and transform it.

        If the constraints are already fitted, it will only transform the data.
        If not, it will fit the constraints and then transform the data.
        The constraints validation is done during the fitting process.

        Args:
            data (pandas.DataFrame):
                The data to validate.
            enforce_constraint_fitting (bool):
                Whether to enforce fitting the constraints again. If set to ``True``, the
                constraints will be fitted again even if they have already been fitted.
                Defaults to ``False``.
        """
        if self._constraints_fitted and not enforce_constraint_fitting:
            for constraint in self._chained_constraints:
                data = constraint.transform(data)

            return data

        metadata = getattr(self, '_original_metadata', self.metadata)
        if hasattr(self, '_reject_sampling_constraints'):
            for constraint in self._reject_sampling_constraints:
                constraint.fit(data=data, metadata=self._original_metadata)

        if hasattr(self, '_chained_constraints'):
            for constraint in self._chained_constraints:
                constraint.fit(data=data, metadata=metadata)
                metadata = constraint.get_updated_metadata(metadata)
                data = constraint.transform(data)

        self._constraints_fitted = True
        return data

    def validate(self, data):
        """Validate data.

        This method will validate the data against:
        - The metadata
        - The constraints

        To make it work with the cags we temporarily set the metadata to the original one
        and then restore it.

        Args:
            data (pandas.DataFrame):
                The data to validate.
        """
        # Suppress duplicate datetime_format warning only when this single-table synthesizer
        # is embedded inside a multi-table synthesizer
        if getattr(self, '_suppress_datetime_format_warning', False):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore',
                    message=r"No 'datetime_format' is present.*",
                    category=UserWarning,
                )
                self._original_metadata.validate_data({self._table_name: data})
        else:
            self._original_metadata.validate_data({self._table_name: data})
        self._validate_transform_constraints(data, enforce_constraint_fitting=True)

        # Retaining the logic of returning errors and raising them here to maintain consistency
        # with the existing workflow with synthesizers
        synthesizer_errors = self._validate(data)  # Validate rules specific to each synthesizer
        if synthesizer_errors:
            raise InvalidDataError(synthesizer_errors)

    def _preprocess_helper(self, data):
        """This method is used to preprocess the data.

        It will:
        - Validate the data.
        - Warn if the model has already been fitted.
        - Validate the data against the constraints and transform it.

        Args:
            data (pandas.DataFrame):
                The data to preprocess.

        Returns:
            pandas.DataFrame:
                The data after constraint transformation.
        """
        self.validate(data)
        if self._fitted:
            msg = (
                'This model has already been fitted. To use the new preprocessed data, '
                "please refit the model using 'fit' or 'fit_processed_data'."
            )
            warnings.warn(msg, RefitWarning)

        data = self._validate_transform_constraints(data)

        return data

    def preprocess(self, data):
        """Transform the raw data to numerical space.

        Args:
            data (pandas.DataFrame):
                The raw data to be transformed.

        Returns:
            pandas.DataFrame:
                The preprocessed data.
        """
        is_converted = self._store_and_convert_original_cols(data)
        data = self._preprocess_helper(data)
        preprocess_data = self._preprocess(data)
        if is_converted:
            data.columns = self._original_columns

        return preprocess_data

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
        SYNTHESIZER_LOGGER.info({
            'EVENT': 'Fit processed data',
            'TIMESTAMP': datetime.datetime.now(),
            'SYNTHESIZER CLASS NAME': self.__class__.__name__,
            'SYNTHESIZER ID': self._synthesizer_id,
            'TOTAL NUMBER OF TABLES': 1,
            'TOTAL NUMBER OF ROWS': len(processed_data),
            'TOTAL NUMBER OF COLUMNS': len(processed_data.columns),
        })

        check_synthesizer_version(self, is_fit_method=True, compare_operator=operator.lt)
        if not processed_data.empty:
            self._fit(processed_data)

        self._fitted = True
        self._fitted_date = datetime.datetime.today().strftime('%Y-%m-%d')
        self._fitted_sdv_version = getattr(version, 'community', None)
        self._fitted_sdv_enterprise_version = getattr(version, 'enterprise', None)

    def fit(self, data):
        """Fit this model to the original data.

        Args:
            data (pandas.DataFrame):
                The raw data (before any transformations) to fit the model to.
        """
        SYNTHESIZER_LOGGER.info({
            'EVENT': 'Fit',
            'TIMESTAMP': datetime.datetime.now(),
            'SYNTHESIZER CLASS NAME': self.__class__.__name__,
            'SYNTHESIZER ID': self._synthesizer_id,
            'TOTAL NUMBER OF TABLES': 1,
            'TOTAL NUMBER OF ROWS': len(data),
            'TOTAL NUMBER OF COLUMNS': len(data.columns),
        })

        check_synthesizer_version(self, is_fit_method=True, compare_operator=operator.lt)
        self._check_input_metadata_updated()
        self._fitted = False
        self._data_processor.reset_sampling()
        self._random_state_set = False
        is_converted = self._store_and_convert_original_cols(data)
        processed_data = self.preprocess(data)
        self.fit_processed_data(processed_data)
        if is_converted:
            data.columns = self._original_columns

    def _validate_fit_before_save(self):
        """Validate that the synthesizer has been fitted before saving."""
        if not self._fitted:
            warnings.warn(
                'You are saving a synthesizer that has not yet been fitted. You will not be able '
                'to sample synthetic data without fitting. We recommend fitting the synthesizer '
                'first and then saving.'
            )

    def save(self, filepath):
        """Save this model instance to the given path using cloudpickle.

        Args:
            filepath (str):
                Path where the synthesizer instance will be serialized.
        """
        self._validate_fit_before_save()
        synthesizer_id = getattr(self, '_synthesizer_id', None)
        SYNTHESIZER_LOGGER.info({
            'EVENT': 'Save',
            'TIMESTAMP': datetime.datetime.now(),
            'SYNTHESIZER CLASS NAME': self.__class__.__name__,
            'SYNTHESIZER ID': synthesizer_id,
        })

        with open(filepath, 'wb') as output:
            cloudpickle.dump(self, output)

    @classmethod
    def load(cls, filepath):
        """Load a single-table synthesizer from a given path.

        Args:
            filepath (str):
                A string describing the filepath of your saved synthesizer.

        Returns:
            SingleTableSynthesizer:
                The loaded synthesizer.
        """
        warn_load_deprecated()
        with open(filepath, 'rb') as f:
            try:
                synthesizer = cloudpickle.load(f)
            except RuntimeError as e:
                err_msg = (
                    'Attempting to deserialize object on a CUDA device but '
                    'torch.cuda.is_available() is False. If you are running on a CPU-only machine,'
                    " please use torch.load with map_location=torch.device('cpu') "
                    'to map your storages to the CPU.'
                )
                if str(e) == err_msg:
                    raise SamplingError(
                        'This synthesizer was created on a machine with GPU but the current '
                        'machine is CPU-only. This feature is currently unsupported. We recommend'
                        ' sampling on the same GPU-enabled machine.'
                    )
                raise e

        _validate_correct_synthesizer_loading(synthesizer, cls)
        check_synthesizer_version(synthesizer)
        check_sdv_versions_and_warn(synthesizer)
        if getattr(synthesizer, '_synthesizer_id', None) is None:
            synthesizer._synthesizer_id = generate_synthesizer_id(synthesizer)

        SYNTHESIZER_LOGGER.info({
            'EVENT': 'Load',
            'TIMESTAMP': datetime.datetime.now(),
            'SYNTHESIZER CLASS NAME': synthesizer.__class__.__name__,
            'SYNTHESIZER ID': synthesizer._synthesizer_id,
        })

        return synthesizer

    def reverse_transform_constraints(self, sampled):
        """Reverse transform the constraints."""
        if hasattr(self, '_chained_constraints') and hasattr(self, '_reject_sampling_constraints'):
            for constraint in reversed(self._chained_constraints):
                sampled = constraint.reverse_transform(sampled)
                valid_rows = constraint.is_valid(sampled)
                sampled = sampled[valid_rows]

            for constraint in reversed(self._reject_sampling_constraints):
                valid_rows = constraint.is_valid(sampled)
                sampled = sampled[valid_rows]

        return sampled


class BaseSingleTableSynthesizer(BaseSynthesizer):
    """Base class for all single-table ``Synthesizers``.

    The ``BaseSingleTableSynthesizer`` class defines the common sampling methods
    for all single-table synthesizers.
    """

    def _set_random_state(self, random_state):
        """Set the random state of the model's random number generator.

        Args:
            random_state (int, tuple[np.random.RandomState, torch.Generator], or None):
                Seed or tuple of random states to use.
        """
        self._model.set_random_state(random_state)
        self._random_state_set = True

    def reset_sampling(self):
        """Reset the sampling to the state that was left right after fitting."""
        self._data_processor.reset_sampling()
        self._random_state_set = False

    @staticmethod
    def _filter_conditions(sampled, conditions, float_rtol):
        """Filter the sampled rows that match the conditions.

        If condition columns are float values, consider a match anything that
        is closer than the given ``float_rtol`` and then make the value exact.

        Args:
            sampled (pandas.DataFrame):
                The sampled rows, reverse transformed.
            conditions (dict):
                The dictionary of conditioning values.
            float_rtol (float):
                Maximum tolerance when considering a float match.

        Returns:
            pandas.DataFrame:
                Rows from the sampled data that match the conditions.
        """
        for column, value in conditions.items():
            column_values = sampled[column]
            if pd.isna(value):
                sampled = sampled[column_values.isna()]
            elif is_float_dtype(column_values.dtype):
                distance = abs(value) * float_rtol
                sampled = sampled[np.abs(column_values - value) <= distance]
                sampled.loc[:, column] = value
            else:
                sampled = sampled[column_values == value]

        return sampled

    def _sample_rows(
        self,
        num_rows,
        conditions=None,
        transformed_conditions=None,
        float_rtol=0.1,
        previous_rows=None,
        keep_extra_columns=False,
    ):
        """Sample rows with the given conditions.

        Input conditions is taken both in the raw input format, which will be used
        for filtering during the reject-sampling loop, and already transformed
        to the model format, which will be passed down to the model if it supports
        conditional sampling natively.

        If condition columns are float values, consider a match anything that
        is closer than the given ``float_rtol`` and then make the value exact.

        If the model does not have any data columns, the result of this call
        is a dataframe of the requested length with no columns in it.

        If there are no columns other than the ``primary_key``, this will proceed to sample
        only the ``primary_key`` using the ``DataProcessor``.

        Args:
            num_rows (int):
                Number of rows to sample.
            conditions (dict):
                The dictionary of conditioning values in the original format.
            transformed_conditions (dict):
                The dictionary of conditioning values transformed to the model format.
            float_rtol (float):
                Maximum tolerance when considering a float match.
            previous_rows (pandas.DataFrame):
                Valid rows sampled in the previous iterations.
            keep_extra_columns (bool):
                Whether to keep extra columns from the sampled data. Defaults to False.

        Returns:
            tuple:
                * pandas.DataFrame:
                    Rows from the sampled data that match the conditions.
                * int:
                    Number of rows that are considered valid.
        """
        if self._model and not self._random_state_set:
            self._set_random_state(FIXED_RNG_SEED)
        need_sample = self._data_processor.get_sdtypes(primary_keys=False) or keep_extra_columns
        if self._model and need_sample:
            if conditions is None:
                raw_sampled = self._sample(num_rows)
            else:
                try:
                    raw_sampled = self._sample(num_rows, transformed_conditions)
                except NotImplementedError:
                    raw_sampled = self._sample(num_rows)
            sampled = self._data_processor.reverse_transform(raw_sampled, conditions=conditions)
            sampled = self.reverse_transform_constraints(sampled)
            if keep_extra_columns:
                input_columns = self._data_processor._hyper_transformer._input_columns
                missing_cols = list(
                    set(raw_sampled.columns) - set(input_columns) - set(sampled.columns)
                )
                sampled = pd.concat([sampled, raw_sampled.loc[sampled.index, missing_cols]], axis=1)

            if previous_rows is not None:
                sampled = pd.concat([previous_rows, sampled], ignore_index=True)
            sampled = self._data_processor.filter_valid(sampled)

            if conditions is not None:
                sampled = self._filter_conditions(sampled, conditions, float_rtol)

            num_valid = len(sampled)

            return sampled, num_valid

        else:
            sampled = pd.DataFrame(index=range(num_rows))
            sampled = self._data_processor.reverse_transform(sampled)
            return sampled, num_rows

    def _sample_batch(
        self,
        batch_size,
        max_tries=100,
        conditions=None,
        transformed_conditions=None,
        float_rtol=0.01,
        progress_bar=None,
        output_file_path=None,
        keep_extra_columns=False,
    ):
        """Sample a batch of rows with the given conditions.

        This will enter a reject-sampling loop in which rows will be sampled until
        all of them are valid and match the requested conditions. If ``max_tries``
        is exceeded, it will return as many rows as it has sampled, which may be less
        than the target number of rows.

        Input conditions is taken both in the raw input format, which will be used
        for filtering during the reject-sampling loop, and already transformed
        to the model format, which will be passed down to the model if it supports
        conditional sampling natively.

        If condition columns are float values, consider a match anything that is
        relatively closer than the given ``float_rtol`` and then make the value exact.

        If the model does not have any data columns, the result of this call
        is a dataframe of the requested length with no columns in it.

        Args:
            batch_size (int):
                Number of rows to sample for this batch. If not given the model
                will generate as many rows as there were in the
                data passed to the ``fit`` method.
            max_tries (int):
                Number of times to retry sampling until the batch size is met.
                Defaults to 100.
            conditions (dict):
                The dictionary of conditioning values in the original input format.
            transformed_conditions (dict):
                The dictionary of conditioning values transformed to the model format.
            float_rtol (float):
                Maximum tolerance when considering a float match.
            progress_bar (tqdm.tqdm or None):
                The progress bar to update when sampling. If None, a new tqdm progress
                bar will be created.
            output_file_path (str or None):
                The file to periodically write sampled rows to. If None, does not write
                rows anywhere.
            keep_extra_columns (bool):
                Whether to keep extra columns from the sampled data. Defaults to False.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        num_rows_to_sample = batch_size

        counter = 0
        num_valid = 0
        prev_num_valid = None
        remaining = batch_size
        sampled = pd.DataFrame()

        while num_valid < batch_size and counter < max_tries:
            prev_num_valid = num_valid
            sampled, num_valid = self._sample_rows(
                num_rows_to_sample,
                conditions,
                transformed_conditions,
                float_rtol,
                sampled,
                keep_extra_columns,
            )

            num_new_valid_rows = num_valid - prev_num_valid
            num_increase = min(num_new_valid_rows, remaining)
            num_sampled = min(len(sampled), batch_size)
            if num_increase > 0:
                if output_file_path:
                    append_kwargs = {'mode': 'a', 'header': False}
                    append_kwargs = append_kwargs if os.path.getsize(output_file_path) > 0 else {}
                    sampled.head(num_sampled).tail(num_increase).to_csv(
                        output_file_path,
                        index=False,
                        **append_kwargs,
                    )

                if progress_bar is not None:
                    progress_bar.update(num_increase)

            remaining = batch_size - num_valid
            valid_rate = max(num_new_valid_rows, 1) / max(num_rows_to_sample, 1)
            num_rows_to_sample = min(10 * batch_size, int(remaining / valid_rate))

            if remaining > 0:
                LOGGER.info(
                    f'{remaining} valid rows remaining. Resampling {num_rows_to_sample} rows'
                )

            counter += 1

        return sampled.head(min(len(sampled), batch_size))

    @staticmethod
    def _make_condition_dfs(conditions):
        """Transform ``conditions`` into a list of dataframes.

        Args:
            conditions (list[sdv.sampling.Condition, sdv.sampling.DataFrameCondition]):
                A list of ``sdv.sampling.Condition`` or ``sdv.sampling.DataFrameCondition``.
                Each ``Condition`` object represents a desired column value mapping and
                the number of rows to generate for that condition.
                Each ``DataFrameCondition`` represents the dataframe to match for the condition.

        Returns:
            list[pandas.DataFrame]:
                A list of ``conditions`` as dataframes.
        """
        condition_dataframes = defaultdict(list)
        for condition in conditions:
            if isinstance(condition, Condition):
                column_values = condition.get_column_values()
                condition_dataframes[tuple(column_values.keys())].append(
                    pd.DataFrame(column_values, index=range(condition.get_num_rows()))
                )
            elif isinstance(condition, DataFrameCondition):
                dataframe = condition.get_dataframe()
                columns = dataframe.columns.tolist()
                condition_dataframes[tuple(columns)].append(dataframe)
            else:
                raise ValueError('`conditions` must be list of Condition or DataFrameCondition')
        return [
            pd.concat(condition_list, ignore_index=True)
            for condition_list in condition_dataframes.values()
        ]

    def _sample_in_batches(
        self,
        num_rows,
        batch_size,
        max_tries_per_batch,
        conditions=None,
        transformed_conditions=None,
        float_rtol=0.01,
        progress_bar=None,
        output_file_path=None,
        keep_extra_columns=False,
    ):
        sampled = []
        batch_size = batch_size if num_rows > batch_size else num_rows
        for step in range(math.ceil(num_rows / batch_size)):
            sampled_rows = self._sample_batch(
                batch_size=batch_size,
                max_tries=max_tries_per_batch,
                conditions=conditions,
                transformed_conditions=transformed_conditions,
                float_rtol=float_rtol,
                progress_bar=progress_bar,
                output_file_path=output_file_path,
                keep_extra_columns=keep_extra_columns,
            )
            sampled.append(sampled_rows)

        sampled = pd.concat(sampled, ignore_index=True) if len(sampled) > 0 else pd.DataFrame()
        return sampled.head(num_rows)

    def _conditionally_sample_rows(
        self,
        dataframe,
        condition,
        transformed_condition,
        max_tries_per_batch=None,
        batch_size=None,
        float_rtol=0.01,
        graceful_reject_sampling=True,
        progress_bar=None,
        output_file_path=None,
        keep_extra_columns=False,
    ):
        batch_size = batch_size or len(dataframe)
        sampled_rows = self._sample_in_batches(
            num_rows=len(dataframe),
            batch_size=batch_size,
            max_tries_per_batch=max_tries_per_batch,
            conditions=condition,
            transformed_conditions=transformed_condition,
            float_rtol=float_rtol,
            progress_bar=progress_bar,
            output_file_path=output_file_path,
            keep_extra_columns=keep_extra_columns,
        )

        if len(sampled_rows) > 0:
            sampled_rows[COND_IDX] = dataframe[COND_IDX].to_numpy()[: len(sampled_rows)]

        elif not graceful_reject_sampling:
            user_msg = (
                f"Unable to sample any rows for the given conditions '{transformed_condition}'. "
            )
            if hasattr(self, '_model') and isinstance(self._model, GaussianMultivariate):
                user_msg = user_msg + (
                    'This may be because the provided values are out-of-bounds in the '
                    'current model. \nPlease try again with a different set of values.'
                )
            else:
                user_msg = user_msg + (
                    f"Try increasing 'max_tries_per_batch' (currently: {max_tries_per_batch}) "
                    f"or increasing 'batch_size' (currently: {batch_size}). Note that "
                    'increasing these values will also increase the sampling time.'
                )

            raise ValueError(user_msg)

        return sampled_rows

    def _sample_with_progress_bar(
        self,
        num_rows,
        max_tries_per_batch=100,
        batch_size=None,
        output_file_path=None,
        show_progress_bar=True,
    ):
        if num_rows is None:
            raise ValueError('You must specify the number of rows to sample (e.g. num_rows=100).')

        sampled = pd.DataFrame()
        if num_rows == 0:
            return sampled

        output_file_path = validate_file_path(output_file_path)
        batch_size = min(batch_size, num_rows) if batch_size else num_rows

        try:
            with tqdm.tqdm(total=num_rows, disable=not show_progress_bar) as progress_bar:
                progress_bar.set_description('Sampling rows')
                sampled = self._sample_in_batches(
                    num_rows=num_rows,
                    batch_size=batch_size,
                    max_tries_per_batch=max_tries_per_batch,
                    progress_bar=progress_bar,
                    output_file_path=output_file_path,
                )

        except (Exception, KeyboardInterrupt) as error:
            handle_sampling_error(output_file_path, error)

        return sampled

    def _validate_fit_before_sample(self):
        if not self._fitted:
            raise SamplingError(
                'This synthesizer has not been fitted. Please fit your synthesizer first before'
                ' sampling synthetic data.'
            )

    def sample(self, num_rows, max_tries_per_batch=100, batch_size=None, output_file_path=None):
        """Sample rows from this table.

        Args:
            num_rows (int):
                Number of rows to sample. This parameter is required.
            max_tries_per_batch (int):
                Number of times to retry sampling until the batch size is met. Defaults to 100.
            batch_size (int or None):
                The batch size to sample. Defaults to ``num_rows``, if None.
            output_file_path (str or None):
                The file to periodically write sampled rows to. If None, does not
                write rows anywhere.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        self._validate_fit_before_sample()
        self._check_input_metadata_updated()
        sample_timestamp = datetime.datetime.now()
        has_constraints = bool(self._data_processor._constraints)
        has_batches = batch_size is not None and batch_size != num_rows
        show_progress_bar = has_constraints or has_batches

        sampled_data = self._sample_with_progress_bar(
            num_rows,
            max_tries_per_batch,
            batch_size,
            output_file_path,
            show_progress_bar=show_progress_bar,
        )

        original_columns = getattr(self, '_original_columns', pd.Index([]))
        if not original_columns.empty:
            sampled_data.columns = self._original_columns

        SYNTHESIZER_LOGGER.info({
            'EVENT': 'Sample',
            'TIMESTAMP': sample_timestamp,
            'SYNTHESIZER CLASS NAME': self.__class__.__name__,
            'SYNTHESIZER ID': self._synthesizer_id,
            'TOTAL NUMBER OF TABLES': 1,
            'TOTAL NUMBER OF ROWS': len(sampled_data),
            'TOTAL NUMBER OF COLUMNS': len(sampled_data.columns),
        })

        return sampled_data

    def _transform_conditions(self, condition_df):
        return self._data_processor.transform(condition_df, is_condition=True)

    def _transform_conditions_chained_constraints(self, condition_df):
        try:
            transformed_condition = self._validate_transform_constraints(condition_df)
            transformed_condition = self._transform_conditions(transformed_condition)
        except ConstraintNotMetError:
            raise ConstraintNotMetError(
                'Provided conditions are not valid for the given constraints.'
            )
        except Exception:
            transformed_condition = self._transform_conditions(condition_df)

        return transformed_condition

    def _sample_with_conditions(
        self,
        conditions,
        max_tries_per_batch,
        batch_size,
        progress_bar=None,
        output_file_path=None,
        keep_extra_columns=False,
    ):
        """Sample rows with conditions.

        Args:
            conditions (pandas.DataFrame):
                A DataFrame representing the conditions to be sampled.
            max_tries_per_batch (int):
                Number of times to retry sampling until the batch size is met. Defaults to 100.
            batch_size (int):
                The batch size to use for each sampling call.
            progress_bar (tqdm.tqdm or None):
                The progress bar to update.
            output_file_path (str or None):
                The file to periodically write sampled rows to. Defaults to None.
            keep_extra_columns (bool):
                Whether to keep extra columns from the sampled data. Defaults to False.

        Returns:
            pandas.DataFrame:
                Sampled data.

        Raises:
            ConstraintsNotMetError:
                If the conditions are not valid for the given constraints.
            ValueError:
                If any of the following happens:
                    * any of the conditions' columns are not valid.
                    * no rows could be generated.
        """
        condition_columns = list(conditions.columns)
        conditions.index.name = COND_IDX
        conditions = conditions.reset_index()
        grouped_conditions = conditions.groupby(_groupby_list(condition_columns), dropna=False)

        # sample
        all_sampled_rows = []

        for group, dataframe in grouped_conditions:
            if not isinstance(group, tuple):
                group = [group]

            condition = dict(zip(condition_columns, group))
            condition_df = dataframe.iloc[0].to_frame().T
            dtypes = conditions.dtypes.to_dict()
            condition_df = condition_df.astype(dtypes)
            if hasattr(self, '_chained_constraints'):
                transformed_condition = self._transform_conditions_chained_constraints(condition_df)
            else:
                try:
                    transformed_condition = self._transform_conditions(condition_df)
                except ConstraintsNotMetError as error:
                    raise ConstraintsNotMetError(
                        'Provided conditions are not valid for the given constraints.'
                    ) from error

            transformed_conditions = pd.concat(
                [transformed_condition] * len(dataframe), ignore_index=True
            )
            transformed_columns = list(transformed_conditions.columns)
            if not transformed_conditions.empty:
                transformed_conditions.index = dataframe.index
                transformed_conditions[COND_IDX] = dataframe[COND_IDX]

            if len(transformed_columns) == 0:
                sampled_rows = self._conditionally_sample_rows(
                    dataframe=dataframe,
                    condition=condition,
                    transformed_condition=None,
                    max_tries_per_batch=max_tries_per_batch,
                    batch_size=batch_size,
                    progress_bar=progress_bar,
                    output_file_path=output_file_path,
                    keep_extra_columns=keep_extra_columns,
                )
                all_sampled_rows.append(sampled_rows)
            else:
                transformed_groups = transformed_conditions.groupby(
                    _groupby_list(transformed_columns)
                )
                for transformed_group, transformed_dataframe in transformed_groups:
                    if not isinstance(transformed_group, tuple):
                        transformed_group = [transformed_group]

                    transformed_condition = dict(zip(transformed_columns, transformed_group))
                    sampled_rows = self._conditionally_sample_rows(
                        dataframe=transformed_dataframe,
                        condition=condition,
                        transformed_condition=transformed_condition,
                        max_tries_per_batch=max_tries_per_batch,
                        batch_size=batch_size,
                        progress_bar=progress_bar,
                        output_file_path=output_file_path,
                        keep_extra_columns=keep_extra_columns,
                    )
                    all_sampled_rows.append(sampled_rows)

        all_sampled_rows = pd.concat(all_sampled_rows)
        if len(all_sampled_rows) == 0:
            return all_sampled_rows

        all_sampled_rows = all_sampled_rows.set_index(COND_IDX)
        all_sampled_rows.index.name = conditions.index.name
        all_sampled_rows = all_sampled_rows.sort_index()

        return all_sampled_rows

    def _validate_conditions_unseen_columns(self, conditions):
        """Validate the user-passed conditions."""
        for column in conditions.columns:
            if hasattr(self, '_original_metadata'):
                if column not in self._original_metadata.tables[self._table_name].columns:
                    raise ValueError(
                        f"Unexpected column name '{column}'. "
                        'Use a column name that was present in the original data.'
                    )
                if column == self._original_metadata.tables[self._table_name].primary_key:
                    raise ValueError(
                        f"Cannot conditionally sample column name '{column}' because it is "
                        'the primary key.'
                    )
            else:
                if column not in self._data_processor.get_sdtypes():
                    raise ValueError(
                        f"Unexpected column name '{column}'. "
                        f'Use a column name that was present in the original data.'
                    )

    def _validate_conditions(self, conditions):
        """Validate the user-passed conditions."""
        for condition_dataframe in conditions:
            self._validate_conditions_unseen_columns(condition_dataframe)

    def sample_from_conditions(
        self, conditions, max_tries_per_batch=100, batch_size=None, output_file_path=None
    ):
        """Sample rows from this table with the given conditions.

        Args:
            conditions (list[sdv.sampling.Condition, sdv.sampling.DataFrameCondition]):
                A list of sdv.sampling.Condition and sdv.sampling.DataFrameCondition objects,
                which specify the column values in a condition, along with the number of
                rows for that condition.
            max_tries_per_batch (int):
                Number of times to retry sampling until the batch size is met. Defaults to 100.
            batch_size (int):
                The batch size to use per sampling call.
            output_file_path (str or None):
                The file to periodically write sampled rows to. Defaults to None.

        Returns:
            pandas.DataFrame:
                Sampled data.

        Raises:
            ConstraintsNotMetError:
                If the conditions are not valid for the given constraints.
            ValueError:
                If any of the following happens:
                    * any of the conditions' columns are not valid.
                    * no rows could be generated.
        """
        self._validate_fit_before_sample()
        output_file_path = validate_file_path(output_file_path)

        num_rows = functools.reduce(
            lambda num_rows, condition: condition.get_num_rows() + num_rows, conditions, 0
        )

        conditions = self._make_condition_dfs(conditions)
        self._validate_conditions(conditions)

        sampled = pd.DataFrame()
        try:
            with tqdm.tqdm(total=num_rows) as progress_bar:
                progress_bar.set_description('Sampling conditions')
                for condition_dataframe in conditions:
                    sampled_for_condition = self._sample_with_conditions(
                        condition_dataframe,
                        max_tries_per_batch,
                        batch_size,
                        progress_bar,
                        output_file_path,
                    )
                    sampled = pd.concat([sampled, sampled_for_condition], ignore_index=True)

            is_reject_sampling = bool(
                hasattr(self, '_model') and not isinstance(self._model, GaussianMultivariate)
            )
            check_num_rows(
                num_rows=len(sampled),
                expected_num_rows=num_rows,
                is_reject_sampling=is_reject_sampling,
                max_tries_per_batch=max_tries_per_batch,
            )

        except (Exception, KeyboardInterrupt) as error:
            handle_sampling_error(output_file_path, error)

        return sampled

    def _validate_known_columns(self, conditions):
        """Validate the user-passed conditions."""
        self._validate_conditions_unseen_columns(conditions)

    def sample_remaining_columns(
        self, known_columns, max_tries_per_batch=100, batch_size=None, output_file_path=None
    ):
        """Sample remaining rows from already known columns.

        Args:
            known_columns (pandas.DataFrame):
                A pandas.DataFrame with the columns that are already known. The output
                is a DataFrame such that each row in the output is sampled
                conditionally on the corresponding row in the input.
            max_tries_per_batch (int):
                Number of times to retry sampling until the batch size is met. Defaults to 100.
            batch_size (int):
                The batch size to use per sampling call.
            output_file_path (str or None):
                The file to periodically write sampled rows to. Defaults to None.

        Returns:
            pandas.DataFrame:
                Sampled data.

        Raises:
            ConstraintsNotMetError:
                If the conditions are not valid for the given constraints.
            ValueError:
                If any of the following happens:
                    * any of the conditions' columns are not valid.
                    * no rows could be generated.
        """
        output_file_path = validate_file_path(output_file_path)

        known_columns = known_columns.copy()
        self._validate_known_columns(known_columns)
        sampled = pd.DataFrame()
        try:
            with tqdm.tqdm(total=len(known_columns)) as progress_bar:
                progress_bar.set_description('Sampling remaining columns')
                sampled = self._sample_with_conditions(
                    known_columns, max_tries_per_batch, batch_size, progress_bar, output_file_path
                )

            is_reject_sampling = hasattr(self, '_model') and not isinstance(
                self._model, copulas.multivariate.GaussianMultivariate
            )

            check_num_rows(
                num_rows=len(sampled),
                expected_num_rows=len(known_columns),
                is_reject_sampling=is_reject_sampling,
                max_tries_per_batch=max_tries_per_batch,
            )

        except (Exception, KeyboardInterrupt) as error:
            handle_sampling_error(output_file_path, error)

        return sampled
