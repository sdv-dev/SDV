"""Base Multi Table Synthesizer class."""

import contextlib
import datetime
import inspect
import operator
import warnings
from collections import defaultdict
from copy import deepcopy

import cloudpickle
import numpy as np
from tqdm import tqdm

from sdv import version
from sdv._utils import (
    check_sdv_versions_and_warn,
    check_synthesizer_version,
    generate_synthesizer_id,
)
from sdv.cag._errors import ConstraintNotMetError
from sdv.cag._utils import _convert_to_snake_case, _get_invalid_rows, _validate_constraints
from sdv.cag.programmable_constraint import ProgrammableConstraint, ProgrammableConstraintHarness
from sdv.errors import (
    InvalidDataError,
    SamplingError,
    SynthesizerInputError,
)
from sdv.logging import disable_single_table_logger, get_sdv_logger
from sdv.metadata.metadata import Metadata
from sdv.metadata.multi_table import MultiTableMetadata
from sdv.single_table.copulas import GaussianCopulaSynthesizer

SYNTHESIZER_LOGGER = get_sdv_logger('MultiTableSynthesizer')
DEPRECATION_MSG = (
    "The 'MultiTableMetadata' is deprecated. Please use the new 'Metadata' class for synthesizers."
)


class BaseMultiTableSynthesizer:
    """Base class for multi table synthesizers.

    The ``BaseMultiTableSynthesizer`` class defines the common API that all the
    multi table synthesizers need to implement, as well as common functionality.

    Args:
        metadata (sdv.metadata.multi_table.MultiTableMetadata):
            Multi table metadata representing the data tables that this synthesizer will be used
            for.
        locales (list or str):
            The default locale(s) to use for AnonymizedFaker transformers.
            Defaults to ``['en_US']``.
        verbose (bool):
            Whether to print progress for fitting or not.
    """

    DEFAULT_SYNTHESIZER_KWARGS = None
    _synthesizer = GaussianCopulaSynthesizer
    _numpy_seed = 73251

    @contextlib.contextmanager
    def _set_temp_numpy_seed(self):
        initial_state = np.random.get_state()
        if isinstance(self._numpy_seed, int):
            np.random.seed(self._numpy_seed)
            np.random.default_rng(self._numpy_seed)
        else:
            np.random.set_state(self._numpy_seed)
            np.random.default_rng(self._numpy_seed[1])
        try:
            yield
        finally:
            self._numpy_seed = np.random.get_state()
            np.random.set_state(initial_state)

    def _initialize_models(self):
        with disable_single_table_logger():
            for table_name, table_metadata in self.metadata.tables.items():
                synthesizer_parameters = {'locales': self.locales}
                synthesizer_parameters.update(self._table_parameters.get(table_name, {}))
                metadata_dict = {'tables': {table_name: table_metadata.to_dict()}}
                metadata = Metadata.load_from_dict(metadata_dict)
                self._table_synthesizers[table_name] = self._synthesizer(
                    metadata=metadata, **synthesizer_parameters
                )
                self._table_synthesizers[table_name]._data_processor.table_name = table_name

    def _get_pbar_args(self, **kwargs):
        """Return a dictionary with the updated keyword args for a progress bar."""
        pbar_args = {'disable': not self.verbose}
        pbar_args.update(kwargs)

        return pbar_args

    def _print(self, text='', **kwargs):
        if self.verbose:
            print(text, **kwargs)  # noqa: T201

    def _check_metadata_updated(self):
        if self.metadata._check_updated_flag():
            self.metadata._reset_updated_flag()
            warnings.warn(
                "We strongly recommend saving the metadata using 'save_to_json' for replicability"
                ' in future SDV versions.'
            )

    def __init__(self, metadata, locales=['en_US'], synthesizer_kwargs=None):
        self.metadata = metadata
        if type(metadata) is MultiTableMetadata:
            warnings.warn(DEPRECATION_MSG, FutureWarning)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message=r'.*column relationship.*')
            self.metadata.validate()

        self._check_metadata_updated()
        self.locales = locales
        self.verbose = False
        self.extended_columns = defaultdict(dict)
        self._table_synthesizers = {}
        self._table_parameters = defaultdict(dict)
        self._original_table_columns = {}
        self._original_metadata = deepcopy(self.metadata)
        self.constraints = []
        self._has_seen_single_table_constraint = False
        if synthesizer_kwargs is not None:
            warn_message = (
                'The `synthesizer_kwargs` parameter is deprecated as of SDV 1.2.0 and does not '
                'affect the synthesizer. Please use the `set_table_parameters` method instead.'
            )
            warnings.warn(warn_message, FutureWarning)

        if self.DEFAULT_SYNTHESIZER_KWARGS:
            for table_name in self.metadata.tables:
                self._table_parameters[table_name] = deepcopy(self.DEFAULT_SYNTHESIZER_KWARGS)

        self._initialize_models()
        self._fitted = False
        self._constraints_fitted = False
        self._creation_date = datetime.datetime.today().strftime('%Y-%m-%d')
        self._fitted_date = None
        self._fitted_sdv_version = None
        self._fitted_sdv_enterprise_version = None
        self._synthesizer_id = generate_synthesizer_id(self)
        SYNTHESIZER_LOGGER.info({
            'EVENT': 'Instance',
            'TIMESTAMP': datetime.datetime.now(),
            'SYNTHESIZER CLASS NAME': self.__class__.__name__,
            'SYNTHESIZER ID': self._synthesizer_id,
        })

    def set_address_columns(self, table_name, column_names, anonymization_level='full'):
        """Set the address multi-column transformer.

        Args:
            table_name (str):
                The name of the table for which the address transformer should be set.
            column_names (tuple[str]):
                The column names to be used for the address transformer.
            anonymization_level (str):
                The anonymization level to use for the address transformer.
        """
        self._validate_table_name(table_name)
        self._table_synthesizers[table_name].set_address_columns(column_names, anonymization_level)

    def _detect_single_table_constraints(self, constraints):
        """Filter out single table constraints from the list of constraints.

        Args:
            constraints (list):
                A list of constraints to filter.
        """
        idx_single_table_constraint = 0 if self._has_seen_single_table_constraint else None
        for idx, constraint in enumerate(constraints):
            if self._has_seen_single_table_constraint and constraint._is_single_table is False:
                raise SynthesizerInputError(
                    'Cannot apply multi-table constraint after single-table constraint has '
                    'been applied.'
                )

            if not constraint._is_single_table:
                continue

            if not self._has_seen_single_table_constraint:
                self._has_seen_single_table_constraint = True
                idx_single_table_constraint = idx

        return idx_single_table_constraint

    def _validate_single_table_constraints(self, constraints):
        for constraint in constraints:
            if getattr(constraint, 'table_name') is None:
                raise SynthesizerInputError(
                    f"The '{constraint.__class__.__name__}' is missing the required parameter "
                    "'table_name'. Please add this parameter to your constraint definition."
                )

    def add_constraints(self, constraints):
        """Add the list of constraint-augmented generation constraints to the synthesizer.

        Args:
            constraints (list):
                A list of constraints to apply to the synthesizer.
        """
        constraints = _validate_constraints(constraints, self._fitted)
        metadata = self.metadata
        multi_table_constraints = []
        single_table_constraints = []
        idx_single_table_constraint = self._detect_single_table_constraints(constraints)
        for idx, constraint in enumerate(constraints):
            if isinstance(constraint, ProgrammableConstraint):
                constraint = ProgrammableConstraintHarness(constraint)

            if idx_single_table_constraint is not None and idx >= idx_single_table_constraint:
                single_table_constraints.append(constraint)
                continue

            multi_table_constraints.append(constraint)
            metadata = constraint.get_updated_metadata(metadata)

        self.metadata = metadata
        self._validate_single_table_constraints(single_table_constraints)
        self.constraints += multi_table_constraints
        self._constraints_fitted = False
        self._initialize_models()
        if single_table_constraints:
            for constraint in single_table_constraints:
                table_name = constraint.table_name
                self._table_synthesizers[table_name].add_constraints([constraint])
                self.metadata = constraint.get_updated_metadata(self.metadata)

    def get_constraints(self):
        """Get a copy of the list of constraints applied to the synthesizer."""
        if not hasattr(self, 'constraints'):
            return []

        constraints = []
        for constraint in self.constraints:
            if isinstance(constraint, ProgrammableConstraintHarness):
                constraints.append(deepcopy(constraint.programmable_constraint))
            else:
                constraints.append(deepcopy(constraint))

        for table_name, synthesizer in self._table_synthesizers.items():
            constraints += synthesizer.get_constraints()

        return constraints

    def validate_constraints(self, synthetic_data):
        """Validate synthetic_data against the constraints.

        Args:
            data (dict[str, pd.DataFrame]): The synthetic data to validate

        Raises:
            ConstraintNotMetError:
                Raised if synthetic data does not match constraints.
        """
        transformed_data = synthetic_data
        for constraint in self.constraints:
            valid_data = constraint.is_valid(data=transformed_data)
            for table_name, valid_table in valid_data.items():
                if not valid_table.all():
                    invalid_rows_str = _get_invalid_rows(valid_table)
                    constraint_name = _convert_to_snake_case(constraint.__class__.__name__)
                    constraint_name = constraint_name.replace('_', ' ')
                    msg = f"Table '{table_name}': The {constraint_name} requirement is not met "
                    msg += f'for row indices: {invalid_rows_str}.'
                    raise ConstraintNotMetError(msg)

            transformed_data = constraint.transform(data=transformed_data)

        for table_name, table_data in transformed_data.items():
            synthesizer = self._table_synthesizers[table_name]
            try:
                synthesizer.validate_constraints(table_data)
            except ConstraintNotMetError as error:
                raise ConstraintNotMetError(f"Table '{table_name}': {error}") from error

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
            error_msg = f"Unrecognized version '{version}', please use 'original' or 'modified'."
            raise ValueError(error_msg)

        if version == 'original' and hasattr(self, '_original_metadata'):
            return Metadata.load_from_dict(self._original_metadata.to_dict())

        return Metadata.load_from_dict(self.metadata.to_dict())

    def _validate_transform_constraints(self, data, enforce_constraint_fitting=False):
        """Validate and transform all constraints during preprocessing.

        Args:
            data (dict[str, pd.DataFrame]):
                The data dictionary.
            enforce_constraint_fitting (bool):
                Whether to enforce fitting the constraints again. If set to ``True``, the
                constraints will be fitted again even if they have already been fitted.
                Defaults to ``False``.
        """
        if self._constraints_fitted and not enforce_constraint_fitting:
            for constraint in self.constraints:
                data = constraint.transform(data)

            return data

        metadata = self._original_metadata
        for constraint in self.constraints:
            constraint.fit(data=data, metadata=metadata)
            metadata = constraint.get_updated_metadata(metadata)
            data = constraint.transform(data)

        self._constraints_fitted = True
        return data

    def _reverse_transform_constraints(self, sampled_data):
        """Reverse transform constraints after sampling."""
        if not hasattr(self, 'constraints'):
            return sampled_data

        for constraint in reversed(self.constraints):
            sampled_data = constraint.reverse_transform(sampled_data)

        from sdv.utils.utils import (
            drop_unknown_references,  # noqa: F401 Lazy import to fix circular dependency
        )

        # issue
        sampled_data = drop_unknown_references(sampled_data, self._original_metadata, verbose=False)
        return sampled_data

    def get_table_parameters(self, table_name):
        """Return the parameters for the given table's synthesizer.

        Args:
            table_name (str):
                Table name for which the parameters should be retrieved.

        Returns:
            parameters (dict):
                A dictionary with the following structure:
                {
                    'synthesizer_name': the string name of the synthesizer for that table,
                    'synthesizer_parameters': the parameters used to instantiate the synthesizer
                }
        """
        table_synthesizer = self._table_synthesizers.get(table_name)
        if not table_synthesizer:
            table_params = {'synthesizer_name': None, 'synthesizer_parameters': {}}
        else:
            table_params = {
                'synthesizer_name': type(table_synthesizer).__name__,
                'synthesizer_parameters': table_synthesizer.get_parameters(),
            }

        return table_params

    def get_parameters(self):
        """Return the parameters used to instantiate the synthesizer and all table synthesizers.

        Returns:
            parameters (dict):
                A dictionary representing the parameters used to instantiate the synthesizer.
        """
        parameters = inspect.signature(self.__init__).parameters
        instantiated_parameters = {}
        for parameter_name in parameters:
            if parameter_name != 'metadata':
                instantiated_parameters[parameter_name] = self.__dict__.get(parameter_name)

        return instantiated_parameters

    def set_table_parameters(self, table_name, table_parameters):
        """Update the table's synthesizer instantiation parameters.

        Args:
            table_name (str):
                Table name for which the parameters should be retrieved.
            table_parameters (dict):
                A dictionary with the parameters as keys and the values to be used to instantiate
                the table's synthesizer.
        """
        table_metadata = self.metadata.get_table_metadata(table_name)
        self._table_synthesizers[table_name] = self._synthesizer(
            metadata=table_metadata, **table_parameters
        )
        self._table_parameters[table_name].update(deepcopy(table_parameters))

    def _validate_all_tables(self, data):
        """Validate every table of the data has a valid table/metadata pair."""
        errors = []
        for table_name, table_data in data.items():
            try:
                self._table_synthesizers[table_name].validate(table_data)

            except InvalidDataError as error:
                error_msg = f"Table: '{table_name}'"
                for _error in error.errors:
                    error_msg += f'\nError: {_error}'

                errors.append(error_msg)

            except ValueError as error:
                errors.append(str(error))

            except KeyError:
                continue

        return errors

    def validate(self, data):
        """Validate the data.

        Validate that the metadata matches the data and thta every table's constraints are valid.

        Args:
            data (dict[str, pd.DataFrame]):
                A dictionary of table names to pd.DataFrames.
        """
        errors = []
        metadata = self._original_metadata
        metadata.validate_data(data)
        for table_name in data:
            if table_name in self._table_synthesizers:
                # Validate rules specific to each synthesizer
                errors += self._table_synthesizers[table_name]._validate(data[table_name])

        if errors:
            raise InvalidDataError(errors)

        self._validate_transform_constraints(data, enforce_constraint_fitting=True)

    def _validate_table_name(self, table_name):
        if table_name not in self._table_synthesizers:
            raise ValueError(
                'The provided data does not match the metadata:'
                f"\nTable '{table_name}' is not present in the metadata."
            )

    def _assign_table_transformers(self, synthesizer, table_name, table_data):
        """Update the ``synthesizer`` to ignore the foreign keys while preprocessing the data."""
        synthesizer.auto_assign_transformers(table_data)
        foreign_key_columns = self.metadata._get_all_foreign_keys(table_name)
        column_name_to_transformers = {column_name: None for column_name in foreign_key_columns}
        synthesizer.update_transformers(column_name_to_transformers)

    def auto_assign_transformers(self, data):
        """Automatically assign the required transformers for the given data and constraints.

        This method will automatically set a configuration to the ``rdt.HyperTransformer``
        with the required transformers for the current data.

        Args:
            data (dict[str, pd.DataFrame]):
                Mapping of table name to pandas.DataFrame.

        Raises:
            InvalidDataError:
                If a table of the data is not present in the metadata.
        """
        for table_name, table_data in data.items():
            self._validate_table_name(table_name)
            synthesizer = self._table_synthesizers[table_name]
            self._assign_table_transformers(synthesizer, table_name, table_data)

    def get_transformers(self, table_name):
        """Get a dictionary mapping of ``column_name`` and ``rdt.transformers``.

        A dictionary representing the column names and the transformers that will be used
        to transform the data.

        Args:
            table_name (string):
                The name of the table of which to get the transformers.

        Returns:
            dict:
                A dictionary mapping with column names and transformers.

        Raises:
            ValueError:
                If ``table_name`` is not present in the metadata.
        """
        self._validate_table_name(table_name)
        return self._table_synthesizers[table_name].get_transformers()

    def update_transformers(self, table_name, column_name_to_transformer):
        """Update any of the transformers assigned to each of the column names.

        Args:
            table_name (string):
                The name of the table of which to update the transformers.
            column_name_to_transformer (dict):
                Dict mapping column names to transformers to be used for that column.

        Raises:
            ValueError:
                If ``table_name`` is not present in the metadata.
        """
        self._validate_table_name(table_name)
        self._table_synthesizers[table_name].update_transformers(column_name_to_transformer)

    def _store_and_convert_original_cols(self, data):
        list_of_changed_tables = []
        for table, dataframe in data.items():
            data_columns = dataframe.columns
            col_name_mapping = {str(col): col for col in data_columns}
            reverse_col_name_mapping = {col: str(col) for col in data_columns}
            self._original_table_columns[table] = col_name_mapping
            dataframe = dataframe.rename(columns=reverse_col_name_mapping)
            for column in data_columns:
                if isinstance(column, int):
                    list_of_changed_tables.append(table)
                    break

            data[table] = dataframe

        return list_of_changed_tables

    def preprocess(self, data):
        """Transform the raw data to numerical space.

        Args:
            data (dict[str, pd.DataFrame]):
                Dictionary mapping each table name to a ``pandas.DataFrame``.

        Returns:
            dict:
                A dictionary with the preprocessed data.
        """
        list_of_changed_tables = self._store_and_convert_original_cols(data)
        self.validate(data)
        data = self._validate_transform_constraints(data)
        if self._fitted:
            warnings.warn(
                'This model has already been fitted. To use the new preprocessed data, '
                "please refit the model using 'fit' or 'fit_processed_data'."
            )

        processed_data = {}
        pbar_args = self._get_pbar_args(desc='Preprocess Tables')
        for table_name, table_data in tqdm(data.items(), **pbar_args):
            synthesizer = self._table_synthesizers[table_name]
            self._assign_table_transformers(synthesizer, table_name, table_data)
            table_data = synthesizer._validate_transform_constraints(table_data)
            processed_data[table_name] = synthesizer._preprocess(table_data)

        for table in list_of_changed_tables:
            data[table] = data[table].rename(columns=self._original_table_columns[table])

        return processed_data

    def _model_tables(self, augmented_data):
        """Model the augmented tables.

        Args:
            augmented_data (dict):
                Dictionary mapping each table name to an augmented ``pandas.DataFrame``.
        """
        raise NotImplementedError()

    def _augment_tables(self, processed_data):
        """Augment the processed data.

        Args:
            processed_data (dict):
                Dictionary mapping each table name to a preprocessed ``pandas.DataFrame``.
        """
        raise NotImplementedError()

    def fit_processed_data(self, processed_data):
        """Fit this model to the transformed data.

        Args:
            processed_data (dict):
                Dictionary mapping each table name to a preprocessed ``pandas.DataFrame``.
        """
        total_rows = 0
        total_columns = 0
        for table in processed_data.values():
            total_rows += len(table)
            total_columns += len(table.columns)

        SYNTHESIZER_LOGGER.info({
            'EVENT': 'Fit processed data',
            'TIMESTAMP': datetime.datetime.now(),
            'SYNTHESIZER CLASS NAME': self.__class__.__name__,
            'SYNTHESIZER ID': self._synthesizer_id,
            'TOTAL NUMBER OF TABLES': len(processed_data),
            'TOTAL NUMBER OF ROWS': total_rows,
            'TOTAL NUMBER OF COLUMNS': total_columns,
        })

        check_synthesizer_version(self, is_fit_method=True, compare_operator=operator.lt)
        with disable_single_table_logger():
            augmented_data = self._augment_tables(processed_data)
            self._model_tables(augmented_data)

        self._fitted = True
        self._fitted_date = datetime.datetime.today().strftime('%Y-%m-%d')
        self._fitted_sdv_version = getattr(version, 'public', None)
        self._fitted_sdv_enterprise_version = getattr(version, 'enterprise', None)

    def fit(self, data):
        """Fit this model to the original data.

        Args:
            data (dict):
                Dictionary mapping each table name to a ``pandas.DataFrame`` in the raw format
                (before any transformations).
        """
        total_rows = 0
        total_columns = 0
        for table in data.values():
            total_rows += len(table)
            total_columns += len(table.columns)

        SYNTHESIZER_LOGGER.info({
            'EVENT': 'Fit',
            'TIMESTAMP': datetime.datetime.now(),
            'SYNTHESIZER CLASS NAME': self.__class__.__name__,
            'SYNTHESIZER ID': self._synthesizer_id,
            'TOTAL NUMBER OF TABLES': len(data),
            'TOTAL NUMBER OF ROWS': total_rows,
            'TOTAL NUMBER OF COLUMNS': total_columns,
        })

        check_synthesizer_version(self, is_fit_method=True, compare_operator=operator.lt)
        self._check_metadata_updated()
        self._fitted = False
        processed_data = self.preprocess(data)
        self._print(text='\n', end='')
        self.fit_processed_data(processed_data)

    def reset_sampling(self):
        """Reset the sampling to the state that was left right after fitting."""
        self._numpy_seed = 73251
        for synthesizer in self._table_synthesizers.values():
            synthesizer.reset_sampling()

    def _sample(self, scale):
        raise NotImplementedError()

    def sample(self, scale=1.0):
        """Generate synthetic data for the entire dataset.

        Args:
            scale (float):
                A float representing how much to scale the data by. If scale is set to ``1.0``,
                this does not scale the sizes of the tables. If ``scale`` is greater than ``1.0``
                create more rows than the original data by a factor of ``scale``.
                If ``scale`` is lower than ``1.0`` create fewer rows by the factor of ``scale``
                than the original tables. Defaults to ``1.0``.
        """
        if not self._fitted:
            raise SamplingError(
                'This synthesizer has not been fitted. Please fit your synthesizer first before '
                'sampling synthetic data.'
            )

        if type(scale) not in (float, int) or not scale > 0:
            raise SynthesizerInputError(
                f"Invalid parameter for 'scale' ({scale}). Please provide a number that is >0.0."
            )

        with self._set_temp_numpy_seed(), disable_single_table_logger():
            sampled_data = self._sample(scale=scale)

        total_rows = 0
        total_columns = 0
        for table in sampled_data.values():
            total_rows += len(table)
            total_columns += len(table.columns)

        table_columns = getattr(self, '_original_table_columns', {})

        for table in sampled_data:
            table_data = sampled_data[table][self.get_metadata().get_column_names(table)]
            if table in table_columns:
                if isinstance(table_columns[table], dict):
                    table_data = table_data.rename(columns=table_columns[table])
                else:
                    table_data.columns = table_columns[table]

            sampled_data[table] = table_data

        SYNTHESIZER_LOGGER.info({
            'EVENT': 'Sample',
            'TIMESTAMP': datetime.datetime.now(),
            'SYNTHESIZER CLASS NAME': self.__class__.__name__,
            'SYNTHESIZER ID': self._synthesizer_id,
            'TOTAL NUMBER OF TABLES': len(sampled_data),
            'TOTAL NUMBER OF ROWS': total_rows,
            'TOTAL NUMBER OF COLUMNS': total_columns,
        })

        return sampled_data

    def get_learned_distributions(self, table_name):
        """Get the marginal distributions used by the ``GaussianCopula`` for a table.

        Return a dictionary mapping the column names with the distribution name and the learned
        parameters for those.

        Args:
            table_name (str):
                Table name for which the parameters should be retrieved.

        Returns:
            dict:
                Dictionary containing the distributions used or detected for each column and the
                learned parameters for those.
        """
        synthesizer = self._table_synthesizers[table_name]
        if hasattr(synthesizer, 'get_learned_distributions'):
            return synthesizer.get_learned_distributions()

        raise SynthesizerInputError(
            f"Learned distributions are not available for the '{table_name}' "
            f"table because it uses the '{synthesizer.__class__.__name__}'."
        )

    def get_loss_values(self, table_name):
        """Get the loss values from a model for a table.

        Return a pandas dataframe mapping of the loss values per epoch of GAN
        based synthesizers

        Args:
            table_name (str):
                Table name for which the parameters should be retrieved.

        Returns:
            pd.DataFrame:
                Dataframe of loss values per epoch
        """
        if table_name not in self._table_synthesizers:
            raise ValueError(f"Table '{table_name}' is not present in the metadata.")

        synthesizer = self._table_synthesizers[table_name]
        if hasattr(synthesizer, 'get_loss_values'):
            return synthesizer.get_loss_values()

        raise SynthesizerInputError(
            f"Loss values are not available for table '{table_name}' "
            'because the table does not use a GAN-based model.'
        )

    def load_custom_constraint_classes(self, filepath, class_names):
        """Load a custom constraint class for each table's synthesizer.

        Args:
            filepath (str):
                String representing the absolute or relative path to the python file where
                the custom constraints are declared.
            class_names (list):
                A list of custom constraint classes to be imported.
        """
        for synthesizer in self._table_synthesizers.values():
            synthesizer.load_custom_constraint_classes(filepath, class_names)

    def get_info(self):
        """Get dictionary with information regarding the synthesizer.

        Return:
            dict:
                * ``class_name``: synthesizer class name
                * ``creation_date``: date of creation
                * ``is_fit``: whether or not the synthesizer has been fit
                * ``last_fit_date``: date for the last time it was fit
                * ``fitted_sdv_version``: version of sdv it was on when fitted
        """
        info = {
            'class_name': self.__class__.__name__,
            'creation_date': self._creation_date,
            'is_fit': self._fitted,
            'last_fit_date': self._fitted_date,
            'fitted_sdv_version': self._fitted_sdv_version,
        }
        if self._fitted_sdv_enterprise_version:
            info['fitted_sdv_enterprise_version'] = self._fitted_sdv_enterprise_version

        return info

    def _validate_fit_before_save(self):
        """Validate that the synthesizer has been fitted before saving."""
        if not self._fitted:
            warnings.warn(
                'You are saving a synthesizer that has not yet been fitted. You will not be able '
                'to sample synthetic data without fitting. We recommend fitting the synthesizer '
                'first and then saving.'
            )

    def save(self, filepath):
        """Save this instance to the given path using cloudpickle.

        Args:
            filepath (str):
                Path where the instance will be serialized.
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
        """Load a multi-table synthesizer from a given path.

        Args:
            filepath (str):
                A string describing the filepath of your saved synthesizer.

        Returns:
            MultiTableSynthesizer:
                The loaded synthesizer.
        """
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
