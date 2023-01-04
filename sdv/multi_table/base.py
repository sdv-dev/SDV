"""Base Multi Table Synthesizer class."""
import contextlib
import warnings
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd

from sdv.single_table.copulas import GaussianCopulaSynthesizer
from sdv.single_table.errors import InvalidDataError


class BaseMultiTableSynthesizer:
    """Base class for multi table synthesizers.

    The ``BaseMultiTableSynthesizer`` class defines the common API that all the
    multi table synthesizers need to implement, as well as common functionality.

    Args:
        metadata (sdv.metadata.multi_table.MultiTableMetadata):
            Multi table metadata representing the data tables that this synthesizer will be used
            for.
    """

    _synthesizer = GaussianCopulaSynthesizer
    _numpy_seed = 73251

    @contextlib.contextmanager
    def _set_temp_numpy_seed(self):
        initial_state = np.random.get_state()
        if isinstance(self._numpy_seed, int):
            np.random.seed(self._numpy_seed)
        else:
            np.random.set_state(self._numpy_seed)

        try:
            yield
        finally:
            self._numpy_seed = np.random.get_state()
            np.random.set_state(initial_state)

    def _initialize_models(self):
        for table_name, table_metadata in self.metadata._tables.items():
            synthesizer_parameters = self._table_parameters.get(table_name, {})
            self._table_synthesizers[table_name] = self._synthesizer(
                metadata=table_metadata,
                **synthesizer_parameters
            )

    def __init__(self, metadata):
        self.metadata = metadata
        self.metadata.validate()
        self._table_synthesizers = {}
        self._table_parameters = defaultdict(dict)
        self._initialize_models()
        self._fitted = False

    def get_table_parameters(self, table_name):
        """Return the parameters that will be used to instantiate the table's synthesizer.

        Args:
            table_name (str):
                Table name for which the parameters should be retrieved.

        Returns:
            parameters (dict):
                A dictionary representing the parameters that will be used to instantiate the
                table's synthesizer.
        """
        return self._table_parameters.get(table_name, {})

    def get_parameters(self, table_name):
        """Return the parameters used to instantiate the table's synthesizer.

        Args:
            table_name (str):
                Table name for which the parameters should be retrieved.

        Returns:
            parameters (dict):
                A dictionary representing the parameters used to instantiate the table's
                synthesizer.
        """
        return self._table_synthesizers.get(table_name).get_parameters()

    def update_table_parameters(self, table_name, table_parameters):
        """Update the table's synthesizer instantiation parameters.

        Args:
            table_name (str):
                Table name for which the parameters should be retrieved.
            table_parameters (dict):
                A dictionary with the parameters as keys and the values to be used to instantiate
                the table's synthesizer.
        """
        self._table_synthesizers[table_name] = self._synthesizer(
            metadata=self.metadata._tables[table_name],
            **table_parameters
        )
        self._table_parameters[table_name].update(deepcopy(table_parameters))

    def get_metadata(self):
        """Return the ``MultiTableMetadata`` for this synthesizer."""
        return self.metadata

    def _validate_foreign_keys(self, data):
        error_msg = None
        errors = []
        for relation in self.metadata._relationships:
            child_table = data.get(relation['child_table_name'])
            parent_table = data.get(relation['parent_table_name'])
            if isinstance(child_table, pd.DataFrame) and isinstance(parent_table, pd.DataFrame):
                child_column = child_table[relation['child_foreign_key']]
                parent_column = parent_table[relation['parent_primary_key']]
                missing_values = child_column[~child_column.isin(parent_column)].unique()
                if any(missing_values):
                    message = ', '.join(missing_values[:5].astype(str))
                    if len(missing_values) > 5:
                        message = f'({message}, + more)'
                    else:
                        message = f'({message})'

                    errors.append(
                        f"Error: foreign key column '{relation['child_foreign_key']}' contains "
                        f'unknown references: {message}. All the values in this column must '
                        'reference a primary key.'
                    )
            if errors:
                error_msg = 'Relationships:\n'
                error_msg += '\n'.join(errors)

        return error_msg

    def validate(self, data):
        """Validate data.

        Args:
            data (dict):
                A dictionary with key as table name and ``pandas.DataFrame`` as value to validate.

        Raises:
            ValueError:
                Raised when data is not of type pd.DataFrame.
            InvalidDataError:
                Raised if:
                    * foreign key does not belong to a primay key
                    * data columns don't match metadata
                    * keys have missing values
                    * primary or alternate keys are not unique
                    * context columns vary for a sequence key
                    * values of a column don't satisfy their sdtype
        """
        errors = []
        missing_tables = set(self.metadata._tables) - set(data)
        if missing_tables:
            errors.append(f'The provided data is missing the tables {missing_tables}.')

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

        foreign_key_errors = self._validate_foreign_keys(data)
        if foreign_key_errors:
            errors.append(foreign_key_errors)

        if errors:
            raise InvalidDataError(errors)

    def _validate_table_name(self, table_name):
        if table_name not in self._table_synthesizers:
            raise InvalidDataError([f"Table '{table_name}' is not present in the metadata."])

    def auto_assign_transformers(self, data):
        """Automatically assign the required transformers for the given data and constraints.

        This method will automatically set a configuration to the ``rdt.HyperTransformer``
        with the required transformers for the current data.

        Args:
            data (dict):
                Mapping of table name to pandas.DataFrame.

        Raises:
            InvalidDataError:
                If a table of the data is not present in the metadata.
        """
        for table_name, table_data in data.items():
            self._validate_table_name(table_name)
            self._table_synthesizers[table_name].auto_assign_transformers(table_data)

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

    def _fit(self, processed_data):
        """Fit the model to the tables.

        Args:
            processed_data (dict):
                Dictionary mapping each table name to a preprocessed ``pandas.DataFrame``.
        """
        raise NotImplementedError()

    def preprocess(self, data):
        """Transform the raw data to numerical space.

        Args:
            data (dict):
                Dictionary mapping each table name to a ``pandas.DataFrame``.

        Returns:
            dict:
                A dictionary with the preprocessed data.
        """
        self.validate(data)
        if self._fitted:
            warnings.warn(
                'This model has already been fitted. To use the new preprocessed data, '
                "please refit the model using 'fit' or 'fit_processed_data'."
            )

        processed_data = {}
        for table_name, table_data in data.items():
            synthesizer = self._table_synthesizers[table_name]
            processed_data[table_name] = synthesizer.preprocess(table_data)

        return processed_data

    def fit_processed_data(self, processed_data):
        """Fit this model to the transformed data.

        Args:
            processed_data (dict):
                Dictionary mapping each table name to a preprocessed ``pandas.DataFrame``.
        """
        self._fit(processed_data)
        self._fitted = True

    def fit(self, data):
        """Fit this model to the original data.

        Args:
            data (dict):
                Dictionary mapping each table name to a ``pandas.DataFrame`` in the raw format
                (before any transformations).
        """
        self._fitted = False
        processed_data = self.preprocess(data)
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
        with self._set_temp_numpy_seed():
            sampled_data = self._sample(scale=scale)

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
        return synthesizer.get_learned_distributions()
