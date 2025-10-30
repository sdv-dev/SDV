"""Hierarchical Modeling Algorithms."""

import logging
import warnings
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
from rdt.transformers import FloatFormatter
from tqdm import tqdm

from sdv._utils import _get_root_tables
from sdv.errors import SynthesizerInputError
from sdv.multi_table.base import BaseMultiTableSynthesizer
from sdv.sampling import BaseHierarchicalSampler

LOGGER = logging.getLogger(__name__)
PERFORMANCE_ALERT_DISPLAY_CAP = 1_000_000
DEFAULT_EXTENDED_COLUMNS_DISTRIBUTION = 'truncnorm'
MAX_NUMBER_OF_COLUMNS = 1000


class HMASynthesizer(BaseHierarchicalSampler, BaseMultiTableSynthesizer):
    """Hierarchical Modeling Algorithm One.

    Args:
        metadata (sdv.metadata.Metadata):
            Metadata representing the data tables that this synthesizer will be used for.
        locales (list or str):
            The default locale(s) to use for AnonymizedFaker transformers.
            Defaults to ``['en_US']``.
        verbose (bool):
            Whether to print progress for fitting or not.
    """

    DEFAULT_SYNTHESIZER_KWARGS = {'default_distribution': 'beta'}
    DISTRIBUTIONS_TO_NUM_PARAMETER_COLUMNS = {
        'beta': 4,
        'truncnorm': 4,
        'gamma': 3,
        'norm': 2,
        'uniform': 2,
    }

    @staticmethod
    def _get_num_data_columns(metadata):
        """Get the number of data columns (i.e. columns that are not keys) for each table.

        Args:
            metadata (sdv.metadata.Metadata):
                Metadata representing the data tables that this synthesizer will be used for.
        """
        columns_per_table = {}
        for table_name, table in metadata.tables.items():
            key_columns = metadata._get_all_keys(table_name)
            num_data_columns = sum([
                1
                for col_name, col_meta in table.columns.items()
                if (
                    col_meta['sdtype'] != 'id'
                    or (col_name not in key_columns and col_meta.get('pii', False) is False)
                )
            ])
            num_extended_columns = 0
            columns_per_table[table_name] = [num_data_columns, num_extended_columns]

        return columns_per_table

    @classmethod
    def _get_num_extended_columns(
        cls, metadata, table_name, parent_table, columns_per_table, distributions=None
    ):
        """Get the number of columns that will be generated for table_name.

        A table generates, for each foreign key:
            - 1 num_rows column
            - n*(n-1)/2 correlation columns for each data column
            - k parameter columns for each data column, where:
                - k = 4 if the distribution is beta or truncnorm (params are a, b, loc, scale)
                - k = 3 if the distribution is gamma (params are a, loc, scale)
                - k = 2 if the distribution is norm or uniform (params are loc, scale)
        """
        if distributions is None:
            distribution = cls.DEFAULT_SYNTHESIZER_KWARGS['default_distribution']
        else:
            distribution = distributions.get(
                table_name, cls.DEFAULT_SYNTHESIZER_KWARGS['default_distribution']
            )

        num_params_data = cls.DISTRIBUTIONS_TO_NUM_PARAMETER_COLUMNS[distribution]
        num_params_extended = cls.DISTRIBUTIONS_TO_NUM_PARAMETER_COLUMNS[
            DEFAULT_EXTENDED_COLUMNS_DISTRIBUTION
        ]
        num_rows_columns = len(metadata._get_foreign_keys(parent_table, table_name))

        # no parameter columns are generated if there are no data or extended columns
        num_data_columns = columns_per_table[table_name][0]
        num_extended_columns = columns_per_table[table_name][1]

        if (num_data_columns + num_extended_columns) == 0:
            return num_rows_columns

        num_parameters_columns = (num_rows_columns * num_data_columns * num_params_data) + (
            num_rows_columns * num_extended_columns * num_params_extended
        )

        num_correlation_columns = (
            num_rows_columns
            * (num_data_columns + num_extended_columns - 1)
            * (num_data_columns + num_extended_columns)
            // 2
        )

        return num_correlation_columns + num_rows_columns + num_parameters_columns

    @classmethod
    def _estimate_columns_traversal(
        cls, metadata, table_name, columns_per_table, visited, distributions=None
    ):
        """Given a table, estimate how many columns each parent will model.

        This method recursively models the children of a table all the way to the leaf nodes.

        Args:
            table_name (str):
                Name of the table to estimate the number of columns for.
            columns_per_table (dict):
                Dict that stores the number of data columns + extended columns for each table.
            visited (set):
                Set of table names that have already been visited.
        """
        for child_name in metadata._get_child_map()[table_name]:
            if child_name not in visited:
                cls._estimate_columns_traversal(
                    metadata, child_name, columns_per_table, visited, distributions
                )

            columns_per_table[table_name][1] += cls._get_num_extended_columns(
                metadata, child_name, table_name, columns_per_table, distributions
            )

            total_cols = sum(columns_list[1] for columns_list in columns_per_table.values())
            if total_cols > PERFORMANCE_ALERT_DISPLAY_CAP:
                return

        visited.add(table_name)

    @classmethod
    def _estimate_num_columns(cls, metadata, distributions=None):
        """Estimate the number of columns that will be modeled for each table.

        This method estimates how many extended columns will be generated during the
        `_augment_tables` method, so it traverses the graph in the same way.
        If that method is ever changed, this should be updated to match.

        After running this method, `columns_per_table` will store an estimate of the
        total number of columns that each table has after running `_augment_tables`,
        that is, the number of extended columns generated by the child tables as well
        as the number of data columns in the table itself. `id` columns, like foreign
        and primary keys, are not counted since they are not modeled.

        Returns:
            dict:
                Dictionary of (table_name: int) mappings, indicating the estimated
                number of columns that will be modeled for each table.
        """
        # This dict will store the number of data columns + extended columns for each table
        # Initialize it with the number of data columns per table
        columns_per_table = cls._get_num_data_columns(metadata)

        # Starting at root tables, recursively estimate the number of columns
        # each table will model
        visited = set()
        for table_name in _get_root_tables(metadata.relationships):
            cls._estimate_columns_traversal(
                metadata, table_name, columns_per_table, visited, distributions
            )
            total_cols = sum(columns_list[1] for columns_list in columns_per_table.values())
            if total_cols > PERFORMANCE_ALERT_DISPLAY_CAP:
                break

        return {
            table_name: sum(columns_list) for table_name, columns_list in columns_per_table.items()
        }

    def __init__(self, metadata, locales=['en_US'], verbose=True):
        BaseMultiTableSynthesizer.__init__(self, metadata, locales=locales)
        self._table_sizes = {}
        self._max_child_rows = {}
        self._min_child_rows = {}
        self._null_child_synthesizers = {}
        self._augmented_tables = []
        self._learned_relationships = 0
        self._default_parameters = {}
        self._parent_extended_columns = defaultdict(list)
        self.verbose = verbose
        BaseHierarchicalSampler.__init__(
            self, self.metadata, self._table_synthesizers, self._table_sizes
        )
        child_tables = set()
        for relationship in metadata.relationships:
            child_tables.add(relationship['child_table_name'])
        for child_table_name in child_tables:
            self.set_table_parameters(child_table_name, {'default_distribution': 'norm'})
        self._print_estimate_warning()

    def set_table_parameters(self, table_name, table_parameters):
        """Update the table's synthesizer instantiation parameters.

        Args:
            table_name (str):
                Table name for which the parameters should be retrieved.
            table_parameters (dict):
                A dictionary with the parameters as keys and the values to be used to instantiate
                the table's synthesizer.
        """
        has_gaussian_kde = any(
            dist == 'gaussian_kde'
            for dist in table_parameters.get('numerical_distributions', {}).values()
        )
        if table_parameters.get('default_distribution') == 'gaussian_kde' or has_gaussian_kde:
            raise SynthesizerInputError(
                "The 'gaussian_kde' is not compatible with the HMA algorithm. Please choose a "
                "different distribution such as 'beta' or 'truncnorm'. Or try a different "
                'algorithm such as HSA.'
            )

        super().set_table_parameters(table_name, table_parameters)

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
        if table_name not in _get_root_tables(self.metadata.relationships):
            raise SynthesizerInputError(
                f"Learned distributions are not available for the '{table_name}' table. "
                'Please choose a table that does not have any parents.'
            )

        return super().get_learned_distributions(table_name)

    def _get_distributions(self):
        distributions = {}
        for table in self.metadata.tables:
            parameters = self.get_table_parameters(table)
            synthesizer_parameters = parameters.get('synthesizer_parameters', {})
            distributions[table] = synthesizer_parameters.get('default_distribution', None)

        return distributions

    def _print_estimate_warning(self):
        total_est_cols = 0
        metadata_columns = self._get_num_data_columns(self.metadata)
        print_table = []
        distributions = self._get_distributions()
        estimated_columns = self._estimate_num_columns(self.metadata, distributions)
        for table, est_cols in estimated_columns.items():
            entry = []
            entry.append(table)
            entry.append(sum(metadata_columns[table]))
            total_est_cols += est_cols
            entry.append(min(est_cols, PERFORMANCE_ALERT_DISPLAY_CAP))
            print_table.append(entry)

        if total_est_cols > MAX_NUMBER_OF_COLUMNS:
            display_total = (
                f'{PERFORMANCE_ALERT_DISPLAY_CAP}+'
                if total_est_cols > PERFORMANCE_ALERT_DISPLAY_CAP
                else f'{total_est_cols}'
            )
            self._print(
                'PerformanceAlert: Using the HMASynthesizer on this metadata '
                'schema is not recommended. To model this data, HMA will '
                f'generate a large number of columns. ({display_total} columns)\n\n'
            )
            self._print(
                pd.DataFrame(
                    print_table, columns=['Table Name', '# Columns in Metadata', 'Est # Columns']
                ).to_string(index=False)
                + '\n'
            )
            self._print(
                'We recommend simplifying your metadata schema using '
                "'sdv.utils.poc.simplify_schema'.\nIf this is not possible, please visit "
                'datacebo.com and reach out to us for enterprise solutions.\n'
            )

    def preprocess(self, data):
        """Transform the raw data to numerical space.

        Args:
            data (dict):
                Dictionary mapping each table name to a ``pandas.DataFrame``.

        Returns:
            dict:
                A dictionary with the preprocessed data.
        """
        processed_data = super().preprocess(data)
        for _, synthesizer in self._table_synthesizers.items():
            synthesizer.reset_sampling()

        return processed_data

    def _set_extended_columns_distributions(self, synthesizer, table_name, valid_columns):
        numerical_distributions = {}
        for extended_column in self._parent_extended_columns[table_name]:
            if extended_column in valid_columns:
                numerical_distributions[extended_column] = DEFAULT_EXTENDED_COLUMNS_DISTRIBUTION

        if numerical_distributions:
            existing = getattr(synthesizer, 'numerical_distributions', {}) or {}
            merged = {**existing, **numerical_distributions}
            synthesizer._set_numerical_distributions(merged)

    def _get_extension(self, child_name, child_table, foreign_key, progress_bar_desc):
        """Generate the extension columns for this child table.

        The resulting dataframe will have an index that contains all the foreign key values.
        The values for a given index are generated by flattening a synthesizer fitted with
        the child rows with that foreign key value.

        Args:
            child_name (str):
                Name of the child table.
            child_table (pandas.DataFrame):
                Data for the child table.
            foreign_key (str):
                Name of the foreign key field.
            progress_bar_desc (str):
                Progress bar description.

        Returns:
            pandas.DataFrame
        """
        table_meta = self._table_synthesizers[child_name].get_metadata()

        extension_rows = []
        foreign_key_columns = self.metadata._get_all_foreign_keys(child_name)
        foreign_key_values = child_table[foreign_key].unique()
        child_table = child_table.set_index(foreign_key)

        index = []
        scale_columns = None
        pbar_args = self._get_pbar_args(desc=progress_bar_desc)

        for foreign_key_value in tqdm(foreign_key_values, **pbar_args):
            try:
                child_rows = child_table.loc[[foreign_key_value]]
            except KeyError:
                # pre pandas 2.1 df.loc[[np.nan]] causes error
                if pd.isna(foreign_key_value):
                    child_rows = child_table[child_table.index.isna()]
                else:
                    raise
            child_rows = child_rows[child_rows.columns.difference(foreign_key_columns)]
            try:
                if child_rows.empty and not pd.isna(foreign_key_value):
                    row = pd.Series({'num_rows': len(child_rows)})
                    row.index = f'__{child_name}__{foreign_key}__' + row.index
                else:
                    synthesizer = self._synthesizer(
                        table_meta,
                        **self._table_parameters[child_name],
                    )
                    self._set_extended_columns_distributions(
                        synthesizer, child_name, child_rows.columns
                    )
                    if not child_rows.empty:
                        synthesizer.fit_processed_data(child_rows.reset_index(drop=True))
                        row = synthesizer._get_parameters()
                        row = pd.Series(row)
                        row.index = f'__{child_name}__{foreign_key}__' + row.index

                    if not pd.isna(foreign_key_value):
                        if scale_columns is None:
                            scale_columns = [
                                column for column in row.index if column.endswith('scale')
                            ]

                        if len(child_rows) == 1:
                            row.loc[scale_columns] = None

                if pd.isna(foreign_key_value):
                    self._null_child_synthesizers[f'__{child_name}__{foreign_key}'] = synthesizer
                else:
                    extension_rows.append(row)
                    index.append(foreign_key_value)
            except Exception:
                # Skip children rows subsets that fail
                pass

        return pd.DataFrame(extension_rows, index=index)

    @staticmethod
    def _clear_nans(table_data, ignore_cols=None):
        columns = set(table_data.columns)
        if ignore_cols is not None:
            columns = columns - set(ignore_cols)
        for column in columns:
            column_data = table_data[column]
            if column_data.dtype in (int, float):
                fill_value = 0 if column_data.isna().all() else column_data.mean()
            else:
                fill_value = column_data.mode()[0]

            table_data[column] = table_data[column].fillna(fill_value)

    def _augment_table(self, table, tables, table_name):
        """Recursively generate the extension columns for the tables in the graph.

        For each of the table's foreign keys, generate the related extension columns,
        and extend the provided table. Generate them first for the top level tables,
        then their children, and so on.

        Args:
            table (pandas.DataFrame):
                The table to extend.
            tables (dict):
                A dictionary mapping table_name to table data (pandas.DataFrame).
            table_name (str):
                The name of the table.

        Returns:
            pandas.DataFrame:
                The extended table.
        """
        self._table_sizes[table_name] = len(table)
        LOGGER.info('Computing extensions for table %s', table_name)
        children = self.metadata._get_child_map()[table_name]
        for child_name in children:
            if child_name not in self._augmented_tables:
                child_table = self._augment_table(tables[child_name], tables, child_name)
            else:
                child_table = tables[child_name]

            foreign_keys = self.metadata._get_foreign_keys(table_name, child_name)
            for foreign_key in foreign_keys:
                progress_bar_desc = (
                    f'({self._learned_relationships + 1}/{len(self.metadata.relationships)}) '
                    f"Tables '{table_name}' and '{child_name}' ('{foreign_key}')"
                )
                extension = self._get_extension(
                    child_name, child_table.copy(), foreign_key, progress_bar_desc
                )
                for column in extension.columns:
                    extension[column] = extension[column].astype(float)
                    if extension[column].isna().all():
                        extension[column] = extension[column].fillna(1e-6)

                    self.extended_columns[child_name][column] = FloatFormatter(
                        enforce_min_max_values=True
                    )
                    self.extended_columns[child_name][column].fit(extension, column)
                table = table.merge(extension, how='left', right_index=True, left_index=True)
                num_rows_key = f'__{child_name}__{foreign_key}__num_rows'
                table[num_rows_key] = table[num_rows_key].fillna(0)
                self._max_child_rows[num_rows_key] = table[num_rows_key].max()
                self._min_child_rows[num_rows_key] = table[num_rows_key].min()
                self._null_foreign_key_percentages[f'__{child_name}__{foreign_key}'] = 1 - (
                    table[num_rows_key].sum() / child_table.shape[0]
                )

                if len(extension.columns) > 0:
                    self._parent_extended_columns[table_name].extend(list(extension.columns))

                tables[table_name] = table
                self._learned_relationships += 1
        self._augmented_tables.append(table_name)

        foreign_keys = self.metadata._get_all_foreign_keys(table_name)
        self._clear_nans(table, ignore_cols=foreign_keys)

        return table

    def _augment_tables(self, processed_data):
        """Fit this ``HMASynthesizer`` instance to the dataset data.

        Args:
            processed_data (dict):
                Dictionary mapping each table name to a preprocessed ``pandas.DataFrame``.
        """
        augmented_data = deepcopy(processed_data)
        self._augmented_tables = []
        self._learned_relationships = 0
        parent_map = self.metadata._get_parent_map()
        self._print(text='Learning relationships:')
        for table_name in processed_data:
            if not parent_map.get(table_name):
                self._augment_table(augmented_data[table_name], augmented_data, table_name)

        LOGGER.info('Augmentation Complete')
        return augmented_data

    def _pop_foreign_keys(self, table_data, table_name):
        """Remove foreign keys from the ``table_data``.

        Args:
            table_data (pd.DataFrame):
                The table that contains the ``foreign_keys``.
            table_name (str):
                The name representing the table.

        Returns:
            keys (dict):
                A dictionary mapping with the foreign key and it's values within the table.
        """
        foreign_keys = self.metadata._get_all_foreign_keys(table_name)
        keys = {}
        for fk in foreign_keys:
            keys[fk] = table_data.pop(fk).to_numpy()
        return keys

    def _model_tables(self, augmented_data):
        """Model the augmented tables.

        Args:
            augmented_data (dict):
                Dictionary mapping each table name to an augmented ``pandas.DataFrame``.
        """
        augmented_data_to_model = [
            (table_name, table) for table_name, table in augmented_data.items()
        ]
        self._print(text='\n', end='')
        pbar_args = self._get_pbar_args(desc='Modeling Tables')
        for table_name, table in tqdm(augmented_data_to_model, **pbar_args):
            keys = self._pop_foreign_keys(table, table_name)
            self._clear_nans(table)
            LOGGER.info(
                'Fitting %s for table %s; shape: %s',
                self._synthesizer.__name__,
                table_name,
                table.shape,
            )

            if not table.empty:
                self._set_extended_columns_distributions(
                    self._table_synthesizers[table_name], table_name, table.columns
                )
                self._table_synthesizers[table_name].fit_processed_data(table)
                table_parameters = self._table_synthesizers[table_name]._get_parameters()
                self._default_parameters[table_name] = {
                    parameter: value
                    for parameter, value in table_parameters.items()
                    if 'univariates' in parameter
                }

            for fk_column_name, fk_values in keys.items():
                table[fk_column_name] = fk_values

    def _extract_parameters(self, parent_row, table_name, foreign_key):
        """Get the params from a generated parent row.

        Args:
            parent_row (pandas.Series):
                A generated parent row.
            table_name (str):
                Name of the table to make the synthesizer for.
            foreign_key (str):
                Name of the foreign key used to form this
                parent child relationship.
        """
        prefix = f'__{table_name}__{foreign_key}__'
        keys = [key for key in parent_row.keys() if key.startswith(prefix)]
        new_keys = {key: key[len(prefix) :] for key in keys}
        flat_parameters = parent_row[keys].astype(float).fillna(1e-6)

        num_rows_key = f'{prefix}num_rows'
        if num_rows_key in flat_parameters:
            num_rows = flat_parameters[num_rows_key]
            flat_parameters[num_rows_key] = min(self._max_child_rows[num_rows_key], round(num_rows))

        flat_parameters = flat_parameters.to_dict()
        for parameter_name, parameter in flat_parameters.items():
            float_formatter = self.extended_columns[table_name][parameter_name]
            flat_parameters[parameter_name] = np.clip(  # this should be revisited in GH#1769
                parameter, float_formatter._min_value, float_formatter._max_value
            )

        return {new_keys[key]: value for key, value in flat_parameters.items()}

    def _recreate_child_synthesizer(self, child_name, parent_name, parent_row):
        # A child table is created based on only one foreign key.
        foreign_key = self.metadata._get_foreign_keys(parent_name, child_name)[0]

        if parent_row is not None:
            parameters = self._extract_parameters(parent_row, child_name, foreign_key)
            default_parameters = getattr(self, '_default_parameters', {}).get(child_name, {})
            table_meta = self.metadata.get_table_metadata(child_name)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore', message=".*The 'SingleTableMetadata' is deprecated.*"
                )
                synthesizer = self._synthesizer(table_meta, **self._table_parameters[child_name])

            extended_columns = getattr(self, '_parent_extended_columns', {}).get(child_name, [])
            if extended_columns:
                self._set_extended_columns_distributions(synthesizer, child_name, extended_columns)
            synthesizer._set_parameters(parameters, default_parameters)
        else:
            synthesizer = self._null_child_synthesizers[f'__{child_name}__{foreign_key}']

        synthesizer._data_processor = self._table_synthesizers[child_name]._data_processor
        if hasattr(self._table_synthesizers[child_name], '_chained_constraints') and hasattr(
            self._table_synthesizers[child_name], '_reject_sampling_constraints'
        ):
            synthesizer._chained_constraints = self._table_synthesizers[
                child_name
            ]._chained_constraints
            synthesizer._reject_sampling_constraints = self._table_synthesizers[
                child_name
            ]._reject_sampling_constraints

        return synthesizer

    @staticmethod
    def _find_parent_id(likelihoods, num_rows):
        """Find the parent id for one row based on the likelihoods of parent id values.

        If likelihoods are invalid, fall back to the num_rows.

        Args:
            likelihoods (pandas.Series):
                The likelihood of parent id values.
            num_rows (pandas.Series):
                The number of times each parent id value appears in the data.

        Returns:
            int:
                The parent id for this row, chosen based on likelihoods.
        """
        mean = likelihoods.mean()
        if (likelihoods == 0).all():
            # All rows got 0 likelihood, fallback to num_rows
            likelihoods = num_rows
        elif pd.isna(mean) or mean == 0:
            # Some rows got singular matrix error and the rest were 0
            # Fallback to num_rows on the singular matrix rows and
            # keep 0s on the rest.
            likelihoods = likelihoods.astype(float).fillna(num_rows)
        else:
            # at least one row got a valid likelihood, so fill the
            # rows that got a singular matrix error with the mean
            likelihoods = likelihoods.fillna(mean)

        total = likelihoods.sum()
        if total == 0:
            # Worse case scenario: we have no likelihoods
            # and all num_rows are 0, so we fallback to uniform
            length = len(likelihoods)
            weights = np.ones(length) / length
        else:
            weights = likelihoods.to_numpy() / total

        candidates, candidate_weights = [], []
        for parent, weight in zip(likelihoods.index.to_list(), weights):
            if num_rows[parent] > 0:
                candidates.append(parent)
                candidate_weights.append(weight)

        # cast candidates to series to ensure np.random.choice uses desired dtype
        candidates = pd.Series(candidates, dtype=likelihoods.index.dtype)

        # All available candidates were assigned 0 likelihood of being the parent id
        if sum(candidate_weights) == 0:
            chosen_parent = np.random.choice(candidates)
        else:
            candidate_weights = np.array(candidate_weights) / np.sum(candidate_weights)
            chosen_parent = np.random.choice(candidates, p=candidate_weights)

        num_rows[chosen_parent] -= 1

        return chosen_parent

    def _get_likelihoods(self, table_rows, parent_rows, table_name, foreign_key):
        """Calculate the likelihood of each parent id value appearing in the data.

        Args:
            table_rows (pandas.DataFrame):
                The rows in the child table.
            parent_rows (pandas.DataFrame):
                The rows in the parent table.
            table_name (str):
                The name of the child table.
            foreign_key (str):
                The foreign key column in the child table.

        Returns:
            pandas.DataFrame:
                A DataFrame of the likelihood of each parent id.
        """
        likelihoods = {}
        table_rows = table_rows.copy()

        data_processor = self._table_synthesizers[table_name]._data_processor
        transformed = data_processor.transform(table_rows)
        if transformed.index.name:
            table_rows = table_rows.set_index(transformed.index.name)

        columns_to_drop = [column for column in transformed.columns if column in table_rows.columns]
        table_rows = pd.concat([transformed, table_rows.drop(columns=columns_to_drop)], axis=1)
        for parent_id, row in parent_rows.iterrows():
            parameters = self._extract_parameters(row, table_name, foreign_key)
            table_meta = self._table_synthesizers[table_name].get_metadata()
            synthesizer = self._synthesizer(table_meta, **self._table_parameters[table_name])
            extended_columns = getattr(self, '_parent_extended_columns', {}).get(table_name, [])
            if extended_columns:
                self._set_extended_columns_distributions(synthesizer, table_name, extended_columns)
            synthesizer._set_parameters(parameters)
            try:
                likelihoods[parent_id] = synthesizer._get_likelihood(table_rows)

            except (AttributeError, np.linalg.LinAlgError):
                likelihoods[parent_id] = None

        null_child_synths = getattr(self, '_null_child_synthesizers', {})
        if f'__{table_name}__{foreign_key}' in null_child_synths:
            try:
                likelihoods[np.nan] = synthesizer._get_likelihood(table_rows)

            except (AttributeError, np.linalg.LinAlgError):
                likelihoods[np.nan] = None

        return pd.DataFrame(likelihoods, index=table_rows.index)

    def _find_parent_ids(self, child_table, parent_table, child_name, parent_name, foreign_key):
        """Find parent ids for the given table and foreign key.

        The parent ids are chosen randomly based on the likelihood of the available
        parent ids in the parent table.

        Args:
            child_table (pd.DataFrame):
                The child table dataframe.
            parent_table (pd.DataFrame):
                The parent table dataframe.
            child_name (str):
                The name of the child table.
            parent_name (dict):
                Map of table name to sampled data (pandas.DataFrame).
            foreign_key (str):
                The name of the foreign key column in the child table.

        Returns:
            pandas.Series:
                The parent ids for the given table data.
        """
        # Create a copy of the parent table with the primary key as index to calculate likelihoods
        primary_key = self.metadata.tables[parent_name].primary_key
        parent_table = parent_table.set_index(primary_key)
        num_rows = parent_table[f'__{child_name}__{foreign_key}__num_rows'].copy()
        num_rows.loc[np.nan] = child_table.shape[0] - num_rows.sum()

        likelihoods = self._get_likelihoods(child_table, parent_table, child_name, foreign_key)
        return likelihoods.apply(self._find_parent_id, axis=1, num_rows=num_rows)

    def _add_foreign_key_columns(self, child_table, parent_table, child_name, parent_name):
        for foreign_key in self.metadata._get_foreign_keys(parent_name, child_name):
            if foreign_key not in child_table:
                parent_ids = self._find_parent_ids(
                    child_table=child_table,
                    parent_table=parent_table,
                    child_name=child_name,
                    parent_name=parent_name,
                    foreign_key=foreign_key,
                )
                child_table[foreign_key] = parent_ids.to_numpy()
