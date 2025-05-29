"""FixedCombinations constraint."""

import uuid

import numpy as np
import pandas as pd

from sdv._utils import _create_unique_name
from sdv.cag._errors import ConstraintNotMetError
from sdv.cag._utils import (
    _get_is_valid_dict,
    _is_list_of_type,
    _remove_columns_from_metadata,
    _validate_table_and_column_names,
    _validate_table_name_if_defined,
)
from sdv.cag.base import BaseConstraint
from sdv.constraints.utils import get_mappable_combination


class FixedCombinations(BaseConstraint):
    """Constraint to ensure that the combinations across multiple columns are fixed.

    One simple example of this constraint can be found in a table that
    contains the columns `country` and `city`, where each country can
    have multiple cities and the same city name can even be found in
    multiple countries, but some combinations of country/city would
    produce invalid results.

    This constraint would ensure that the combinations of country/city
    found in the sampled data always stay within the combinations previously
    seen during training.

    Args:
        column_names (list[str]):
            Names of the columns that need to produce fixed combinations. Must
            contain at least two columns.
        table_name (str, optional):
            The name of the table that contains the columns. Optional if the
            data is only a single table. Defaults to None.
    """

    def __init__(self, column_names, table_name=None):
        super().__init__()
        if not _is_list_of_type(column_names):
            raise ValueError('`column_names` must be a list of strings.')

        if len(column_names) < 2:
            raise ValueError('FixedCombinations constraint requires at least two columns.')

        _validate_table_name_if_defined(table_name)

        self.column_names = column_names
        self.table_name = table_name
        self._joint_column = '#'.join(self.column_names)
        self._combinations = None

    def _validate_constraint_with_metadata(self, metadata):
        """Validate the constraint is compatible with the provided metadata.

        Validates that:
        - If `table_name` is None then the metadata contains only a single table
        - If `table_name` is not None, then the metadata contains a table with the same name.
        - All columns in `column_names` are present in the target table.
        - All columns in `column_names` have either 'categorical' or 'boolean' as their sdtype.
        - Columns relationships do not partially contain columns in `column_names`.
        """
        _validate_table_and_column_names(self.table_name, self.column_names, metadata)
        table_name = self._get_single_table_name(metadata)
        for column in self.column_names:
            col_sdtype = metadata.tables[table_name].columns[column]['sdtype']
            if col_sdtype not in ['boolean', 'categorical']:
                raise ConstraintNotMetError(
                    f"Column '{column}' has an incompatible sdtype ('{col_sdtype}'). The column "
                    "sdtype must be either 'boolean' or 'categorical'."
                )

        column_set = set(self.column_names)
        for column_relationship in metadata.tables[table_name].column_relationships:
            relationship_columns = set(column_relationship['column_names'])
            if not (
                column_set.issuperset(relationship_columns)
                or relationship_columns.isdisjoint(column_set)
            ):
                bad_columns = "', '".join(list(relationship_columns.intersection(column_set)))
                raise ConstraintNotMetError(
                    f"Cannot apply constraint because columns ['{bad_columns}'] are part of a "
                    'column relationship.'
                )

    def _validate_constraint_with_data(self, data, metadata):
        """Validate the data is compatible with the constraint.

        For FixedCombinations, this is a NOP.
        """
        return

    def _get_updated_metadata(self, metadata):
        """Get the new output metadata after applying the constraint to the input metadata."""
        table_name = self._get_single_table_name(metadata)
        combination_column = _create_unique_name(
            self._joint_column, metadata.tables[table_name].columns.keys()
        )

        metadata = metadata.to_dict()
        metadata['tables'][table_name]['columns'][combination_column] = {'sdtype': 'categorical'}
        return _remove_columns_from_metadata(
            metadata, table_name, columns_to_drop=self.column_names
        )

    def _fit(self, data, metadata):
        """Fit the constraint."""
        table_name = self._get_single_table_name(metadata)
        table_data = data[table_name]
        self._joint_column = _create_unique_name(
            self._joint_column, metadata.tables[table_name].columns.keys()
        )
        self._combinations = table_data[self.column_names].drop_duplicates().copy()
        self._combinations_to_uuids = {}
        self._uuids_to_combinations = {}
        for combination in self._combinations.itertuples(index=False, name=None):
            mappable_combination = get_mappable_combination(combination)
            uuid_str = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(mappable_combination)))
            self._combinations_to_uuids[mappable_combination] = uuid_str
            self._uuids_to_combinations[uuid_str] = mappable_combination

    def _transform(self, data):
        """Transform the data.

        The transformation consist on removing all the ``self.column_names`` from
        the table, and replacing them with a unique identifier that maps to
        that unique combination of column values under the previously computed
        combined column name.

        Args:
            data (dict[pd.DataFrame]):
                Table data.

        Returns:
            (dict[pd.DataFrame]):
                Transformed data.
        """
        table_name = self._get_single_table_name(self.metadata)
        table_data = data[table_name]
        # To make the NaN to None mapping work for pd.Categorical data, we need to convert
        # the columns to object before replacing NaNs with None.
        table_data[self.column_names] = table_data[self.column_names].astype({
            col: object
            for col in self.column_names
            if isinstance(table_data[col].dtype, pd.CategoricalDtype)
        })

        table_data[self.column_names] = table_data[self.column_names].replace({np.nan: None})
        combinations = table_data[self.column_names].itertuples(index=False, name=None)
        uuids = map(self._combinations_to_uuids.get, combinations)
        table_data[self._joint_column] = list(uuids)
        data[table_name] = table_data.drop(self.column_names, axis=1)
        return data

    def _reverse_transform(self, data):
        """Reverse transform the data.

        The transformation is reversed by popping the joint column from
        the table, mapping it back to the original combination of column values,
        and then setting all the columns back to the table with the original
        names.

        Args:
            data (dict[pd.DataFrame)]:
                Transformed data.

        Returns:
            dict[pd.DataFrame]:
                Reverse transformed data.
        """
        table_name = self._get_single_table_name(self.metadata)
        table_data = data[table_name]
        columns = table_data.pop(self._joint_column).map(self._uuids_to_combinations)

        for index, column in enumerate(self.column_names):
            table_data[column] = columns.str[index]

        return data

    def _is_valid(self, data):
        """Determine whether the data matches the constraint."""
        table_name = self._get_single_table_name(self.metadata)
        is_valid = _get_is_valid_dict(data, table_name)
        merged = data[table_name].merge(
            self._combinations, how='left', on=self.column_names, indicator=self._joint_column
        )
        is_valid[table_name] = merged[self._joint_column] == 'both'
        return is_valid
