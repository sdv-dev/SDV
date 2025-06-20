"""Programmable constraints base classes."""

from copy import deepcopy

from sdv.cag._errors import ConstraintNotMetError
from sdv.cag.base import BaseConstraint


class ProgrammableConstraint:
    """Base class to create programmable constraints."""

    _is_single_table = False

    def validate(self, metadata):
        """Validates that the metadata is compatible with the constraint and its parameters.

        Args:
            metadata (sdv.Metadata):
                The metadata to validate.

        Raises:
            Should raise an error if the metadata is incompatible with the constraint,
            e.g. an expected table/column is missing or has the wrong sdtype.
        """
        return

    def validate_input_data(self, data):
        """Validates that the data is compatible with the constraint and its parameters.

        Args:
            data (dict[pd.DataFrame]):
                The input data to validate.

        Raises:
            Should raise an error if the input data is incompatible with the constraint,
            e.g. a column has an unexpected value or incorrect dtype.
        """
        return

    def fit(self, data, metadata):
        """Given data and metadata, the constraint can learn additional info and save for later use.

        Args:
            metadata (sdv.Metadata):
                The metadata to fit on.
            data (dict[pd.DataFrame]):
                The input data to fit on.
        """
        return

    def transform(self, data):
        """Update the data in order to adhere to the constraint.

        Args:
            data (dict[pd.DataFrame]):
                The data to transform.

        Returns:
            dict[pd.DataFrame]:
                The transformed data.
        """
        raise NotImplementedError()

    def get_updated_metadata(self, metadata):
        """Returns how the metadata is updated as a result of transforming the data.

        Based on only the metadata and the parameters, the constraint should be able to
        determine how this metadata would update as a result of transforming the data.

        Args:
            metadata (sdv.Metadata):
                The input metadata.

        Returns:
            sdv.Metadata:
                The metadata for the transformed data.
        """
        raise NotImplementedError()

    def reverse_transform(self, transformed_data):
        """Given the transformed data, change it back into the original format.

        Args:
            transformed_data (dict[pd.DataFrame]):
                The data to reverse transform.

        Returns:
            dict[pd.DataFrame]:
                The reverse transformed data.
        """
        raise NotImplementedError()

    def is_valid(self, synthetic_data):
        """Given synthetic data, validate that it matches the constraint.

        Validate that the synthetic data is valid for the constraint by looking at
        the parameters as well as any learned information

        Args:
            synthetic_data (dict[pd.DataFrame]):
                The data to validate.

        Returns:
            dict[pd.Series]:
                For each table, a boolean Series indicating whether or not
                the corresponding row for the table is valid for this constraint.
        """
        raise NotImplementedError()

    def fix_data(self, synthetic_data):
        """Given synthetic data, fix it to match this constraint.

        Required for use with the DayZSynthesizer. Fixes the synthetic data so that
        it matches the constraint. You may modify the data or remove invalid data.
        This method should not require the constraint to be fit first.

        Args:
            synthetic_data (dict[pd.DataFrame]):
                The DayZ data to fix.

        Returns:
            dict[pd.DataFrame]:
                The “fixed” DayZ data so that it matches the constraint
        """
        return synthetic_data


class SingleTableProgrammableConstraint(ProgrammableConstraint):
    """Single table variant of the base programmable constraint class."""

    _is_single_table = True


class ProgrammableConstraintHarness(BaseConstraint):
    """Constraint interface class for programmable constraints."""

    def __init__(self, programmable_constraint):
        super().__init__()
        self.programmable_constraint = programmable_constraint
        self.table_name = None
        self._is_single_table = self.programmable_constraint._is_single_table

    def _validate_constraint_with_metadata(self, metadata):
        if self.programmable_constraint._is_single_table and len(metadata.tables) != 1:
            raise ConstraintNotMetError(
                'SingleTableProgrammableConstraint cannot be used with multi-table metadata. '
                'Please use the ProgrammableConstraint base class instead.'
            )

        self.programmable_constraint.validate(metadata)

    def _validate_constraint_with_data(self, data, metadata):
        if self._is_single_table:
            data = data[self._get_single_table_name(metadata)]

        self.programmable_constraint.validate_input_data(data)

    def _get_updated_metadata(self, metadata):
        metadata = deepcopy(metadata)
        return self.programmable_constraint.get_updated_metadata(metadata)

    def _fit(self, data, metadata):
        if self._is_single_table:
            data = data[self._table_name]

        self.programmable_constraint.fit(data, metadata)

    def _transform(self, data):
        if self._is_single_table:
            data = data[self._table_name]

        transformed = self.programmable_constraint.transform(data)

        if self._is_single_table:
            return {self._table_name: transformed}

        return transformed

    def _reverse_transform(self, data):
        if self._is_single_table:
            data = data[self._table_name]

        reverse_transformed = self.programmable_constraint.reverse_transform(data)

        if self._is_single_table:
            return {self._table_name: reverse_transformed}

        return reverse_transformed

    def _is_valid(self, data, metadata):
        if self._is_single_table:
            table_name = self._get_single_table_name(metadata)
            data = data[table_name]

        is_valid = self.programmable_constraint.is_valid(data)

        if self._is_single_table:
            return {table_name: is_valid}

        return is_valid
