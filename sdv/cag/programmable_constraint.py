"""Programmable constraints base classes."""

import inspect
from copy import deepcopy

from sdv.cag.base import BaseConstraint


class ProgrammableConstraint:
    """Base class to create programmable constraints."""

    _is_single_table = True

    @classmethod
    def load_constraint_from_dict(cls, parameters):
        """Uses the given parameters to recreate an instance of the constraint."""
        return cls(**parameters)

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


class ProgrammableConstraintHarness(BaseConstraint):
    """Constraint interface class for programmable constraints."""

    def __init__(self, programmable_constraint):
        super().__init__()
        self.programmable_constraint = programmable_constraint
        self.table_name = getattr(self.programmable_constraint, 'table_name', None)
        self._is_single_table = self.programmable_constraint._is_single_table

    def get_constraint_dict(self):
        """Return the constraint as a serialiazable dict.

        Returns:
            dict:
                A dictionary with the following keys:
                    - `class_name` [str]: The name of the constraint class.
                    - `parameters` [dict]: A dictionary of the init parameters used to
                       create this constraint instance.
        """
        args = inspect.getfullargspec(self.programmable_constraint.__init__)
        keys = args.args[1:]
        instanced = {
            key: getattr(
                self.programmable_constraint,
                key,
                getattr(self.programmable_constraint, f'_{key}', None),
            )
            for key in keys
        }
        missing_attrs = list(set(keys) - set(instanced.keys()))
        if missing_attrs:
            missing_attrs = sorted(missing_attrs)
            raise AttributeError(
                'Cannot convert constraint to dictionary because required parameters '
                f'{missing_attrs} are not saved as attributes on the constraint.'
            )

        return {
            'class_name': self.programmable_constraint.__class__.__name__,
            'parameters': instanced,
        }

    def _validate_constraint_with_metadata(self, metadata):
        self.programmable_constraint.validate(metadata)

    def _validate_constraint_with_data(self, data, metadata):
        self.programmable_constraint.validate_input_data(data)

    def _get_updated_metadata(self, metadata):
        metadata = deepcopy(metadata)
        return self.programmable_constraint.get_updated_metadata(metadata)

    def _fit(self, data, metadata):
        self.programmable_constraint.fit(data, metadata)

    def _transform(self, data):
        return self.programmable_constraint.transform(data)

    def _reverse_transform(self, data):
        return self.programmable_constraint.reverse_transform(data)

    def _is_valid(self, data, metadata):
        return self.programmable_constraint.is_valid(data)
