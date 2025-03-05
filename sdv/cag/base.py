"""Base Multi-Table Pattern."""

import pandas as pd

from sdv.errors import NotFittedError


class BasePattern:
    """Base CAG Pattern Class."""

    def __init__(self):
        self.metadata = None
        self._fitted = False
        self._single_table = False

    def _get_single_table_name(self, metadata):
        if not hasattr(self, 'table_name'):
            raise ValueError('No ``table_name`` attribute has been set.')

        return metadata._get_single_table_name() if self.table_name is None else self.table_name

    def _validate_pattern_with_metadata(self, metadata):
        raise NotImplementedError()

    def _validate_pattern_with_data(self, data, metadata):
        raise NotImplementedError()

    def validate(self, data=None, metadata=None):
        """Validate the data/metadata meets the pattern requirements.

        Args:
            data (dict[str, pd.DataFrame], optional)
                The data dictionary. If `None`, ``validate`` will skip data validation.
            metadata (sdv.Metadata, optional)
                The input metadata. If `None`, pattern must have been fitted and ``validate``
                will use the metadata saved during fitting.
        """
        if metadata is None:
            if self.metadata is None:
                raise NotFittedError('Pattern must be fit before validating without metadata.')

            metadata = self.metadata

        self._validate_pattern_with_metadata(metadata)
        if data is not None:
            if self._single_table:
                data = {self._table_name: data}
            elif isinstance(data, pd.DataFrame):
                table_name = self._get_single_table_name(metadata)
                data = {table_name: data}

            elif isinstance(data, pd.DataFrame):
                table_name = self._get_single_table_name(metadata)
                data = {table_name: data}

            self._validate_pattern_with_data(data, metadata)

    def _get_updated_metadata(self, metadata):
        return metadata

    def get_updated_metadata(self, metadata):
        """Get the updated metadata after applying the pattern to the input metadata.

        Args:
            metadata (sdv.Metadata):
                The input metadata to apply the pattern to.
        """
        self.validate(metadata=metadata)
        return self._get_updated_metadata(metadata)

    def _fit(self, data, metadata):
        raise NotImplementedError

    def fit(self, data, metadata):
        """Fit the pattern with data and metadata.

        Args:
            data (dict[pd.DataFrame]):
                The data dictionary to fit the pattern on.
            metadata (sdv.Metadata):
                The metadata to fit the pattern on.
        """
        self._validate_pattern_with_metadata(metadata)
        if isinstance(data, pd.DataFrame):
            self._single_table = True
            self._table_name = self._get_single_table_name(metadata)
            data = {self._table_name: data}

        self._validate_pattern_with_data(data, metadata)
        self._fit(data, metadata)
        self.metadata = metadata

        self._dtypes = {table: data[table].dtypes.to_dict() for table in metadata.tables}
        self._original_data_columns = {
            table: metadata.tables[table].get_column_names() for table in metadata.tables
        }
        self._fitted = True

    def _transform(self, data):
        raise NotImplementedError

    def transform(self, data):
        """Transform the data.

        Args:
            data (dict[str, pd.DataFrame])
                The input data dictionary to be transformed.
        """
        if not self._fitted:
            raise NotFittedError('Pattern must be fit using ``fit`` before transforming.')

        self.validate(data)
        if self._single_table:
            data = {self._table_name: data}

        data = {table: table_data.copy() for table, table_data in data.items()}
        transformed_data = self._transform(data)
        if self._single_table:
            return transformed_data[self._table_name]

        return transformed_data

    def _reverse_transform(self, data):
        raise NotImplementedError

    def reverse_transform(self, data):
        """Reverse transform the data back into the original space.

        Args:
            data (dict[str, pd.DataFrame])
                The transformed data dictionary to be reverse transformed.
        """
        if self._single_table:
            data = {self._table_name: data}

        data = {table: table_data.copy() for table, table_data in data.items()}
        reverse_transformed = self._reverse_transform(data)
        for table_name, table in reverse_transformed.items():
            table = table[self._original_data_columns[table_name]]
            reverse_transformed[table_name] = table.astype(self._dtypes[table_name])

        if self._single_table:
            return reverse_transformed[self._table_name]

        return reverse_transformed

    def _is_valid(self, data):
        raise NotImplementedError

    def is_valid(self, data):
        """Say whether the given table rows are valid.

        Args:
            data (pd.DataFrame or dict[pd.DataFrame]):
                Table data.

        Returns:
            pd.Series or dict[pd.Series]:
                Series of boolean values indicating if the row is valid for the pattern or not.
        """
        if not self._fitted:
            raise NotFittedError(
                'Pattern must be fit using ``fit`` before determining if data is valid.'
            )

        if self._single_table:
            data = {self._table_name: data}

        is_valid_data = self._is_valid(data)
        if self._single_table:
            return is_valid_data[self._table_name]

        return is_valid_data
