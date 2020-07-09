import pickle

from sdv.metadata import Table


class BaseTabularModel():
    """Base class for all the tabular models.

    The ``BaseTabularModel`` class defines the common API that all the
    TabularModels need to implement, as well as common functionality.
    """

    TRANSFORMER_TEMPLATES = None

    _metadata = None

    def __init__(self, field_names=None, primary_key=None, field_types=None,
                 anonymize_fields=None, table_metadata=None, *args, **kwargs):
        """Initialize a Tabular Model.

        Args:
            field_names (list[str]):
                List of names of the fields that need to be modeled
                and included in the generated output data. Any additional
                fields found in the data will be ignored and will not be
                included in the generated output, except if they have
                been added as primary keys or fields to anonymize.
                If ``None``, all the fields found in the data are used.
            primary_key (str, list[str] or dict[str, dict]):
                Specification about which field or fields are the
                primary key of the table and information about how
                to generate them.
            field_types (dict[str, dict]):
                Dictinary specifying the data types and subtypes
                of the fields that will be modeled. Field types and subtypes
                combinations must be compatible with the SDV Metadata Schema.
            anonymize_fields (dict[str, str]):
                Dict specifying which fields to anonymize and what faker
                category they belong to.
            table_metadata (dict or metadata.Table):
                Table metadata instance or dict representation.
                If given alongside any other metadata-related arguments, an
                exception will be raised.
                If not given at all, it will be built using the other
                arguments or learned from the data.
            *args, **kwargs:
                Subclasses will add any arguments or keyword arguments needed.
        """
        if table_metadata is not None:
            if isinstance(table_metadata, dict):
                table_metadata = Table(table_metadata,)

            for arg in (field_names, primary_key, field_types, anonymize_fields):
                if arg:
                    raise ValueError(
                        'If table_metadata is given {} must be None'.format(arg.__name__))

            self._metadata = table_metadata

        else:
            self._field_names = field_names
            self._primary_key = primary_key
            self._field_types = field_types
            self._anonymize_fields = anonymize_fields

    def _fit_metadata(self, data):
        """Generate a new Table metadata and fit it to the data.

        The information provided will be used to create the Table instance
        and then the rest of information will be learned from the given
        data.

        Args:
            data (pandas.DataFrame):
                Data to learn from.
        """
        metadata = Table(
            field_names=self._field_names,
            primary_key=self._primary_key,
            field_types=self._field_types,
            anonymize_fields=self._anonymize_fields,
            transformer_templates=self.TRANSFORMER_TEMPLATES,
        )
        metadata.fit(data)

        self._metadata = metadata

    def fit(self, data):
        """Fit this model to the data.

        If the table metadata has not been given, learn it from the data.

        Args:
            data (pandas.DataFrame or str):
                Data to fit the model to. It can be passed as a
                ``pandas.DataFrame`` or as an ``str``.
                If an ``str`` is passed, it is assumed to be
                the path to a CSV file which can be loaded using
                ``pandas.read_csv``.
        """
        if self._metadata is None:
            self._fit_metadata(data)

        self._size = len(data)

        transformed = self._metadata.transform(data)
        self._fit(transformed)

    def get_metadata(self):
        """Get metadata about the table.

        This will return an ``sdv.metadata.Table`` object containing
        the information about the data that this model has learned.

        This Table metadata will contain some common information,
        such as field names and data types, as well as additional
        information that each Sub-class might add, such as the
        observed data field distributions and their parameters.

        Returns:
            sdv.metadata.Table:
                Table metadata.
        """
        return self._metadata

    def sample(self, size=None, values=None):
        """Sample rows from this table.

        Args:
            size (int):
                Number of rows to sample. If not given the model
                will generate as many rows as there were in the
                data passed to the ``fit`` method.
            values (dict):    <- FUTURE
                Fixed values to use for knowledge-based sampling.
                In case the model does not support knowledge-based
                sampling, a discard+resample strategy will be used.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        size = size or self._size
        sampled = self._sample(size)
        return self._metadata.reverse_transform(sampled)

    def get_parameters(self):
        """Get the parameters learned from the data.

        The result is a flat dict (single level) which contains
        all the necessary parameters to be able to reproduce
        this model.

        Subclasses which are not parametric, such as DeepLearning
        based models, raise a NonParametricError indicating that
        this method is not supported for their implementation.

        Returns:
            parameters (dict):
                flat dict (single level) which contains all the
                necessary parameters to be able to reproduce
                this model.

        Raises:
            NonParametricError:
                If the model is not parametric or cannot be described
                using a simple dictionary.
        """
        raise NotImplementedError()

    @classmethod
    def from_parameters(cls):
        """Regenerate a previously learned model from its parameters.

        Subclasses which are not parametric, such as DeepLearning
        based models, raise a NonParametricError indicating that
        this method is not supported for their implementation.

        Returns:
            BaseTabularModel:
                New instance with its parameters set.

        Raises:
            NonParametricError:
                If the model is not parametric or cannot be described
                using a simple dictionary.
        """
        raise NotImplementedError()

    def save(self, path):
        """Save this model instance to the given path using pickle.

        Args:
            path (str):
                Path where the SDV instance will be serialized.
        """
        with open(path, 'wb') as output:
            pickle.dump(self, output)

    @classmethod
    def load(cls, path):
        """Load a TabularModel instance from a given path.

        Args:
            path (str):
                Path from which to load the instance.

        Returns:
            TabularModel:
                The loaded tabular model.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
