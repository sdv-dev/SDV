"""Wrapper around CTGAN model."""


from sdv.tabular.base import BaseTabularModel


class CTGAN(BaseTabularModel):
    """Model wrapping ``CTGANSynthesizer`` model.

    Args:
        field_names (list[str]):
            List of names of the fields that need to be modeled
            and included in the generated output data. Any additional
            fields found in the data will be ignored and will not be
            included in the generated output.
            If ``None``, all the fields found in the data are used.
        field_types (dict[str, dict]):
            Dictinary specifying the data types and subtypes
            of the fields that will be modeled. Field types and subtypes
            combinations must be compatible with the SDV Metadata Schema.
        field_transformers (dict[str, str]):
            Dictinary specifying which transformers to use for each field.
            Available transformers are:

                * ``integer``: Uses a ``NumericalTransformer`` of dtype ``int``.
                * ``float``: Uses a ``NumericalTransformer`` of dtype ``float``.
                * ``categorical``: Uses a ``CategoricalTransformer`` without gaussian noise.
                * ``categorical_fuzzy``: Uses a ``CategoricalTransformer`` adding gaussian noise.
                * ``one_hot_encoding``: Uses a ``OneHotEncodingTransformer``.
                * ``label_encoding``: Uses a ``LabelEncodingTransformer``.
                * ``boolean``: Uses a ``BooleanTransformer``.
                * ``datetime``: Uses a ``DatetimeTransformer``.

        anonymize_fields (dict[str, str]):
            Dict specifying which fields to anonymize and what faker
            category they belong to.
        primary_key (str):
            Name of the field which is the primary key of the table.
        constraints (list[Constraint, dict]):
            List of Constraint objects or dicts.
        table_metadata (dict or metadata.Table):
            Table metadata instance or dict representation.
            If given alongside any other metadata-related arguments, an
            exception will be raised.
            If not given at all, it will be built using the other
            arguments or learned from the data.
        epochs (int):
            Number of training epochs. Defaults to 300.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Resiudal Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear
            Layer will be created for each one of the values provided. Defaults to (256, 256).
        batch_size (int):
            Number of data samples to process in each step.
        verbose (bool):
            Whether to print fit progress on stdout. Defaults to ``False``.
        cuda (bool or str):
            If ``True``, use CUDA. If an ``str``, use the indicated device.
            If ``False``, do not use cuda at all.
    """

    _model = None

    _DTYPE_TRANSFORMERS = {
        'O': 'label_encoding'
    }

    def __init__(self, field_names=None, field_types=None, field_transformers=None,
                 anonymize_fields=None, primary_key=None, constraints=None, table_metadata=None,
                 cuda=True, synthesizer="CTGAN", **kwargs):
        super().__init__(
            field_names=field_names,
            primary_key=primary_key,
            field_types=field_types,
            field_transformers=field_transformers,
            anonymize_fields=anonymize_fields,
            constraints=constraints,
            table_metadata=table_metadata
        )
        try:
            from ctgan import CTGANSynthesizer, TVAESynthesizer

            _default_parameters = {}
            for parameter, default_value in kwargs.items():
                _default_parameters[parameter] = default_value

            if synthesizer == "CTGAN":
                self._model = CTGANSynthesizer(**_default_parameters)
            elif synthesizer == "TVAE":
                self._model = TVAESynthesizer(**_default_parameters)

        except ImportError as ie:
            ie.msg += (
                '\n\nIt seems like `ctgan` is not installed.\n'
                'Please install it using:\n\n    pip install sdv[ctgan]'
            )
            raise

        self._cuda = cuda

    def _fit(self, table_data):
        """Fit the model to the table.

        Args:
            table_data (pandas.DataFrame):
                Data to be learned.
        """
        import torch
        if not self._cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(self._cuda, str):
            device = self._cuda
        else:
            device = 'cuda'

        self._model.device = torch.device(device)

        categoricals = [
            field
            for field, meta in self._metadata.get_fields().items()
            if meta['type'] == 'categorical'
        ]

        self._model.fit(
            table_data,
            discrete_columns=categoricals
        )

    def _sample(self, num_rows):
        """Sample the indicated number of rows from the model.

        Args:
            num_rows (int):
                Amount of rows to sample.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        return self._model.sample(num_rows)
