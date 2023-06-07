"""Combination of GaussianCopula transformation and GANs."""
import logging
from copy import deepcopy

import rdt

from sdv.single_table.copulas import GaussianCopulaSynthesizer
from sdv.single_table.ctgan import CTGANSynthesizer
from sdv.single_table.utils import (
    log_numerical_distributions_error, validate_numerical_distributions)

LOGGER = logging.getLogger(__name__)


class CopulaGANSynthesizer(CTGANSynthesizer):
    """Combination of GaussianCopula transformation and GANs.

    This model extends the ``CTGAN`` model to add the flexibility of the GaussianCopula
    transformations provided by the ``GaussianNormalizer`` from ``RDT``.

    Overall, the fitting process consists of the following steps:

    1. Transform each non categorical variable from the input
       data using a ``GaussianNormalizer``:

       i. If not specified, find out the distribution which each one
          of the variables from the input dataset has.
       ii. Transform each variable to a standard normal space by applying
           the CDF of the corresponding distribution and later on applying
           an inverse CDF from a standard normal distribution.

    2. Fit CTGAN with the transformed table.

    And the process of sampling is:

    1. Sample using CTGAN
    2. Reverse the previous transformation by applying the CDF of a standard normal
       distribution and then inverting the CDF of the distribution that correpsonds
       to each variable.

    The arguments of this model are the same as for CTGAN except for two additional
    arguments, ``numerical_distributions`` and ``default_distribution`` that give the
    ability to define specific transformations for individual fields as well as
    which distribution to use by default if no specific distribution has been selected.

    Distributions can be passed as a ``copulas`` univariate instance or as one
    of the following string values:

    * ``norm``: Use a norm distribution.
    * ``beta``: Use a Beta distribution.
    * ``truncnorm``: Use a truncnorm distribution.
    * ``uniform``: Use a uniform distribution.
    * ``gamma``: Use a Gamma distribution.
    * ``gaussian_kde``: Use a GaussianKDE distribution. This model is non-parametric,
      so using this will make ``get_parameters`` unusable.


    Args:
        metadata (sdv.metadata.SingleTableMetadata):
            Single table metadata representing the data that this synthesizer will be used for.
        enforce_min_max_values (bool):
            Specify whether or not to clip the data returned by ``reverse_transform`` of
            the numerical transformer, ``FloatFormatter``, to the min and max values seen
            during ``fit``. Defaults to ``True``.
        enforce_rounding (bool):
            Define rounding scheme for ``numerical`` columns. If ``True``, the data returned
            by ``reverse_transform`` will be rounded as in the original data. Defaults to ``True``.
        locales (list or str):
            The default locale(s) to use for AnonymizedFaker transformers. Defaults to ``None``.
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
        epochs (int):
            Number of training epochs. Defaults to 300.
        cuda (bool or str):
            If ``True``, use CUDA. If an ``str``, use the indicated device.
            If ``False``, do not use cuda at all.
        numerical_distributions (dict):
            Dictionary that maps field names from the table that is being modeled with
            the distribution that needs to be used. The distributions can be passed as either
            a ``copulas.univariate`` instance or as one of the following values:

                * ``norm``: Use a norm distribution.
                * ``beta``: Use a Beta distribution.
                * ``truncnorm``: Use a truncnorm distribution.
                * ``uniform``: Use a uniform distribution.
                * ``gamma``: Use a Gamma distribution.
                * ``gaussian_kde``: Use a GaussianKDE distribution. This model is non-parametric,
                  so using this will make ``get_parameters`` unusable.

        default_distribution (str):
            Copulas univariate distribution to use by default. Valid options are:

                * ``norm``: Use a norm distribution.
                * ``beta``: Use a Beta distribution.
                * ``truncnorm``: Use a truncnorm distribution.
                * ``uniform``: Use a uniform distribution.
                * ``gamma``: Use a Gamma distribution.
                * ``gaussian_kde``: Use a GaussianKDE distribution. This model is non-parametric,
                  so using this will make ``get_parameters`` unusable.
             Defaults to ``beta``.
    """

    _gaussian_normalizer_hyper_transformer = None

    def __init__(self, metadata, enforce_min_max_values=True, enforce_rounding=True, locales=None,
                 embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=False, epochs=300, pac=10, cuda=True,
                 numerical_distributions=None, default_distribution=None):

        super().__init__(
            metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
            locales=locales,
            embedding_dim=embedding_dim,
            generator_dim=generator_dim,
            discriminator_dim=discriminator_dim,
            generator_lr=generator_lr,
            generator_decay=generator_decay,
            discriminator_lr=discriminator_lr,
            discriminator_decay=discriminator_decay,
            batch_size=batch_size,
            discriminator_steps=discriminator_steps,
            log_frequency=log_frequency,
            verbose=verbose,
            epochs=epochs,
            pac=pac,
            cuda=cuda,
        )

        validate_numerical_distributions(numerical_distributions, self.metadata.columns)
        self.numerical_distributions = numerical_distributions or {}
        self.default_distribution = default_distribution or 'beta'

        self._default_distribution = GaussianCopulaSynthesizer.get_distribution_class(
            default_distribution or 'beta'
        )
        self._numerical_distributions = {
            field: GaussianCopulaSynthesizer.get_distribution_class(distribution)
            for field, distribution in self.numerical_distributions.items()
        }

    def _create_gaussian_normalizer_config(self, processed_data):
        columns = self.metadata.columns
        transformers = {}
        sdtypes = {}
        for column in processed_data.columns:
            sdtype = columns.get(column, {}).get('sdtype')
            if column in columns and sdtype not in ['categorical', 'boolean']:
                sdtypes[column] = 'numerical'
                distribution = self._numerical_distributions.get(
                    column,
                    self._default_distribution
                )

                transformers[column] = rdt.transformers.GaussianNormalizer(
                    missing_value_generation='from_column',
                    distribution=distribution,
                )

            else:
                sdtypes[column] = sdtype or 'categorical'
                transformers[column] = None

        return {'transformers': transformers, 'sdtypes': sdtypes}

    def _fit(self, processed_data):
        """Fit the model to the table.

        Args:
            processed_data (pandas.DataFrame):
                Data to be learned.
        """
        log_numerical_distributions_error(
            self.numerical_distributions, processed_data.columns, LOGGER)

        gaussian_normalizer_config = self._create_gaussian_normalizer_config(processed_data)
        self._gaussian_normalizer_hyper_transformer = rdt.HyperTransformer()
        self._gaussian_normalizer_hyper_transformer.set_config(gaussian_normalizer_config)
        processed_data = self._gaussian_normalizer_hyper_transformer.fit_transform(processed_data)

        super()._fit(processed_data)

    def _sample(self, num_rows, conditions=None):
        """Sample the indicated number of rows from the model.

        Args:
            num_rows (int):
                Amount of rows to sample.
            conditions (dict):
                If specified, this dictionary maps column names to the column
                value. Then, this method generates ``num_rows`` samples, all of
                which are conditioned on the given variables.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        sampled = super()._sample(num_rows, conditions)
        return self._gaussian_normalizer_hyper_transformer.reverse_transform(sampled)

    def get_learned_distributions(self):
        """Get the marginal distributions used by the ``CTGANSynthesizer``.

        Return a dictionary mapping the column names with the distribution name and the learned
        parameters for those.

        Returns:
            dict:
                Dictionary containing the distributions used or detected for each column and the
                learned parameters for those.
        """
        if not self._fitted:
            raise ValueError(
                "Distributions have not been learned yet. Please fit your model first using 'fit'."
            )

        field_transformers = self._gaussian_normalizer_hyper_transformer.field_transformers

        learned_distributions = {}
        for column_name, transformer in field_transformers.items():
            if isinstance(transformer, rdt.transformers.GaussianNormalizer):
                learned_params = deepcopy(transformer._univariate.to_dict())
                learned_params.pop('type')
                distribution = self.numerical_distributions.get(
                    column_name,
                    self.default_distribution
                )
                learned_distributions[column_name] = {
                    'distribution': distribution,
                    'learned_parameters': learned_params
                }

        return learned_distributions
