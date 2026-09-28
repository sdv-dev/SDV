import re
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest
from copulas.univariate import BetaUnivariate, GammaUnivariate, UniformUnivariate
from rdt.transformers import GaussianNormalizer

from sdv.errors import SynthesizerInputError
from sdv.metadata.metadata import Metadata
from sdv.single_table.copulagan import CopulaGANSynthesizer


class TestCopulaGANSynthesizer:
    def test___init__(self):
        """Test creating an instance of ``CopulaGANSynthesizer``."""
        # Setup
        metadata = Metadata()
        enforce_min_max_values = True
        enforce_rounding = True

        # Run
        instance = CopulaGANSynthesizer(
            metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
        )

        # Assert
        assert instance.enforce_min_max_values is True
        assert instance.enforce_rounding is True
        assert instance.embedding_dim == 128
        assert instance.generator_dim == (256, 256)
        assert instance.discriminator_dim == (256, 256)
        assert instance.generator_lr == 2e-4
        assert instance.generator_decay == 1e-6
        assert instance.discriminator_lr == 2e-4
        assert instance.discriminator_decay == 1e-6
        assert instance.batch_size == 500
        assert instance.discriminator_steps == 1
        assert instance.log_frequency is True
        assert instance.verbose is False
        assert instance.epochs == 300
        assert instance.pac == 10
        assert instance.enable_gpu is True
        assert instance.numerical_distributions == {}
        assert instance.default_distribution == 'beta'
        assert instance._numerical_distributions == {}
        assert instance._default_distribution == BetaUnivariate

    def test___init__with_unified_metadata(self):
        """Test creating an instance of ``CopulaGANSynthesizer`` with Metadata."""
        # Setup
        metadata = Metadata()
        enforce_min_max_values = True
        enforce_rounding = True

        # Run
        instance = CopulaGANSynthesizer(
            metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
        )

        # Assert
        assert instance.enforce_min_max_values is True
        assert instance.enforce_rounding is True
        assert instance.embedding_dim == 128
        assert instance.generator_dim == (256, 256)
        assert instance.discriminator_dim == (256, 256)
        assert instance.generator_lr == 2e-4
        assert instance.generator_decay == 1e-6
        assert instance.discriminator_lr == 2e-4
        assert instance.discriminator_decay == 1e-6
        assert instance.batch_size == 500
        assert instance.discriminator_steps == 1
        assert instance.log_frequency is True
        assert instance.verbose is False
        assert instance.epochs == 300
        assert instance.pac == 10
        assert instance.enable_gpu is True
        assert instance.numerical_distributions == {}
        assert instance.default_distribution == 'beta'
        assert instance._numerical_distributions == {}
        assert instance._default_distribution == BetaUnivariate

    def test___init__custom(self):
        """Test creating an instance of ``CopulaGANSynthesizer`` with custom parameters."""
        # Setup
        metadata = Metadata()
        metadata.add_table('table')
        metadata.add_column('field', 'table', sdtype='numerical')
        enforce_min_max_values = False
        enforce_rounding = False
        embedding_dim = 64
        generator_dim = (128, 128)
        discriminator_dim = (128, 128)
        generator_lr = 1e-4
        generator_decay = 2e-6
        discriminator_lr = 3e-4
        discriminator_decay = 1e-6
        batch_size = 250
        discriminator_steps = 2
        log_frequency = False
        verbose = True
        epochs = 150
        pac = 5
        cuda = False
        numerical_distributions = {'field': 'gamma'}
        default_distribution = 'uniform'

        # Run
        instance = CopulaGANSynthesizer(
            metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
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
            enable_gpu=cuda,
            numerical_distributions=numerical_distributions,
            default_distribution=default_distribution,
        )

        # Assert
        assert instance.enforce_min_max_values is False
        assert instance.enforce_rounding is False
        assert instance.embedding_dim == embedding_dim
        assert instance.generator_dim == generator_dim
        assert instance.discriminator_dim == discriminator_dim
        assert instance.generator_lr == generator_lr
        assert instance.generator_decay == generator_decay
        assert instance.discriminator_lr == discriminator_lr
        assert instance.discriminator_decay == discriminator_decay
        assert instance.batch_size == batch_size
        assert instance.discriminator_steps == discriminator_steps
        assert instance.log_frequency == log_frequency
        assert instance.verbose is True
        assert instance.epochs == epochs
        assert instance.pac == pac
        assert instance.enable_gpu is False
        assert instance.numerical_distributions == {'field': 'gamma'}
        assert instance._numerical_distributions == {'field': GammaUnivariate}
        assert instance.default_distribution == 'uniform'
        assert instance._default_distribution == UniformUnivariate

    def test___init__incorrect_numerical_distributions(self):
        """Test it crashes when ``numerical_distributions`` receives a non-dictionary."""
        # Setup
        metadata = Metadata()
        numerical_distributions = 'invalid'

        # Run
        err_msg = 'numerical_distributions can only be None or a dict instance.'
        with pytest.raises(TypeError, match=err_msg):
            CopulaGANSynthesizer(metadata, numerical_distributions=numerical_distributions)

    def test___init__invalid_column_numerical_distributions(self):
        """Test it crashes when ``numerical_distributions`` includes invalid columns."""
        # Setup
        metadata = Metadata()
        numerical_distributions = {'totally_fake_column_name': 'beta'}

        # Run
        err_msg = re.escape(
            'Invalid column names found in the numerical_distributions dictionary '
            "{'totally_fake_column_name'}. The column names you provide must be present "
            'in the metadata.'
        )
        with pytest.raises(SynthesizerInputError, match=err_msg):
            CopulaGANSynthesizer(metadata, numerical_distributions=numerical_distributions)

    def test_get_params(self):
        """Test that inherited method ``get_params`` returns all the specific init parameters."""
        # Setup
        metadata = Metadata()
        instance = CopulaGANSynthesizer(metadata)

        # Run
        result = instance.get_parameters()

        # Assert
        assert result == {
            'enforce_min_max_values': True,
            'enforce_rounding': True,
            'locales': ['en_US'],
            'embedding_dim': 128,
            'generator_dim': (256, 256),
            'discriminator_dim': (256, 256),
            'generator_lr': 2e-4,
            'generator_decay': 1e-6,
            'discriminator_lr': 2e-4,
            'discriminator_decay': 1e-6,
            'batch_size': 500,
            'discriminator_steps': 1,
            'log_frequency': True,
            'verbose': False,
            'epochs': 300,
            'pac': 10,
            'enable_gpu': True,
            'numerical_distributions': {},
            'default_distribution': 'beta',
        }

    @patch('sdv.single_table.copulagan.rdt')
    def test__create_gaussian_normalizer_config(self, mock_rdt):
        """Test that a configuration for the numerical data has been created.

        A configuration for the ``rdt.HyperTransformer`` has to be created with
        ``GaussianNormalizer`` the only transformer for the ``numerical`` sdtypes.
        The rest of columns will be treated as ``categorical`` and the transformers set to
        ``None`` which is not to transform the data.
        """
        # Setup
        numerical_distributions = {'age': 'gamma'}
        metadata = Metadata()
        metadata.add_table('table')
        metadata.add_column('name', 'table', sdtype='categorical')
        metadata.add_column('age', 'table', sdtype='numerical')
        metadata.add_column('account', 'table', sdtype='numerical')

        instance = CopulaGANSynthesizer(metadata, numerical_distributions=numerical_distributions)
        processed_data = pd.DataFrame({
            'name': ['John', 'Doe', 'John Doe', 'John Doe Doe'],
            'age': np.arange(4),
            'account': np.arange(4),
            'name#age': np.arange(4),
        })

        # Run
        config = instance._create_gaussian_normalizer_config(processed_data)

        # Assert
        expected_calls = [
            call(missing_value_generation='from_column', distribution=GammaUnivariate),
            call(missing_value_generation='from_column', distribution=BetaUnivariate),
        ]
        expected_config = {
            'transformers': {
                'name': None,
                'age': mock_rdt.transformers.GaussianNormalizer.return_value,
                'account': mock_rdt.transformers.GaussianNormalizer.return_value,
                'name#age': None,
            },
            'sdtypes': {
                'name': 'categorical',
                'age': 'numerical',
                'account': 'numerical',
                'name#age': 'categorical',
            },
        }
        assert config == expected_config
        assert mock_rdt.transformers.GaussianNormalizer.call_args_list == expected_calls

    @patch('sdv.single_table.utils.warnings')
    @patch('sdv.single_table.copulagan.CTGANSynthesizer._fit')
    @patch('sdv.single_table.copulagan.rdt')
    def test__fit_logging(self, mock_rdt, mock_ctgansynthesizer__fit, mock_warnings):
        """Test a message is logged.

        A message should be logged if the columns passed in ``numerical_distributions``
        were renamed/dropped during preprocessing.
        """
        # Setup
        metadata = Metadata()
        metadata.add_table('table')
        metadata.add_column('col', 'table', sdtype='numerical')
        numerical_distributions = {'col': 'gamma'}
        instance = CopulaGANSynthesizer(metadata, numerical_distributions=numerical_distributions)
        processed_data = pd.DataFrame()

        # Run
        instance._fit(processed_data)

        # Assert
        warning_message = (
            "Cannot use distribution 'gamma' for column 'col' because the column is not "
            'statistically modeled.'
        )
        mock_warnings.warn.assert_called_once_with(warning_message, UserWarning)

    @patch('sdv.single_table.copulagan.CTGANSynthesizer._fit')
    @patch('sdv.single_table.copulagan.rdt')
    def test__fit(self, mock_rdt, mock_ctgansynthesizer__fit):
        """Test the ``_fit`` method for ``CopulaGANSynthesizer``.

        Test that when we call ``_fit`` a new instance of ``rdt.HyperTransformer`` is
        created, that only transforms numerical data using ``GaussianNormalizer`` transformer with
        one of the ``copulas`` distributions.
        """
        # Setup
        metadata = Metadata()
        instance = CopulaGANSynthesizer(metadata)
        instance._create_gaussian_normalizer_config = Mock()
        processed_data = pd.DataFrame()

        # Run
        instance._fit(processed_data)

        # Assert
        hypertransformer = instance._gaussian_normalizer_hyper_transformer
        assert hypertransformer == mock_rdt.HyperTransformer.return_value
        hypertransformer.set_config.assert_called_once_with(
            instance._create_gaussian_normalizer_config.return_value
        )

        hypertransformer.fit_transform.assert_called_once_with(processed_data)
        mock_ctgansynthesizer__fit.assert_called_once_with(
            hypertransformer.fit_transform.return_value
        )

    def test_get_learned_distributions(self):
        """Test that ``get_learned_distributions`` returns a dict.

        Test that it returns a dictionary with the name of the columns and the learned
        distribution and it's parameters.
        """
        # Setup
        data = pd.DataFrame({'zero': [0, 0, 0], 'one': [1, 1, 1]})
        stm = Metadata()
        stm.detect_from_dataframes({'table': data})
        cgs = CopulaGANSynthesizer(stm)
        zero_transformer_mock = Mock(spec_set=GaussianNormalizer)
        zero_transformer_mock._univariate.to_dict.return_value = {
            'a': 1.0,
            'b': 1.0,
            'loc': 0.0,
            'scale': 0.0,
            'type': None,
        }
        one_transformer_mock = Mock(spec_set=GaussianNormalizer)
        one_transformer_mock._univariate.to_dict.return_value = {
            'a': 1.0,
            'b': 1.0,
            'loc': 1.0,
            'scale': 0.0,
            'type': None,
        }
        cgs._gaussian_normalizer_hyper_transformer = Mock()
        cgs._gaussian_normalizer_hyper_transformer.field_transformers = {
            'zero': zero_transformer_mock,
            'one': one_transformer_mock,
        }
        cgs._fitted = True

        # Run
        result = cgs.get_learned_distributions()

        # Assert
        assert result == {
            'zero': {
                'distribution': 'beta',
                'learned_parameters': {'a': 1.0, 'b': 1.0, 'loc': 0.0, 'scale': 0.0},
            },
            'one': {
                'distribution': 'beta',
                'learned_parameters': {'a': 1.0, 'b': 1.0, 'loc': 1.0, 'scale': 0.0},
            },
        }

    def test_get_learned_distributions_nothing_learned(self):
        """Test that ``get_learned_distributions`` returns an empty dict when nothing is learned."""
        # Setup
        metadata = Metadata().load_from_dict({
            'tables': {
                'table1': {
                    'columns': {
                        'col_1': {'sdtype': 'id'},
                        'col_2': {'sdtype': 'credit_card_number'},
                    },
                    'primary_key': 'col_1',
                }
            }
        })
        data = pd.DataFrame({'col_1': range(100), 'col_2': range(100)})
        synthesizer = CopulaGANSynthesizer(metadata, default_distribution='beta')
        synthesizer.fit(data)

        # Run
        result = synthesizer.get_learned_distributions()

        # Assert
        assert result == {}

    def test_get_learned_distributions_raises_an_error(self):
        """Test that ``get_learned_distributions`` raises an error."""
        # Setup
        data = pd.DataFrame({'zero': [0, 0, 0], 'one': [1, 1, 1]})
        stm = Metadata()
        stm.detect_from_dataframes({'table': data})
        cgs = CopulaGANSynthesizer(stm)

        # Run and Assert
        error_msg = re.escape(
            "Distributions have not been learned yet. Please fit your model first using 'fit'."
        )
        with pytest.raises(ValueError, match=error_msg):
            cgs.get_learned_distributions()
