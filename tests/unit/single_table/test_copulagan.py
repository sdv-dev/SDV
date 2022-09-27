from unittest.mock import patch

from copulas.univariate import BetaUnivariate, GammaUnivariate, UniformUnivariate

from sdv.metadata.single_table import SingleTableMetadata
from sdv.single_table.copulagan import CopulaGANSynthesizer


class TestCopulaGANSynthesizer:

    @patch('sdv.single_table.copulagan.rdt')
    def test___init__(self, mock_rdt):
        """Test creating an instance of ``CopulaGANSynthesizer``."""
        # Setup
        metadata = SingleTableMetadata()
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
        assert instance.cuda is True
        assert instance.numerical_distributions == {}
        assert instance.default_distribution == 'beta'
        assert instance._numerical_distributions == {}
        assert instance._default_distribution == BetaUnivariate
        assert instance._hyper_transformer == mock_rdt.HyperTransformer.return_value
        mock_rdt.HyperTransformer.assert_called_once()

    @patch('sdv.single_table.copulagan.rdt')
    def test___init__custom(self, mock_rdt):
        """Test creating an instance of ``CopulaGANSynthesizer`` with custom parameters."""
        # Setup
        metadata = SingleTableMetadata()
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
            cuda=cuda,
            numerical_distributions=numerical_distributions,
            default_distribution=default_distribution
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
        assert instance.cuda is False
        assert instance.numerical_distributions == {'field': 'gamma'}
        assert instance._numerical_distributions == {'field': GammaUnivariate}
        assert instance.default_distribution == 'uniform'
        assert instance._default_distribution == UniformUnivariate
        assert instance._hyper_transformer == mock_rdt.HyperTransformer.return_value
        mock_rdt.HyperTransformer.assert_called_once()

    def test_get_params(self):
        """Test that the inherit method ``get_params`` returns all the specific init parameters."""
        # Setup
        metadata = SingleTableMetadata()
        instance = CopulaGANSynthesizer(metadata)

        # Run
        result = instance.get_parameters()

        # Assert
        assert result == {
            'enforce_min_max_values': True,
            'enforce_rounding': True,
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
            'cuda': True,
            'numerical_distributions': {},
            'default_distribution': 'beta',
        }
