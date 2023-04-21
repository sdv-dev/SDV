from unittest.mock import Mock, patch

from sdv.metadata.single_table import SingleTableMetadata
from sdv.single_table.ctgan import CTGANSynthesizer, TVAESynthesizer


class TestCTGANSynthesizer:

    def test___init__(self):
        """Test creating an instance of ``CTGANSynthesizer``."""
        # Setup
        metadata = SingleTableMetadata()
        enforce_min_max_values = True
        enforce_rounding = True

        # Run
        instance = CTGANSynthesizer(
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

    def test___init__custom(self):
        """Test creating an instance of ``CTGANSynthesizer`` with custom parameters."""
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

        # Run
        instance = CTGANSynthesizer(
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

    def test_get_parameters(self):
        """Test that inherited method ``get_parameters`` returns the specific init parameters."""
        # Setup
        metadata = SingleTableMetadata()
        instance = CTGANSynthesizer(metadata)

        # Run
        result = instance.get_parameters()

        # Assert
        assert result == {
            'enforce_min_max_values': True,
            'enforce_rounding': True,
            'locales': None,
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
        }

    @patch('sdv.single_table.ctgan.CTGAN')
    @patch('sdv.single_table.ctgan.detect_discrete_columns')
    def test__fit(self, mock_detect_discrete_columns, mock_ctgan):
        """Test the ``_fit`` from ``CTGANSynthesizer``.

        Test that when we call ``_fit`` a new instance of ``CTGAN`` is created as a model
        with the previously set parameters. Then this is being fitted with the ``discrete_columns``
        that have been detected by the utility function.
        """
        # Setup
        metadata = SingleTableMetadata()
        instance = CTGANSynthesizer(metadata)
        processed_data = Mock()

        # Run
        instance._fit(processed_data)

        # Assert
        mock_detect_discrete_columns.assert_called_once_with(metadata, processed_data)
        mock_ctgan.assert_called_once_with(
            batch_size=500,
            cuda=True,
            discriminator_decay=1e-6,
            discriminator_dim=(256, 256),
            discriminator_lr=2e-4,
            discriminator_steps=1,
            embedding_dim=128,
            epochs=300,
            generator_decay=1e-6,
            generator_dim=(256, 256),
            generator_lr=2e-4,
            log_frequency=True,
            pac=10,
            verbose=False,
        )

        assert instance._model == mock_ctgan.return_value
        instance._model.fit.assert_called_once_with(
            processed_data,
            discrete_columns=mock_detect_discrete_columns.return_value
        )


class TestTVAESynthesizer:

    def test___init__(self):
        """Test creating an instance of ``TVAESynthesizer``."""
        # Setup
        metadata = SingleTableMetadata()
        enforce_min_max_values = True
        enforce_rounding = True

        # Run
        instance = TVAESynthesizer(
            metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
        )

        # Assert
        assert instance.enforce_min_max_values is True
        assert instance.enforce_rounding is True
        assert instance.embedding_dim == 128
        assert instance.compress_dims == (128, 128)
        assert instance.decompress_dims == (128, 128)
        assert instance.l2scale == 1e-5
        assert instance.batch_size == 500
        assert instance.epochs == 300
        assert instance.loss_factor == 2
        assert instance.cuda is True

    def test___init__custom(self):
        """Test creating an instance of ``TVAESynthesizer`` with custom parameters."""
        # Setup
        metadata = SingleTableMetadata()
        enforce_min_max_values = False
        enforce_rounding = False
        embedding_dim = 64
        compress_dims = (64, 64)
        decompress_dims = (64, 64)
        l2scale = 2e-5
        batch_size = 250
        epochs = 150
        loss_factor = 4
        cuda = False

        # Run
        instance = TVAESynthesizer(
            metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
            embedding_dim=embedding_dim,
            compress_dims=compress_dims,
            decompress_dims=decompress_dims,
            l2scale=l2scale,
            batch_size=batch_size,
            epochs=epochs,
            loss_factor=loss_factor,
            cuda=cuda,
        )

        # Assert
        assert instance.enforce_min_max_values is False
        assert instance.enforce_rounding is False
        assert instance.embedding_dim == 64
        assert instance.compress_dims == (64, 64)
        assert instance.decompress_dims == (64, 64)
        assert instance.l2scale == 2e-5
        assert instance.batch_size == 250
        assert instance.epochs == 150
        assert instance.loss_factor == 4
        assert instance.cuda is False

    def test_get_parameters(self):
        """Test that inherited method ``get_parameters`` returns the specific init parameters."""
        # Setup
        metadata = SingleTableMetadata()
        instance = TVAESynthesizer(metadata)

        # Run
        result = instance.get_parameters()

        # Assert
        assert result == {
            'enforce_min_max_values': True,
            'enforce_rounding': True,
            'embedding_dim': 128,
            'compress_dims': (128, 128),
            'decompress_dims': (128, 128),
            'l2scale': 1e-5,
            'batch_size': 500,
            'epochs': 300,
            'loss_factor': 2,
            'cuda': True,
        }

    @patch('sdv.single_table.ctgan.TVAE')
    @patch('sdv.single_table.ctgan.detect_discrete_columns')
    def test__fit(self, mock_detect_discrete_columns, mock_tvae):
        """Test the ``_fit`` from ``TVAESynthesizer``.

        Test that when we call ``_fit`` a new instance of ``TVAE`` is created as a model
        with the previously set parameters. Then this is being fitted with the ``discrete_columns``
        that have been detected by the utility function.
        """
        # Setup
        metadata = SingleTableMetadata()
        instance = TVAESynthesizer(metadata)
        processed_data = Mock()

        # Run
        instance._fit(processed_data)

        # Assert
        mock_detect_discrete_columns.assert_called_once_with(metadata, processed_data)
        mock_tvae.assert_called_once_with(
            batch_size=500,
            compress_dims=(128, 128),
            cuda=True,
            decompress_dims=(128, 128),
            embedding_dim=128,
            epochs=300,
            l2scale=1e-5,
            loss_factor=2,
        )

        assert instance._model == mock_tvae.return_value
        instance._model.fit.assert_called_once_with(
            processed_data,
            discrete_columns=mock_detect_discrete_columns.return_value
        )
