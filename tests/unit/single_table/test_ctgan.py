import re
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from sdmetrics import visualization

from sdv.errors import InvalidDataTypeError, NotFittedError
from sdv.metadata.single_table import SingleTableMetadata
from sdv.single_table.ctgan import CTGANSynthesizer, TVAESynthesizer, _validate_no_category_dtype


def test__validate_no_category_dtype():
    """Test that 'category' dtype causes error."""
    # Setup
    data = pd.DataFrame({
        'category1': pd.Categorical(['a', 'a', 'b']),
        'value': [0, 1, 2],
        'category2': pd.Categorical([0, 1, 2]),
    })

    # Run and Assert
    expected = re.escape(
        "Columns ['category1', 'category2'] are stored as a 'category' type, which is not "
        "supported. Please cast these columns to an 'object' to continue."
    )
    with pytest.raises(InvalidDataTypeError, match=expected):
        _validate_no_category_dtype(data)


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

    def test___init__with_unified_metadata(self):
        """Test creating an instance of ``CTGANSynthesizer`` with Metadata."""
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
            'cuda': True,
        }

    def test__estimate_num_columns(self):
        """Test that ``_estimate_num_columns`` returns without crashing the number of columns."""
        # Setup
        metadata = SingleTableMetadata()
        metadata.add_column('id', sdtype='numerical')
        metadata.add_column('name', sdtype='categorical')
        metadata.add_column('surname', sdtype='categorical')
        data = pd.DataFrame({
            'id': np.random.rand(1_001),
            'name': [f'cat_{i}' for i in range(1_001)],
            'surname': [f'cat_{i}' for i in range(1_001)],
        })

        instance = CTGANSynthesizer(metadata)
        instance._data_processor.fit(data)

        # Remove the 'surname' transformer to replicate issue #1717
        instance._data_processor._hyper_transformer.field_transformers.pop('surname')

        # Run
        result = instance._estimate_num_columns(data)

        # Assert
        assert result == {'id': 11, 'name': 1001, 'surname': 1001}

    def test_preprocessing_many_categories(self, capfd):
        """Test a message is printed during preprocess when a column has many categories."""
        # Setup
        metadata = SingleTableMetadata()
        metadata.add_column('name_longer_than_Original_Column_Name', sdtype='numerical')
        metadata.add_column('categorical', sdtype='categorical')
        data = pd.DataFrame({
            'name_longer_than_Original_Column_Name': np.random.rand(1_001),
            'categorical': [f'cat_{i}' for i in range(1_001)],
        })
        instance = CTGANSynthesizer(metadata)

        # Run
        instance.preprocess(data)

        # Assert
        out, _err = capfd.readouterr()
        assert out == (
            'PerformanceAlert: Using the CTGANSynthesizer on this data is not recommended. '
            'To model this data, CTGAN will generate a large number of columns.'
            '\n\n'
            'Original Column Name                  Est # of Columns (CTGAN)\n'
            'name_longer_than_Original_Column_Name 11\n'
            'categorical                           1001'
            '\n\n'
            'We recommend preprocessing discrete columns that can have many values, '
            "using 'update_transformers'. Or you may drop columns that are not necessary "
            'to model. (Exit this script using ctrl-C)\n'
        )

    def test_preprocessing_few_categories(self, capfd):
        """Test a message is not printed during preprocess when a column has few categories."""
        # Setup
        metadata = SingleTableMetadata()
        metadata.add_column('name_longer_than_Original_Column_Name', sdtype='numerical')
        metadata.add_column('categorical', sdtype='categorical')
        data = pd.DataFrame({
            'name_longer_than_Original_Column_Name': np.random.rand(10),
            'categorical': [f'cat_{i}' for i in range(10)],
        })
        instance = CTGANSynthesizer(metadata)

        # Run
        instance.preprocess(data)

        # Assert
        out, _err = capfd.readouterr()
        assert out == ''

    @patch('sdv.single_table.ctgan.CTGAN')
    @patch('sdv.single_table.ctgan.detect_discrete_columns')
    @patch('sdv.single_table.ctgan._validate_no_category_dtype')
    def test__fit(self, mock_category_validate, mock_detect_discrete_columns, mock_ctgan):
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
        mock_category_validate.assert_called_once_with(processed_data)
        mock_detect_discrete_columns.assert_called_once_with(
            metadata, processed_data, instance._data_processor._hyper_transformer.field_transformers
        )
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
            processed_data, discrete_columns=mock_detect_discrete_columns.return_value
        )

    def test_get_loss_values(self):
        """Test the ``get_loss_values`` method from ``CTGANSynthesizer."""
        # Setup
        mock_model = Mock()
        loss_values = pd.DataFrame({'Epoch': [0, 1, 2], 'Loss': [0.8, 0.6, 0.5]})
        mock_model.loss_values = loss_values
        metadata = SingleTableMetadata()
        instance = CTGANSynthesizer(metadata)
        instance._model = mock_model
        instance._fitted = True

        # Run
        actual_loss_values = instance.get_loss_values()

        # Assert
        pd.testing.assert_frame_equal(actual_loss_values, loss_values)

    def test_get_loss_values_error(self):
        """Test the ``get_loss_values`` errors if synthesizer has not been fitted."""
        # Setup
        metadata = SingleTableMetadata()
        instance = CTGANSynthesizer(metadata)

        # Run / Assert
        msg = 'Loss values are not available yet. Please fit your synthesizer first.'
        with pytest.raises(NotFittedError, match=msg):
            instance.get_loss_values()

    @patch('sdv.single_table.ctgan.px.line')
    def test_get_loss_values_plot(self, mock_line_plot):
        """Test the ``get_loss_values_plot`` method from ``CTGANSynthesizer."""
        # Setup
        metadata = SingleTableMetadata()
        instance = CTGANSynthesizer(metadata)
        mock_loss_value = Mock()
        mock_loss_value.item.return_value = 0.1
        mock_model = Mock()
        loss_values = pd.DataFrame({
            'Epoch': [0, 1, 2],
            'Generator Loss': [mock_loss_value, mock_loss_value, mock_loss_value],
            'Discriminator Loss': [mock_loss_value, mock_loss_value, mock_loss_value],
        })
        mock_model.loss_values = loss_values
        instance._model = mock_model
        instance._fitted = True

        # Run
        instance.get_loss_values_plot()
        fig = mock_line_plot.call_args[1]
        assert fig['x'] == 'Epoch'
        assert fig['y'] == ['Generator Loss', 'Discriminator Loss']
        assert fig['color_discrete_map'] == {
            'Generator Loss': visualization.PlotConfig.DATACEBO_DARK,
            'Discriminator Loss': visualization.PlotConfig.DATACEBO_GREEN,
        }


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
        assert instance.verbose is False
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
        verbose = True
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
            verbose=verbose,
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
        assert instance.verbose is True
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
            'verbose': False,
            'epochs': 300,
            'loss_factor': 2,
            'cuda': True,
        }

    @patch('sdv.single_table.ctgan.TVAE')
    @patch('sdv.single_table.ctgan.detect_discrete_columns')
    @patch('sdv.single_table.ctgan._validate_no_category_dtype')
    def test__fit(self, mock_category_validate, mock_detect_discrete_columns, mock_tvae):
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
        mock_category_validate.assert_called_once_with(processed_data)
        mock_detect_discrete_columns.assert_called_once_with(
            metadata, processed_data, instance._data_processor._hyper_transformer.field_transformers
        )
        mock_tvae.assert_called_once_with(
            batch_size=500,
            compress_dims=(128, 128),
            cuda=True,
            decompress_dims=(128, 128),
            embedding_dim=128,
            verbose=False,
            epochs=300,
            l2scale=1e-5,
            loss_factor=2,
        )

        assert instance._model == mock_tvae.return_value
        instance._model.fit.assert_called_once_with(
            processed_data, discrete_columns=mock_detect_discrete_columns.return_value
        )

    def test_get_loss_values(self):
        """Test the ``get_loss_values`` method from ``TVAESynthesizer."""
        # Setup
        mock_model = Mock()
        loss_values = pd.DataFrame({'Epoch': [0, 1, 2], 'Loss': [0.8, 0.6, 0.5]})
        mock_model.loss_values = loss_values
        metadata = SingleTableMetadata()
        instance = TVAESynthesizer(metadata)
        instance._model = mock_model
        instance._fitted = True

        # Run
        actual_loss_values = instance.get_loss_values()

        # Assert
        pd.testing.assert_frame_equal(actual_loss_values, loss_values)

    def test_get_loss_values_error(self):
        """Test the ``get_loss_values`` errors if synthesizer has not been fitted."""
        # Setup
        metadata = SingleTableMetadata()
        instance = TVAESynthesizer(metadata)

        # Run / Assert
        msg = 'Loss values are not available yet. Please fit your synthesizer first.'
        with pytest.raises(NotFittedError, match=msg):
            instance.get_loss_values()
