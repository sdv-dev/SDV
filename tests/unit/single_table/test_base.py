import re
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from rdt.transformers import (
    BinaryEncoder, FloatFormatter, GaussianNormalizer, OneHotEncoder, RegexGenerator)

from sdv.errors import Error
from sdv.metadata.single_table import SingleTableMetadata
from sdv.single_table import (
    CopulaGANSynthesizer, CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer)
from sdv.single_table.base import BaseSynthesizer
from sdv.single_table.errors import InvalidDataError


class TestBaseSynthesizer:

    @patch('sdv.single_table.base.DataProcessor')
    def test___init__(self, mock_data_processor):
        """Test instantiating with default values."""
        # Setup
        metadata = Mock()

        # Run
        instance = BaseSynthesizer(metadata)

        # Assert
        assert instance.enforce_min_max_values is True
        assert instance.enforce_rounding is True
        assert instance._data_processor == mock_data_processor.return_value
        mock_data_processor.assert_called_once_with(metadata)

    @patch('sdv.single_table.base.DataProcessor')
    def test___init__custom(self, mock_data_processor):
        """Test that instantiating with custom parameters are properly stored in the instance."""
        # Setup
        metadata = Mock()

        # Run
        instance = BaseSynthesizer(metadata, enforce_min_max_values=False, enforce_rounding=False)

        # Assert
        assert instance.enforce_min_max_values is False
        assert instance.enforce_rounding is False
        assert instance._data_processor == mock_data_processor.return_value
        mock_data_processor.assert_called_once_with(metadata)

    @patch('sdv.single_table.base.DataProcessor')
    def test_get_parameters(self, mock_data_processor):
        """Test that it returns every ``init`` parameter without the ``metadata``."""
        # Setup
        metadata = Mock()
        instance = BaseSynthesizer(metadata, enforce_min_max_values=False, enforce_rounding=False)

        # Run
        parameters = instance.get_parameters()

        # Assert
        assert 'metadata' not in parameters
        assert parameters == {'enforce_min_max_values': False, 'enforce_rounding': False}

    @patch('sdv.single_table.base.DataProcessor')
    def test_get_metadata(self, mock_data_processor):
        """Test that it returns the ``metadata`` object."""
        # Setup
        metadata = Mock()
        instance = BaseSynthesizer(metadata, enforce_min_max_values=False, enforce_rounding=False)

        # Run
        result = instance.get_metadata()

        # Assert
        assert result == metadata

    @patch('sdv.single_table.base.DataProcessor')
    def test__fit(self, mock_data_processor):
        """Test that ``NotImplementedError`` is being raised."""
        # Setup
        metadata = Mock()
        data = Mock()
        instance = BaseSynthesizer(metadata)

        # Run and Assert
        with pytest.raises(NotImplementedError, match=''):
            instance._fit(data)

    def test_fit_processed_data(self):
        """Test that ``fit_processed_data`` calls the ``_fit``."""
        # Setup
        instance = Mock()
        processed_data = Mock()

        # Run
        BaseSynthesizer.fit_processed_data(instance, processed_data)

        # Assert
        instance._fit.assert_called_once_with(processed_data)

    def test_fit(self):
        """Test that ``fit`` calls ``preprocess`` and the ``fit_processed_data``.

        When fitting, the synthsizer has to ``preprocess`` the data and with the output
        of this method, call the ``fit_processed_data``
        """
        # Setup
        instance = Mock()
        processed_data = Mock()

        # Run
        BaseSynthesizer.fit(instance, processed_data)

        # Assert
        instance.preprocess.assert_called_once_with(processed_data)
        instance.fit_processed_data.assert_called_once_with(instance.preprocess.return_value)

    def test_validate_type(self):
        """Test error is raised if data is not ``pd.DataFrame``."""
        # Setup
        data = np.ndarray([])
        metadata = SingleTableMetadata()
        instance = BaseSynthesizer(metadata)

        # Run and Assert
        err_msg = "Data must be a DataFrame, not a <class 'numpy.ndarray'>."
        with pytest.raises(ValueError, match=err_msg):
            instance.validate(data)

    def test_validate_data_columns_in_empty_metadata(self):
        """Test error is raised if data is passed and metadata is empty."""
        # Setup
        data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
        })
        metadata = SingleTableMetadata()
        instance = BaseSynthesizer(metadata)

        # Run and Assert
        err_msg = re.escape(
            'The provided data does not match the metadata:'
            "\nThe columns ['col1', 'col2'] are not present in the metadata."
        )
        with pytest.raises(InvalidDataError, match=err_msg):
            instance.validate(data)

    def test_validate_data_columns_in_metadata(self):
        """Test error is raised if data columns don't match metadata columns."""
        # Setup
        data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9],
        })
        metadata = SingleTableMetadata()
        metadata.add_column('col1', sdtype='numerical')
        metadata.add_column('col4', sdtype='numerical')
        metadata.add_column('col5', sdtype='numerical')
        instance = BaseSynthesizer(metadata)

        # Run and Assert
        err_msg = re.escape(
            'The provided data does not match the metadata:'
            "\nThe columns ['col2', 'col3'] are not present in the metadata."
            '\n'
            "\nThe metadata columns ['col4', 'col5'] are not present in the data."
        )
        with pytest.raises(InvalidDataError, match=err_msg):
            instance.validate(data)

    def test_validate_keys_with_missing_values(self):
        """Test error is raised if keys contain missing values.

        Setup:
            A ``SingleTableMetadata`` instance with one primary key and multiple sequence
            and alternate keys. All the columns contain missing values except for one
            squence key and one alternate key, so we can ensure those don't show up
            in the error message.
        """
        data = pd.DataFrame({
            'pk_col': [0, 1, np.nan],
            'sk_col1': [0, 1, None],
            'sk_col2': [0, 1, np.nan],
            'sk_col3': [0, 1, 2],
            'ak_col1': [0, 1, None],
            'ak_col2': [0, 1, np.nan],
            'ak_col3': [0, 1, 2]
        })
        metadata = SingleTableMetadata()
        metadata.add_column('pk_col', sdtype='numerical')
        metadata.add_column('sk_col1', sdtype='numerical')
        metadata.add_column('sk_col2', sdtype='numerical')
        metadata.add_column('sk_col3', sdtype='numerical')
        metadata.add_column('ak_col1', sdtype='numerical')
        metadata.add_column('ak_col2', sdtype='numerical')
        metadata.add_column('ak_col3', sdtype='numerical')
        metadata.set_primary_key('pk_col')
        metadata.set_sequence_key(('sk_col1', 'sk_col2', 'sk_col3'))
        metadata.add_alternate_keys(['ak_col1', 'ak_col2', 'ak_col3'])
        instance = BaseSynthesizer(metadata)

        # Run and Assert
        err_msg = re.escape(
            'The provided data does not match the metadata:'
            "\nKey column 'ak_col1' contains missing values."
            '\n'
            "\nKey column 'ak_col2' contains missing values."
            '\n'
            "\nKey column 'pk_col' contains missing values."
            '\n'
            "\nKey column 'sk_col1' contains missing values."
            '\n'
            "\nKey column 'sk_col2' contains missing values."
        )
        with pytest.raises(InvalidDataError, match=err_msg):
            instance.validate(data)

    def test_validate_keys_with_missing2(self):
        """Test error is raised if keys contain missing values.

        Test the case with a single sequence key.
        """
        data = pd.DataFrame({
            'pk_col': [1],
            'sk_col': [None]
        })
        metadata = SingleTableMetadata()
        metadata.add_column('pk_col', sdtype='numerical')
        metadata.add_column('sk_col', sdtype='numerical')
        metadata.set_primary_key('pk_col')
        metadata.set_sequence_key('sk_col')
        instance = BaseSynthesizer(metadata)

        # Run and Assert
        err_msg = re.escape(
            'The provided data does not match the metadata:'
            "\nKey column 'sk_col' contains missing values."
        )
        with pytest.raises(InvalidDataError, match=err_msg):
            instance.validate(data)

    def test_validate_keys_not_unique(self):
        """Test error is raised if primary or alternate keys are not unique."""
        data = pd.DataFrame({
            'pk_col': [0, 1, 1, 0, 2],
            'ak_col1': [0, 1, 0, 3, 3],
            'ak_col2': [2, 2, 2, 2, 2],
            'ak_col3': [0, 1, 2, 3, 4]
        })
        metadata = SingleTableMetadata()
        metadata.add_column('pk_col', sdtype='numerical')
        metadata.add_column('ak_col1', sdtype='numerical')
        metadata.add_column('ak_col2', sdtype='numerical')
        metadata.add_column('ak_col3', sdtype='numerical')
        metadata.set_primary_key('pk_col')
        metadata.add_alternate_keys(['ak_col1', 'ak_col2', 'ak_col3'])
        instance = BaseSynthesizer(metadata)

        # Run and Assert
        err_msg = re.escape(
            'The provided data does not match the metadata:'
            "\nKey column 'ak_col1' contains repeating values: [0, 3]"
            '\n'
            "\nKey column 'ak_col2' contains repeating values: [2]"
            '\n'
            "\nKey column 'pk_col' contains repeating values: [0, 1]"
        )
        with pytest.raises(InvalidDataError, match=err_msg):
            instance.validate(data)

    def test_validate_context_columns_unique_per_sequence_key(self):
        """Test error is raised if context column values vary for each tuple of sequence keys.

        Setup:
            A ``SingleTableMetadata`` instance where the context columns vary for different
            combinations of values of the sequence keys.
        """
        # Setup
        data = pd.DataFrame({
            'sk_col1': [1, 1, 2, 2, 2],
            'sk_col2': [1, 1, 2, 2, 3],
            'ct_col1': [1, 2, 2, 3, 2],
            'ct_col2': [3, 3, 4, 3, 2],
        })
        metadata = SingleTableMetadata()
        metadata.add_column('sk_col1', sdtype='numerical')
        metadata.add_column('sk_col2', sdtype='numerical')
        metadata.add_column('ct_col1', sdtype='numerical')
        metadata.add_column('ct_col2', sdtype='numerical')
        metadata.set_sequence_key(('sk_col1', 'sk_col2'))
        instance = BaseSynthesizer(metadata)
        instance._data_processor._model_kwargs = {
            'context_columns': ['ct_col1', 'ct_col2']  # NOTE: change to actual value
        }

        # Run and Assert
        err_msg = re.escape(
            'The provided data does not match the metadata:'
            "\nContext column(s) {'ct_col1': {1, 2}} are changing inside the "
            "sequence keys (['sk_col1', 'sk_col2']: (1, 1))."
            '\n'
            "\nContext column(s) {'ct_col1': {2, 3}, 'ct_col2': {3, 4}} are changing inside the "
            "sequence keys (['sk_col1', 'sk_col2']: (2, 2))."
        )
        with pytest.raises(InvalidDataError, match=err_msg):
            instance.validate(data)

    def test_validate_empty(self):
        """Test method doesn't raise when data is empty.

        Setup:
            ``SingleTableMetadata`` with one column for each sdtype and for each key.
        """
        data = pd.DataFrame({
            'pk_col': [],
            'sk_col': [],
            'ak_col': [],
            'bool_col': [],
            'num_col': [],
            'date_col': [],
        })
        metadata = SingleTableMetadata()
        metadata.add_column('bool_col', sdtype='boolean')
        metadata.add_column('num_col', sdtype='numerical')
        metadata.add_column('date_col', sdtype='datetime')
        metadata.add_column('pk_col', sdtype='numerical')
        metadata.add_column('sk_col', sdtype='numerical')
        metadata.add_column('ak_col', sdtype='numerical')
        metadata.set_primary_key('pk_col')
        metadata.set_sequence_key('sk_col')
        metadata.add_alternate_keys(['ak_col'])
        instance = BaseSynthesizer(metadata)

        # Run
        instance.validate(data)

    def test_validate_no_keys(self):
        """Test method passes even if no keys are passed."""
        data = pd.DataFrame({
            'bool_col': [1, 2, 3],
            'num_col': [1, 2, 3],
            'date_col': [1, 2, 3],
        })
        metadata = SingleTableMetadata()
        metadata.add_column('bool_col', sdtype='numerical')
        metadata.add_column('num_col', sdtype='numerical')
        metadata.add_column('date_col', sdtype='numerical')
        instance = BaseSynthesizer(metadata)

        # Run
        instance.validate(data)

    def test_validate_empty_dataframe(self):
        """Test method doesn't raise when data is an empty dataframe."""
        data = pd.DataFrame()
        metadata = SingleTableMetadata()
        instance = BaseSynthesizer(metadata)

        # Run
        instance.validate(data)

    def test_validate_sdtypes(self):
        """Test error is raised if column values don't satisfy their sdtype.

        Setup:
            A ``SingleTableMetadata`` instance with two columns of each sdtype: numerical,
            boolean and datetime. The first column of each will have 4 invalid values,
            while the second column will have at most 3.
        """
        # Setup
        data = pd.DataFrame({
            'date1': ['10', True, 'b', 'bla', None],
            'date2': ['2021-10-10', '05-10-2021', pd.Timestamp(1), datetime(1, 1, 1), '2020-1-33'],
            'bool1': ['a', 0, '10', True, 'b'],
            'bool2': ['True', False, np.nan, float('nan'), None],
            'num1': ['a', 0, '10', True, False],
            'num2': [-1.2, datetime(1, 1, 1), np.nan, float('nan'), None],
        })
        metadata = SingleTableMetadata()
        metadata.add_column('date1', sdtype='datetime')
        metadata.add_column('date2', sdtype='datetime')
        metadata.add_column('bool1', sdtype='boolean')
        metadata.add_column('bool2', sdtype='boolean')
        metadata.add_column('num1', sdtype='numerical')
        metadata.add_column('num2', sdtype='numerical')
        instance = BaseSynthesizer(metadata)

        # Run and Assert
        err_msg = re.escape(
            'The provided data does not match the metadata:'
            "\nInvalid values found for datetime column 'date1': ['10', True, 'b', '+ 1 more']."
            '\n'
            "\nInvalid values found for datetime column 'date2': ['2020-1-33']."
            '\n'
            "\nInvalid values found for boolean column 'bool1': [0, '10', 'a', '+ 1 more']."
            '\n'
            "\nInvalid values found for boolean column 'bool2': ['True']."
            '\n'
            "\nInvalid values found for numerical column 'num1': ['10', False, True, '+ 1 more']."
            '\n'
            "\nInvalid values found for numerical column 'num2': "
            '[datetime.datetime(1, 1, 1, 0, 0)].'
        )
        with pytest.raises(InvalidDataError, match=err_msg):
            instance.validate(data)

    def test_validate(self):
        """Test the method doesn't crash when the passed data is valid.

        Setup:
            ``SingleTableMetadata`` describing at least one valid column of each key and sdtype.
        """
        # Setup
        data = pd.DataFrame({
            'pk_col': [0, 1, 2],
            'sk_col1': [0, 1, 2],
            'sk_col2': [0, 1, 2],
            'ak_col1': [0, 1, 2],
            'ak_col2': [0, 1, 2],
            'numerical_col': [np.nan, -1, 1.54],
            'date_col': [np.nan, '2021-02-10', '2021-05-10'],
            'bool_col': [np.nan, True, False],

        })
        metadata = SingleTableMetadata()
        metadata.add_column('pk_col', sdtype='numerical')
        metadata.add_column('sk_col1', sdtype='numerical')
        metadata.add_column('sk_col2', sdtype='numerical')
        metadata.add_column('ak_col1', sdtype='numerical')
        metadata.add_column('ak_col2', sdtype='numerical')
        metadata.add_column('numerical_col', sdtype='numerical')
        metadata.add_column('date_col', sdtype='datetime')
        metadata.add_column('bool_col', sdtype='boolean')
        metadata.set_primary_key('pk_col')
        metadata.set_sequence_key(('sk_col1', 'sk_col2'))
        metadata.add_alternate_keys(['ak_col1', 'ak_col2'])
        instance = BaseSynthesizer(metadata)

        # Run
        instance.validate(data)

    def test_update_transformers_invalid_sdtype(self):
        """Test error is raised if transformer doesn't match column sdtype."""
        # Setup
        column_name_to_transformer = {
            'col1': BinaryEncoder(),
            'col2': FloatFormatter()
        }
        metadata = SingleTableMetadata()
        metadata.add_column('col1', sdtype='boolean')
        metadata.add_column('col2', sdtype='categorical')
        instance = BaseSynthesizer(metadata)

        # Run and Assert
        err_msg = re.escape(
            "Column 'col2' is a categorical column, which is incompatible "
            "with the 'FloatFormatter' transformer."
        )
        with pytest.raises(Error, match=err_msg):
            instance.update_transformers(column_name_to_transformer)

    def test_update_transformers_invalid_keys(self):
        """Test error is raised if passed transformer doesn't match key column.

        The transformers of a key column must be either AnonymizedFaker or RegexGenerator.
        Raise an error if any other transformer is passed.
        """
        # Setup
        column_name_to_transformer = {
            'col2': RegexGenerator(),
            'col3': FloatFormatter()
        }
        metadata = SingleTableMetadata()
        metadata.add_column('col2', sdtype='text')
        metadata.add_column('col3', sdtype='numerical')
        metadata.set_sequence_key(('col2'))
        metadata.add_alternate_keys(['col3'])
        instance = BaseSynthesizer(metadata)

        # Run and Assert
        err_msg = re.escape(
            "Column 'col3' is a key. It cannot be preprocessed using "
            "the 'FloatFormatter' transformer."
        )
        with pytest.raises(Error, match=err_msg):
            instance.update_transformers(column_name_to_transformer)

    def test_update_transformers_already_fitted(self):
        """Test error is raised if passed transformer was already fitted."""
        # Setup
        fitted_transformer = FloatFormatter()
        fitted_transformer.fit(pd.DataFrame({'col': [1]}), 'col')
        column_name_to_transformer = {
            'col1': BinaryEncoder(),
            'col2': fitted_transformer
        }
        metadata = SingleTableMetadata()
        metadata.add_column('col1', sdtype='boolean')
        metadata.add_column('col2', sdtype='numerical')
        instance = BaseSynthesizer(metadata)

        # Run and Assert
        err_msg = "Transformer for column 'col2' has already been fit on data."
        with pytest.raises(Error, match=err_msg):
            instance.update_transformers(column_name_to_transformer)

    def test_update_transformers_warns_gaussian_copula(self):
        """Test warning is raised when ohe is used for categorical column in the GaussianCopula."""
        # Setup
        column_name_to_transformer = {
            'col1': OneHotEncoder(),
            'col2': FloatFormatter()
        }
        metadata = SingleTableMetadata()
        metadata.add_column('col1', sdtype='categorical')
        metadata.add_column('col2', sdtype='numerical')
        instance = GaussianCopulaSynthesizer(metadata)
        instance._data_processor.fit(pd.DataFrame({'col1': [1, 2], 'col2': [1, 2]}))

        # Run and Assert
        warning = re.escape(
            "Using a OneHotEncoder transformer for column 'col1' "
            'may slow down the preprocessing and modeling times.'
        )
        with pytest.warns(UserWarning, match=warning):
            instance.update_transformers(column_name_to_transformer)

        field_transformers = instance._data_processor._hyper_transformer.field_transformers
        assert len(field_transformers) == 2
        assert isinstance(field_transformers['col1'], OneHotEncoder)
        assert isinstance(field_transformers['col2'], FloatFormatter)

    def test_update_transformers_warns_models(self):
        """Test warnings if transformer is added for a column that is auto-assigned to None.

        NOTE: improve description after figuring out what this is supposed to be testing :)
        """
        # Setup
        column_name_to_transformer = {
            'col1': OneHotEncoder(),
            'col2': FloatFormatter()
        }
        metadata = SingleTableMetadata()
        metadata.add_column('col1', sdtype='categorical')
        metadata.add_column('col2', sdtype='numerical')

        # NOTE: when PARSynthesizer is implemented, add it here as well
        for model in [CTGANSynthesizer, CopulaGANSynthesizer, TVAESynthesizer]:
            instance = model(metadata)
            instance._data_processor.fit(pd.DataFrame({'col1': [1, 2], 'col2': [1, 2]}))

            # Run and Assert
            warning = re.escape(
                "Replacing the default transformer for column 'col1' might "
                'impact the quality of your synthetic data.'
            )
            with pytest.warns(UserWarning, match=warning):
                instance.update_transformers(column_name_to_transformer)

    def test_update_transformers(self):
        """Test method correctly updates the transformers in the HyperTransformer."""
        # Setup
        column_name_to_transformer = {
            'col1': GaussianNormalizer(),
            'col2': GaussianNormalizer()
        }
        metadata = SingleTableMetadata()
        metadata.add_column('col1', sdtype='numerical')
        metadata.add_column('col2', sdtype='numerical')
        instance = BaseSynthesizer(metadata)
        instance._data_processor.fit(pd.DataFrame({'col1': [1, 2], 'col2': [1, 2]}))

        # Run
        instance.update_transformers(column_name_to_transformer)

        # Assert
        field_transformers = instance._data_processor._hyper_transformer.field_transformers
        assert len(field_transformers) == 2
        assert isinstance(field_transformers['col1'], GaussianNormalizer)
        assert isinstance(field_transformers['col2'], GaussianNormalizer)
