import re
from datetime import date, datetime
from unittest.mock import ANY, MagicMock, Mock, call, mock_open, patch

import numpy as np
import pandas as pd
import pytest
from copulas.multivariate import GaussianMultivariate
from rdt.transformers import (
    BinaryEncoder, FloatFormatter, GaussianNormalizer, OneHotEncoder, RegexGenerator)

from sdv.constraints.errors import AggregateConstraintsError
from sdv.errors import ConstraintsNotMetError, SynthesizerInputError
from sdv.metadata.single_table import SingleTableMetadata
from sdv.sampling.tabular import Condition
from sdv.single_table import (
    CopulaGANSynthesizer, CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer)
from sdv.single_table.base import COND_IDX, BaseSingleTableSynthesizer
from sdv.single_table.errors import InvalidDataError


class TestBaseSingleTableSynthesizer:

    def test__update_default_transformers(self):
        """Test that ``instance._data_processor._update_transformers_by_sdtypes`` is called.

        Test when there are ``_model_sdtype_transformers`` set, this method will call
        the data processor and update the default ones.
        """
        # Setup
        instance = Mock()
        instance._model_sdtype_transformers = {
            'categorical': None,
            'numerical': 'FloatTransformer'
        }

        # Run
        BaseSingleTableSynthesizer._update_default_transformers(instance)

        # Assert
        call_list = instance._data_processor._update_transformers_by_sdtypes.call_args_list
        assert call_list == [call('categorical', None), call('numerical', 'FloatTransformer')]

    @patch('sdv.single_table.base.DataProcessor')
    def test___init__(self, mock_data_processor):
        """Test instantiating with default values."""
        # Setup
        metadata = Mock()

        # Run
        instance = BaseSingleTableSynthesizer(metadata)

        # Assert
        assert instance.enforce_min_max_values is True
        assert instance.enforce_rounding is True
        assert instance._data_processor == mock_data_processor.return_value
        assert instance._random_state_set is False
        assert instance._fitted is False
        mock_data_processor.assert_called_once_with(
            metadata=metadata,
            enforce_rounding=instance.enforce_rounding,
            enforce_min_max_values=instance.enforce_min_max_values,
            locales=instance.locales
        )
        metadata.validate.assert_called_once_with()

    @patch('sdv.single_table.base.DataProcessor')
    def test___init__custom(self, mock_data_processor):
        """Test that instantiating with custom parameters are properly stored in the instance."""
        # Setup
        metadata = Mock()

        # Run
        instance = BaseSingleTableSynthesizer(
            metadata,
            enforce_min_max_values=False,
            enforce_rounding=False,
            locales='en_CA'
        )

        # Assert
        assert instance.enforce_min_max_values is False
        assert instance.enforce_rounding is False
        assert instance.locales == 'en_CA'
        assert instance._data_processor == mock_data_processor.return_value
        mock_data_processor.assert_called_once_with(
            metadata=metadata,
            enforce_rounding=instance.enforce_rounding,
            enforce_min_max_values=instance.enforce_min_max_values,
            locales=instance.locales
        )
        metadata.validate.assert_called_once_with()

    def test___init__invalid_enforce_min_max_values(self):
        """Test it crashes when ``enforce_min_max_values`` is not a boolean."""
        # Run and Assert
        err_msg = re.escape(
            "Invalid value 'invalid' for parameter 'enforce_min_max_values'."
            ' Please provide True or False.'
        )
        with pytest.raises(SynthesizerInputError, match=err_msg):
            BaseSingleTableSynthesizer(SingleTableMetadata(), enforce_min_max_values='invalid')

    def test___init__invalid_enforce_rounding(self):
        """Test it crashes when ``enforce_rounding`` is not a boolean."""
        # Run and Assert
        err_msg = re.escape(
            "Invalid value 'invalid' for parameter 'enforce_rounding'."
            ' Please provide True or False.'
        )
        with pytest.raises(SynthesizerInputError, match=err_msg):
            BaseSingleTableSynthesizer(SingleTableMetadata(), enforce_rounding='invalid')

    @patch('sdv.single_table.base.DataProcessor')
    def test_get_parameters(self, mock_data_processor):
        """Test that it returns every ``init`` parameter without the ``metadata``."""
        # Setup
        metadata = Mock()
        instance = BaseSingleTableSynthesizer(
            metadata,
            enforce_min_max_values=False,
            enforce_rounding=False,
            locales='en_CA'
        )

        # Run
        parameters = instance.get_parameters()

        # Assert
        assert 'metadata' not in parameters
        assert parameters == {
            'enforce_min_max_values': False,
            'enforce_rounding': False,
            'locales': 'en_CA'
        }

    @patch('sdv.single_table.base.DataProcessor')
    def test_get_metadata(self, mock_data_processor):
        """Test that it returns the ``metadata`` object."""
        # Setup
        metadata = Mock()
        instance = BaseSingleTableSynthesizer(
            metadata,
            enforce_min_max_values=False,
            enforce_rounding=False
        )

        # Run
        result = instance.get_metadata()

        # Assert
        assert result == metadata

    def test_auto_assign_transformers(self):
        """Test that the ``DataProcessor.prepare_for_fitting`` is being called."""
        # Setup
        instance = Mock()
        data = pd.DataFrame({
            'name': ['John', 'Doe', 'Johanna'],
            'salary': [80., 90., 120.]
        })

        # Run
        BaseSingleTableSynthesizer.auto_assign_transformers(instance, data)

        # Assert
        instance._data_processor.prepare_for_fitting.assert_called_once_with(data)

    def test_get_transformers(self):
        """Test that this returns the field transformers from the ``HyperTransformer``."""
        # Setup
        instance = Mock()
        instance._data_processor._hyper_transformer.field_transformers = {
            'name': 'FrequencyEncoder',
            'salary': 'FloatFormatter',
            'salary#name': 'LabelEncoder',
            'address': None
        }
        instance.metadata.columns = {
            'salary': {'sdtype': 'numerical'},
            'name': {'sdtype': 'categorical'},
            'address': {'sdtype': 'address'}
        }

        # Run
        result = BaseSingleTableSynthesizer.get_transformers(instance)

        # Assert
        assert result == {
            'salary': 'FloatFormatter',
            'name': 'FrequencyEncoder',
            'address': None,
            'salary#name': 'LabelEncoder'
        }

    def test_get_transformers_with_columns_dropped_by_constraint(self):
        """Test that this returns the transformers that are within the ``field_transformers``."""
        # Setup
        instance = Mock()
        instance._data_processor._hyper_transformer.field_transformers = {
            'salary#name': 'LabelEncoder',
            'address': None
        }
        instance.metadata.columns = {
            'salary': {'sdtype': 'numerical'},
            'name': {'sdtype': 'categorical'},
            'address': {'sdtype': 'address'}
        }

        # Run
        result = BaseSingleTableSynthesizer.get_transformers(instance)

        # Assert
        assert result == {
            'address': None,
            'salary#name': 'LabelEncoder'
        }

    def test_get_transformers_raises_an_error(self):
        """Test that this raises an error when there are no field transformers."""
        # Setup
        instance = Mock()
        instance._data_processor._hyper_transformer.field_transformers = {}

        # Run and Assert
        error_msg = re.escape(
            "No transformers were returned in 'get_transformers'. Use "
            "'auto_assign_transformers' or 'fit' to create them."
        )
        with pytest.raises(ValueError, match=error_msg):
            BaseSingleTableSynthesizer.get_transformers(instance)

    def test__preprocess(self):
        """Test the method preprocesses the data.

        The method calls the ``validate`` function with the data, then fits the
        ``instance._data_processor`` and returns the output of the transformation.
        """
        # Setup
        instance = Mock()
        instance._fitted = True
        data = pd.DataFrame({
            'name': ['John', 'Doe', 'John Doe']
        })

        # Run
        result = BaseSingleTableSynthesizer._preprocess(instance, data)

        # Assert
        instance.validate.assert_called_once()
        pd.testing.assert_frame_equal(
            instance.validate.call_args_list[0][0][0],
            data
        )
        assert result == instance._data_processor.transform.return_value
        instance._data_processor.fit.assert_called_once()
        pd.testing.assert_frame_equal(
            data,
            instance._data_processor.fit.call_args_list[0][0][0]
        )
        pd.testing.assert_frame_equal(
            data,
            instance._data_processor.transform.call_args_list[0][0][0]
        )

    @patch('sdv.single_table.base.warnings')
    def test_preprocess(self, mock_warnings):
        """Test the preprocess method.

        The preprocess method raises a warning if it was already fitted and then calls
        ``_preprocess``.
        """
        # Setup
        instance = Mock()
        instance._fitted = True
        data = pd.DataFrame({
            'name': ['John', 'Doe', 'John Doe']
        })

        # Run
        BaseSingleTableSynthesizer.preprocess(instance, data)

        # Assert
        expected_warning = (
            'This model has already been fitted. To use the new preprocessed data, please '
            "refit the model using 'fit' or 'fit_processed_data'."
        )
        mock_warnings.warn.assert_called_once_with(expected_warning)
        instance._preprocess.assert_called_once_with(data)

    @patch('sdv.single_table.base.DataProcessor')
    def test__fit(self, mock_data_processor):
        """Test that ``NotImplementedError`` is being raised."""
        # Setup
        metadata = Mock()
        data = Mock()
        instance = BaseSingleTableSynthesizer(metadata)

        # Run and Assert
        with pytest.raises(NotImplementedError, match=''):
            instance._fit(data)

    def test_fit_processed_data(self):
        """Test that ``fit_processed_data`` calls the ``_fit``."""
        # Setup
        instance = Mock()
        processed_data = Mock()
        processed_data.empty = False

        # Run
        BaseSingleTableSynthesizer.fit_processed_data(instance, processed_data)

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
        instance._random_state_set = True
        instance._fitted = True

        # Run
        BaseSingleTableSynthesizer.fit(instance, processed_data)

        # Assert
        assert instance._random_state_set is False
        instance._data_processor.reset_sampling.assert_called_once_with()
        instance._preprocess.assert_called_once_with(processed_data)
        instance.fit_processed_data.assert_called_once_with(instance._preprocess.return_value)

    def test_validate_type(self):
        """Test error is raised if data is not ``pd.DataFrame``."""
        # Setup
        data = np.ndarray([])
        metadata = SingleTableMetadata()
        instance = BaseSingleTableSynthesizer(metadata)

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
        instance = BaseSingleTableSynthesizer(metadata)

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
        instance = BaseSingleTableSynthesizer(metadata)

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
        metadata.add_column('pk_col', sdtype='id')
        metadata.add_column('sk_col1', sdtype='id')
        metadata.add_column('sk_col2', sdtype='id')
        metadata.add_column('sk_col3', sdtype='id')
        metadata.add_column('ak_col1', sdtype='id')
        metadata.add_column('ak_col2', sdtype='id')
        metadata.add_column('ak_col3', sdtype='id')
        metadata.set_primary_key('pk_col')
        metadata.set_sequence_key(('sk_col1', 'sk_col2', 'sk_col3'))
        metadata.add_alternate_keys(['ak_col1', 'ak_col2', 'ak_col3'])
        instance = BaseSingleTableSynthesizer(metadata)

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

    def test_validate_keys_with_missing_with_single_sequence_key(self):
        """Test error is raised if keys contain missing values.

        Test the case with a single sequence key.
        """
        data = pd.DataFrame({
            'pk_col': [1],
            'sk_col': [None]
        })
        metadata = SingleTableMetadata()
        metadata.add_column('pk_col', sdtype='id')
        metadata.add_column('sk_col', sdtype='id')
        metadata.set_primary_key('pk_col')
        metadata.set_sequence_key('sk_col')
        instance = BaseSingleTableSynthesizer(metadata)

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
        metadata.add_column('pk_col', sdtype='id')
        metadata.add_column('ak_col1', sdtype='id')
        metadata.add_column('ak_col2', sdtype='id')
        metadata.add_column('ak_col3', sdtype='id')
        metadata.set_primary_key('pk_col')
        metadata.add_alternate_keys(['ak_col1', 'ak_col2', 'ak_col3'])
        instance = BaseSingleTableSynthesizer(metadata)

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
        metadata.add_column('pk_col', sdtype='id')
        metadata.add_column('sk_col', sdtype='id')
        metadata.add_column('ak_col', sdtype='id')
        metadata.set_primary_key('pk_col')
        metadata.set_sequence_key('sk_col')
        metadata.add_alternate_keys(['ak_col'])
        instance = BaseSingleTableSynthesizer(metadata)

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
        instance = BaseSingleTableSynthesizer(metadata)

        # Run
        instance.validate(data)

    def test_validate_empty_dataframe(self):
        """Test method doesn't raise when data is an empty dataframe."""
        data = pd.DataFrame()
        metadata = SingleTableMetadata()
        instance = BaseSingleTableSynthesizer(metadata)

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
        instance = BaseSingleTableSynthesizer(metadata)

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

    def test_validate_datetime_sdtype(self):
        """Test validation for columns with datetime format.

        If the datetime format is provided, then the values must match. Otherwise an error should
        be raised.
        """
        # Setup
        data = pd.DataFrame({
            'date_str': [
                '20220902110443000000',
                '20220916230356000000',
                '20220826173917000000',
                '20220826212135000000',
                '20220929111311000000'
            ],
            'date_int': [
                20220902110443000000,
                20220916230356000000,
                20220826173917000000,
                20220826212135000000,
                20220929111311000000
            ],
            'bad_date': [
                2022090,
                20220916230356000000,
                2022,
                20220826212135000000,
                20220929111311000000
            ]
        })
        metadata = SingleTableMetadata()
        metadata.add_column('date_str', sdtype='datetime', datetime_format='%Y%m%d%H%M%S%f')
        metadata.add_column('date_int', sdtype='datetime', datetime_format='%Y%m%d%H%M%S%f')
        metadata.add_column('bad_date', sdtype='datetime', datetime_format='%Y%m%d%H%M%S%f')
        instance = BaseSingleTableSynthesizer(metadata)

        # Run and Assert
        err_msg = re.escape(
            'The provided data does not match the metadata:'
            "\nInvalid values found for datetime column 'bad_date': [2022, 2022090]."
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
        metadata.add_column('pk_col', sdtype='id')
        metadata.add_column('sk_col1', sdtype='id')
        metadata.add_column('sk_col2', sdtype='id')
        metadata.add_column('ak_col1', sdtype='id')
        metadata.add_column('ak_col2', sdtype='id')
        metadata.add_column('numerical_col', sdtype='numerical')
        metadata.add_column('date_col', sdtype='datetime')
        metadata.add_column('bool_col', sdtype='boolean')
        metadata.set_primary_key('pk_col')
        metadata.set_sequence_key(('sk_col1', 'sk_col2'))
        metadata.add_alternate_keys(['ak_col1', 'ak_col2'])
        instance = BaseSingleTableSynthesizer(metadata)

        # Run
        instance.validate(data)

    def test__validate_constraints(self):
        """Test that ``_validate_constraints`` calls ``fit`` and returns any errors."""
        # Setup
        instance = Mock()
        msg = 'Invalid data for constraint.'
        instance._data_processor._fit_constraints.side_effect = AggregateConstraintsError([msg])
        data = object()

        # Run
        errors = BaseSingleTableSynthesizer._validate_constraints(instance, data)

        # Assert
        assert str(errors[0]) == '\nInvalid data for constraint.'
        assert len(errors) == 1
        instance._data_processor._fit_constraints.assert_called_once_with(data)

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
        metadata.add_column('col2', sdtype='id')
        metadata.add_column('col3', sdtype='id')
        metadata.set_sequence_key(('col2'))
        metadata.add_alternate_keys(['col3'])
        instance = BaseSingleTableSynthesizer(metadata)

        # Run and Assert
        err_msg = re.escape(
            "Column 'col3' is a key. It cannot be preprocessed using "
            "the 'FloatFormatter' transformer."
        )
        with pytest.raises(SynthesizerInputError, match=err_msg):
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
        instance = BaseSingleTableSynthesizer(metadata)

        # Run and Assert
        err_msg = "Transformer for column 'col2' has already been fit on data."
        with pytest.raises(SynthesizerInputError, match=err_msg):
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
        """Test warning is raised for some models.

        A warning should be raised if a transformer is assigned to boolean/categorical columns for
        the CTGANSynthesizer, CopulaGANSynthesizer, TVAESynthesizer and PARSynthesizer models.
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

    def test_update_transformers_warns_fitted(self):
        """Test warning is raised if model is fitted.

        A warning telling the user they need to refit the model should be raised if a transformer
        is updated after the model has been fitted.
        """
        # Setup
        column_name_to_transformer = {
            'col1': GaussianNormalizer(),
            'col2': GaussianNormalizer()
        }
        metadata = SingleTableMetadata()
        metadata.add_column('col1', sdtype='numerical')
        metadata.add_column('col2', sdtype='numerical')
        instance = BaseSingleTableSynthesizer(metadata)
        instance._data_processor.fit(pd.DataFrame({'col1': [1, 2], 'col2': [1, 2]}))
        instance._fitted = True

        # Run and Assert
        warning_msg = re.escape('For this change to take effect, please refit the synthesizer '
                                'using `fit`.')
        with pytest.warns(UserWarning, match=warning_msg):
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
        instance = BaseSingleTableSynthesizer(metadata)
        instance._data_processor.fit(pd.DataFrame({'col1': [1, 2], 'col2': [1, 2]}))

        # Run
        instance.update_transformers(column_name_to_transformer)

        # Assert
        field_transformers = instance._data_processor._hyper_transformer.field_transformers
        assert len(field_transformers) == 2
        assert isinstance(field_transformers['col1'], GaussianNormalizer)
        assert isinstance(field_transformers['col2'], GaussianNormalizer)

    @patch('sdv.single_table.base.DataProcessor')
    def test__set_random_state(self, mock_data_processor):
        """Test that ``_model.set_random_state`` is being called with the input value.

        When passing a random state to the method ``_set_random_state``, this calls the
        instance's model ``set_random_state`` method in order to assign it to it.
        """
        # Setup
        rng_seed = Mock()
        metadata = Mock()
        instance = BaseSingleTableSynthesizer(metadata)
        instance._model = Mock()

        # Run
        instance._set_random_state(rng_seed)

        # Assert
        instance._model.set_random_state.assert_called_once_with(rng_seed)
        assert instance._random_state_set is True

    def test_reset_sampling(self):
        """Test the ``reset_sampling`` method.

        Ensure that the ``reset_sampling`` sets the ``instance._random_state_set`` is set to
        ``False`` and the ``instance._data_processor.reset_sampling`` is being called.
        """
        # Setup
        instance = Mock()
        instance._random_state_set = True

        # Run
        BaseSingleTableSynthesizer.reset_sampling(instance)

        # Assert
        assert instance._random_state_set is False
        instance._data_processor.reset_sampling.assert_called_once_with()

    def test__filter_conditions(self):
        """Test that the method filters out data that doesn't meet the conditions."""
        # Setup
        sampled = pd.DataFrame({
            'position': [
                'Software Engineer',
                'Data Scientist',
                'Statistician',
                'Computer Systems Analyst',
                'Software Engineer',
                'Software Engineer',
                'Computer Systems Analyst',
                'Software Engineer',
            ],
            'salary': [80., 90., 50., 60., 82., 80., 80., 81.]
        })
        conditions = {'position': 'Software Engineer', 'salary': 80.}
        float_rtol = 0.01

        # Run
        filtered_data = BaseSingleTableSynthesizer._filter_conditions(
            sampled, conditions, float_rtol)

        # Assert
        expected_data = pd.DataFrame({
            'position': ['Software Engineer', 'Software Engineer'],
            'salary': [80., 80.]
        }, index=[0, 5])
        pd.testing.assert_frame_equal(filtered_data, expected_data)

    def test__sample_rows_without_conditions(self):
        """Test that sample rows calls ``_sample`` when conditions is ``None``.

        Also ensure that when ``_random_state_set`` is ``False`` this calls
        ``_set_random_state`` with the ``FIXED_RNG_SEED`` which is ``73251``.
        """
        # Setup
        data = pd.DataFrame({
            'name': ['John', 'Doe', 'John Doe']
        })
        instance = Mock()
        instance._random_state_set = False
        instance._sample.return_value = pd.DataFrame()
        instance._data_processor.reverse_transform.return_value = data
        instance._data_processor.filter_valid.return_value = data
        instance._data_processor._hyper_transformer._input_columns = []

        # Run
        sampled, num_valid = BaseSingleTableSynthesizer._sample_rows(instance, 3)

        # Assert
        assert num_valid == 3
        pd.testing.assert_frame_equal(sampled, data)
        instance._sample.assert_called_once_with(3)
        instance._data_processor.reverse_transform.assert_called_once_with(
            instance._sample.return_value
        )
        instance._data_processor.filter_valid.assert_called_once_with(
            instance._data_processor.reverse_transform.return_value
        )
        instance._set_random_state.assert_called_once_with(73251)

    def test__sample_rows_with_conditions(self):
        """Test that sample rows calls with the transformed conditions the ``_sample``."""
        # Setup
        data = pd.DataFrame({
            'name': ['John', 'Doe', 'John Doe'],
            'salary': [90.0, 100.0, 80.0]
        })
        instance = Mock()
        instance._sample.return_value = pd.DataFrame()
        instance._data_processor.reverse_transform.return_value = data
        instance._data_processor._hyper_transformer._input_columns = []
        instance._filter_conditions.return_value = data[data.name == 'John Doe']
        conditions = {'salary': 80.}
        transformed_conditions = {'salary': 80.0}

        # Run
        sampled, num_valid = BaseSingleTableSynthesizer._sample_rows(
            instance,
            3,
            conditions=conditions,
            transformed_conditions=transformed_conditions
        )

        # Assert
        assert num_valid == 1
        pd.testing.assert_frame_equal(sampled, data[data.name == 'John Doe'])
        instance._sample.assert_called_once_with(3, {'salary': 80.0})
        instance._data_processor.reverse_transform.assert_called_once_with(
            instance._sample.return_value
        )
        instance._data_processor.filter_valid.assert_called_once_with(
            instance._data_processor.reverse_transform.return_value
        )

    def test__sample_rows_with_previous_rows(self):
        """Test that previous rows are being concatenated when provided to ``_sample``."""
        # Setup
        previous_rows = pd.DataFrame({
            'name': ['John Doe', 'John Doe', 'John Doe'],
            'salary': [80.0, 80.0, 80.0]
        })
        data = pd.DataFrame({
            'name': ['John', 'Doe', 'John Doe'],
            'salary': [90.0, 100.0, 80.0]
        })

        instance = Mock()
        instance._sample.return_value = pd.DataFrame()
        instance._data_processor._hyper_transformer._input_columns = []
        instance._data_processor.filter_valid = lambda x: x
        instance._data_processor.reverse_transform.return_value = data

        # Run
        sampled, num_valid = BaseSingleTableSynthesizer._sample_rows(
            instance, 3, previous_rows=previous_rows)

        # Assert
        expected_data = pd.DataFrame({
            'name': ['John Doe', 'John Doe', 'John Doe', 'John', 'Doe', 'John Doe'],
            'salary': [80.0, 80.0, 80.0, 90.0, 100.0, 80.0]
        })
        assert num_valid == 6
        pd.testing.assert_frame_equal(sampled, expected_data)
        instance._sample.assert_called_once_with(3)
        instance._data_processor.reverse_transform.assert_called_once_with(
            instance._sample.return_value
        )

    def test__sample_rows_notimplementederror(self):
        """Test when the model does not support conditional sampling and raises an error."""
        # Setup
        data = pd.DataFrame({
            'name': ['John', 'Doe', 'John Doe'],
            'salary': [90.0, 100.0, 80.0]
        })
        instance = Mock()
        instance._data_processor.reverse_transform.return_value = data
        instance._data_processor._hyper_transformer._input_columns = []
        instance._filter_conditions.return_value = data[data.name == 'John Doe']
        conditions = {'salary': 80.}
        transformed_conditions = {'salary': 80.0}
        instance._sample.side_effect = [NotImplementedError, pd.DataFrame()]

        # Run
        sampled, num_valid = BaseSingleTableSynthesizer._sample_rows(
            instance,
            3,
            conditions=conditions,
            transformed_conditions=transformed_conditions
        )

        # Assert
        assert num_valid == 1
        pd.testing.assert_frame_equal(sampled, data[data.name == 'John Doe'])
        assert instance._sample.call_args_list == [call(3, {'salary': 80.0}), call(3)]

    def test__sample_rows_sdtypes_is_empty(self):
        """Test when ``_data_processor.get_sdtypes`` with ``primary_keys=False`` is empty.

        This test is when ``_data_processor.get_sdtypes`` returns an empty dictionary, which leads
        to only sampling using ``_data_processor.reverse_transform``.
        """
        # Setup
        instance = Mock()
        instance._data_processor.get_sdtypes.return_value = {}
        instance._data_processor.reverse_transform.side_effect = lambda x: x

        # Run
        sampled, num_rows = BaseSingleTableSynthesizer._sample_rows(instance, 10)

        # Assert
        assert num_rows == 10
        pd.testing.assert_frame_equal(sampled, pd.DataFrame(index=range(10)))
        instance._data_processor.reverse_transform.assert_called_once()

    def test__sample_batch_without_saving_to_file(self):
        """Test the ``_sample_batch`` without storing the samples in a file."""
        # Setup
        sampled_data = pd.DataFrame({
            'name': ['John', 'Doe', 'John Doe'],
            'salary': [80., 60., 100.]
        })
        instance = Mock()
        instance.metadata.columns.keys.return_value = ['name', 'salary']
        instance._sample_rows.return_value = (sampled_data, 3)

        # Run
        result = BaseSingleTableSynthesizer._sample_batch(
            instance,
            batch_size=3,
            max_tries=100,
            conditions=None,
            transformed_conditions=None,
            float_rtol=0.01,
            progress_bar=None,
            output_file_path=None,
        )

        # Assert
        pd.testing.assert_frame_equal(result, sampled_data)
        _sample_rows_args = instance._sample_rows.call_args[0]
        rows, conditions, trans_cond, float_rtol, sampled, keep_extra_columns = _sample_rows_args
        assert rows == 3
        assert conditions is None
        assert trans_cond is None
        assert float_rtol == 0.01
        pd.testing.assert_frame_equal(sampled, pd.DataFrame())
        assert keep_extra_columns is False

    def test__sample_batch_with_sampled_data_bigger_than_batch_size(self):
        """Test ``sampled_data`` is bigger than the batch size.

        If the sampled data is bigger than the batch size, the returned data frame must be the
        ``batch_size`` size.
        """
        # Setup
        sampled_data = pd.DataFrame({
            'name': ['John', 'Doe', 'John Doe', 'John Doe John'],
            'salary': [80., 60., 100., 300.]
        })
        instance = Mock()
        instance.metadata.columns.keys.return_value = ['name', 'salary']
        instance._sample_rows.return_value = (sampled_data, 3)

        # Run
        result = BaseSingleTableSynthesizer._sample_batch(
            instance,
            batch_size=3,
            max_tries=100,
            conditions=None,
            transformed_conditions=None,
            float_rtol=0.01,
            progress_bar=None,
            output_file_path=None,
        )

        # Assert
        pd.testing.assert_frame_equal(result, sampled_data.head(3))
        _sample_rows_args = instance._sample_rows.call_args[0]
        rows, conditions, trans_cond, float_rtol, sampled, keep_extra_columns = _sample_rows_args
        assert rows == 3
        assert conditions is None
        assert trans_cond is None
        assert float_rtol == 0.01
        pd.testing.assert_frame_equal(sampled, pd.DataFrame())
        assert keep_extra_columns is False

    def test__sample_batch_max_tries_reached(self):
        """Test that when ``max_tries`` is reached, a break occurs."""
        # Setup
        sampled_data = pd.DataFrame({
            'name': ['John', 'Doe', 'John Doe', 'John Doe John'],
            'salary': [80., 60., 100., 300.]
        })
        instance = Mock()
        instance.metadata.columns.keys.return_value = ['name', 'salary']
        instance._sample_rows.return_value = (sampled_data, 3)

        # Run
        result = BaseSingleTableSynthesizer._sample_batch(
            instance,
            batch_size=10,
            max_tries=2,
            conditions=None,
            transformed_conditions=None,
            float_rtol=0.01,
            progress_bar=None,
            output_file_path=None,
        )

        # Assert
        pd.testing.assert_frame_equal(result, sampled_data)
        _sample_rows_args = instance._sample_rows.call_args[0]
        rows, conditions, trans_cond, float_rtol, sampled, keep_extra_columns = _sample_rows_args
        assert instance._sample_rows.call_count == 2

    def test__sample_batch_storing_output_file(self, tmpdir):
        """Test that an output file is properly stored while sampling.

        Test that an output file is properly stored as well as that the progress bar is updated.
        """
        # Setup
        sampled_data = pd.DataFrame({
            'name': ['John', 'Doe', 'John Doe', 'John Doe John'],
            'salary': [80., 60., 100., 300.]
        })
        instance = Mock()
        instance.metadata.columns.keys.return_value = ['name', 'salary']
        instance._sample_rows.side_effect = [
            (sampled_data, 4),
            (sampled_data, 5),
            (sampled_data, 10),
        ]
        mock_progress_bar = Mock()

        path = tmpdir / 'file.csv'
        # Create an empty csv file
        open(path, mode='a').close()

        # Run
        result = BaseSingleTableSynthesizer._sample_batch(
            instance,
            batch_size=10,
            max_tries=100,
            conditions=None,
            transformed_conditions=None,
            float_rtol=0.01,
            progress_bar=mock_progress_bar,
            output_file_path=path,
        )

        # Assert
        pd.testing.assert_frame_equal(result, sampled_data)
        assert instance._sample_rows.call_count == 3
        expected_stored_data = pd.DataFrame({
            'name': [
                'John',
                'Doe',
                'John Doe',
                'John Doe John',
                'John Doe John',
                'John',
                'Doe',
                'John Doe',
                'John Doe John',
            ],
            'salary': [
                80.,
                60.,
                100.,
                300.,
                300.,
                80.,
                60.,
                100.,
                300.,
            ]
        })
        data = pd.read_csv(path)
        pd.testing.assert_frame_equal(expected_stored_data, data)
        mock_progress_bar.update.call_count == 3

    def test__make_condition_dfs(self):
        """Test that the condition dfs are being created as expected."""
        # Setup
        condition_a = Condition({'name': 'John Doe'})
        condition_b = Condition({'salary': 80.})

        # Run
        result = BaseSingleTableSynthesizer._make_condition_dfs([condition_a, condition_b])

        # Assert
        expected_result = [
            pd.DataFrame({'name': ['John Doe']}),
            pd.DataFrame({'salary': [80.]}),
        ]

        for res, exp in zip(result, expected_result):
            pd.testing.assert_frame_equal(res, exp)

    def test__sample_in_batches(self):
        """Test that this method calls and concatenates the output of ``_sample_batch``."""
        # Setup
        first_data = pd.DataFrame({
            'name': ['John', 'Doe', 'John Doe'],
            'salary': [60., 70., 80.]
        })
        second_data = pd.DataFrame({
            'name': ['Johana', 'Doe', 'Johana Doe'],
            'salary': [65., 75., 85.]
        })
        instance = Mock()
        instance._sample_batch.side_effect = [first_data, second_data]

        # Run
        result = BaseSingleTableSynthesizer._sample_in_batches(
            instance,
            num_rows=6,
            batch_size=3,
            max_tries_per_batch=100,
            conditions='conditions',
            transformed_conditions='transformed_conditions',
            float_rtol=0.02,
            progress_bar='progress_bar',
            output_file_path='output_file_path',
        )

        # Assert
        expected_result = pd.DataFrame({
            'name': ['John', 'Doe', 'John Doe', 'Johana', 'Doe', 'Johana Doe'],
            'salary': [60., 70., 80., 65., 75., 85.]
        })
        pd.testing.assert_frame_equal(result, expected_result)
        assert instance._sample_batch.call_count == 2
        expected_call = call(
            batch_size=3,
            max_tries=100,
            conditions='conditions',
            transformed_conditions='transformed_conditions',
            float_rtol=0.02,
            progress_bar='progress_bar',
            output_file_path='output_file_path',
        )
        assert expected_call == instance._sample_batch.call_args_list[0]
        assert expected_call == instance._sample_batch.call_args_list[1]

    def test__conditionally_sample_rows(self):
        """Test when sampled rows is bigger than 0."""
        # Setup
        transformed_data = pd.DataFrame({
            COND_IDX: [0, 1, 2],
            'salary': [65., 75., 85.]
        })
        data = pd.DataFrame({
            'name': ['Johana', 'Doe', 'Johana Doe'],
            'salary': [65., 75., 85.],
        })
        instance = Mock()
        instance._sample_in_batches.return_value = data
        condition = Mock()
        transformed_condition = Mock()

        # Run
        result = BaseSingleTableSynthesizer._conditionally_sample_rows(
            instance,
            transformed_data,
            condition,
            transformed_condition,
        )

        # Assert
        expected_data = pd.DataFrame({
            'name': ['Johana', 'Doe', 'Johana Doe'],
            'salary': [65., 75., 85.],
            COND_IDX: [0, 1, 2]
        })
        pd.testing.assert_frame_equal(expected_data, result)
        instance._sample_in_batches.assert_called_once_with(
            num_rows=3,
            batch_size=3,
            max_tries_per_batch=None,
            conditions=condition,
            transformed_conditions=transformed_condition,
            float_rtol=0.01,
            progress_bar=None,
            output_file_path=None
        )

    def test__conditionally_sample_rows_no_rows_sampled_error(self):
        """Test that an error is raised when no rows are sampled."""
        # Setup
        transformed_data = pd.DataFrame({
            COND_IDX: [0, 1, 2],
            'salary': [65., 75., 85.]
        })
        instance = Mock()
        instance._sample_in_batches.return_value = []
        condition = Mock()
        transformed_condition = 'fancy_condition'

        # Run and Assert
        expected_message = re.escape(
            'Unable to sample any rows for the given conditions '
            "'fancy_condition'. Try increasing 'max_tries_per_batch' (currently: None) "
            "or increasing 'batch_size' (currently: 3). Note that "
            'increasing these values will also increase the sampling time.'
        )
        with pytest.raises(ValueError, match=expected_message):
            BaseSingleTableSynthesizer._conditionally_sample_rows(
                instance,
                transformed_data,
                condition,
                transformed_condition,
                graceful_reject_sampling=False,
            )

    def test__conditionally_sample_rows_raises_value_error_model_is_gm(self):
        """Test when there are no sampled rows and the model is ``GaussianMultivariate``.

        When no ``graceful_reject_sampling`` is used and the model is ``GaussianMultivariate``
        the sampling raises a ``ValueError`` with a specific message for this model.
        """
        # Setup
        transformed_data = pd.DataFrame({
            COND_IDX: [0, 1, 2],
            'salary': [65., 75., 85.]
        })
        instance = Mock()
        instance._sample_in_batches.return_value = []
        condition = Mock()
        transformed_condition = 'fancy_condition'
        instance._model = GaussianMultivariate()

        # Run and Assert
        expected_message = re.escape(
            'Unable to sample any rows for the given conditions '
            "'fancy_condition'. This may be because the provided values are out-of-bounds in the "
            'current model. \nPlease try again with a different set of values.'
        )
        with pytest.raises(ValueError, match=expected_message):
            BaseSingleTableSynthesizer._conditionally_sample_rows(
                instance,
                transformed_data,
                condition,
                transformed_condition,
                graceful_reject_sampling=False,
            )

    def test__conditionally_sample_rows_with_graceful_reject_sampling(self):
        """Test when no rows are sampled but ``graceful_reject_sampling`` is ``True``."""
        # Setup
        transformed_data = pd.DataFrame({
            COND_IDX: [0, 1, 2],
            'salary': [65., 75., 85.]
        })
        instance = Mock()
        instance._sample_in_batches.return_value = []
        condition = Mock()
        transformed_condition = 'fancy_condition'
        instance._model = GaussianMultivariate()

        # Run
        result = BaseSingleTableSynthesizer._conditionally_sample_rows(
            instance,
            transformed_data,
            condition,
            transformed_condition,
            graceful_reject_sampling=True,
        )

        # Assert
        assert result == []

    def test__sample_with_progress_bar_num_rows_is_none(self):
        """Test that a ``ValueError`` is raised when ``num_rows`` is ``None``."""
        # Setup
        instance = Mock()
        num_rows = None

        # Run and Assert
        expected_message = re.escape(
            'You must specify the number of rows to sample (e.g. num_rows=100).'
        )
        with pytest.raises(ValueError, match=expected_message):
            BaseSingleTableSynthesizer._sample_with_progress_bar(instance, num_rows)

    def test__sample_with_progress_bar_num_rows_is_zero(self):
        """Test that an empty dataframe is returned when ``num_rows`` is 0."""
        # Setup
        instance = Mock()
        num_rows = 0

        # Run
        result = BaseSingleTableSynthesizer._sample_with_progress_bar(instance, num_rows)

        # Assert
        pd.testing.assert_frame_equal(result, pd.DataFrame())

    @patch('sdv.single_table.base.tqdm')
    @patch('sdv.single_table.base.validate_file_path')
    def test__sample_with_progress_bar_returns_sampled_data(self,
                                                            mock_validate_file_path, mock_tqdm):
        """Test that ``_sample_in_batches`` is being called and it's output is returned."""
        # Setup
        progress_bar = MagicMock()
        mock_tqdm.tqdm.return_value = progress_bar
        instance = Mock()
        instance._sample_in_batches.return_value = pd.DataFrame({
            'name': ['John', 'Johanna', 'Doe']
        })

        # Run
        result = BaseSingleTableSynthesizer._sample_with_progress_bar(instance, 10)

        # Assert
        expected_result = pd.DataFrame({'name': ['John', 'Johanna', 'Doe']})
        pd.testing.assert_frame_equal(result, expected_result)
        mock_tqdm.tqdm.assert_called_once_with(total=10, disable=False)
        progress_bar.__enter__.return_value.set_description.assert_called_once_with(
            'Sampling rows'
        )
        instance._sample_in_batches.assert_called_once_with(
            num_rows=10,
            batch_size=10,
            max_tries_per_batch=100,
            progress_bar=progress_bar.__enter__.return_value,
            output_file_path=mock_validate_file_path.return_value,
        )

    @patch('sdv.single_table.base.handle_sampling_error')
    @patch('sdv.single_table.base.tqdm')
    @patch('sdv.single_table.base.validate_file_path')
    def test__sample_with_progress_bar_handle_sampling_error(
        self, mock_validate_file_path, mock_tqdm, mock_handle_sampling_error
    ):
        """Test the error handling when we are using ``_sample_in_batches``."""
        # Setup
        progress_bar = MagicMock()
        mock_tqdm.tqdm.return_value = progress_bar
        instance = Mock()
        keyboard_error = KeyboardInterrupt()
        instance._sample_in_batches.side_effect = [keyboard_error]
        mock_validate_file_path.return_value = 'temp_file'

        # Run
        result = BaseSingleTableSynthesizer._sample_with_progress_bar(instance, 10)

        # Assert
        expected_result = pd.DataFrame()
        pd.testing.assert_frame_equal(result, expected_result)
        mock_tqdm.tqdm.assert_called_once_with(total=10, disable=False)
        progress_bar.__enter__.return_value.set_description.assert_called_once_with(
            'Sampling rows'
        )
        instance._sample_in_batches.assert_called_once_with(
            num_rows=10,
            batch_size=10,
            max_tries_per_batch=100,
            progress_bar=progress_bar.__enter__.return_value,
            output_file_path=mock_validate_file_path.return_value,
        )
        mock_handle_sampling_error.assert_called_once_with(False, 'temp_file', keyboard_error)

    @patch('sdv.single_table.base.os')
    @patch('sdv.single_table.base.tqdm')
    @patch('sdv.single_table.base.validate_file_path')
    def test__sample_with_progress_bar_removing_temp_file(
        self, mock_validate_file_path, mock_tqdm, mock_os
    ):
        """Test that the temporary file is removed after sampling."""
        # Setup
        progress_bar = MagicMock()
        mock_tqdm.tqdm.return_value = progress_bar
        instance = Mock()
        instance._sample_in_batches.return_value = pd.DataFrame()
        mock_validate_file_path.return_value = '.sample.csv.temp'

        # Run
        result = BaseSingleTableSynthesizer._sample_with_progress_bar(instance, 10)

        # Assert
        expected_result = pd.DataFrame()
        pd.testing.assert_frame_equal(result, expected_result)
        mock_tqdm.tqdm.assert_called_once_with(total=10, disable=False)
        progress_bar.__enter__.return_value.set_description.assert_called_once_with(
            'Sampling rows'
        )
        instance._sample_in_batches.assert_called_once_with(
            num_rows=10,
            batch_size=10,
            max_tries_per_batch=100,
            progress_bar=progress_bar.__enter__.return_value,
            output_file_path=mock_validate_file_path.return_value,
        )
        mock_os.remove.assert_called_once_with('.sample.csv.temp')
        mock_os.path.exists.assert_called_once_with('.sample.csv.temp')

    def test_sample(self):
        """Test that we use ``_sample_with_progress_bar`` in this method."""
        # Setup
        num_rows = 10
        max_tries_per_batch = 50
        batch_size = 5
        output_file_path = 'temp.csv'
        instance = Mock()
        instance.get_metadata.return_value._constraints = False

        # Run
        result = BaseSingleTableSynthesizer.sample(
            instance,
            num_rows,
            max_tries_per_batch,
            batch_size,
            output_file_path,
        )

        # Assert
        instance._sample_with_progress_bar.assert_called_once_with(
            10,
            50,
            5,
            'temp.csv',
            show_progress_bar=True
        )
        assert result == instance._sample_with_progress_bar.return_value

    def test__validate_conditions(self):
        """Test that conditions are within the ``data_processor`` fields."""
        # Setup
        instance = Mock()
        instance._data_processor.get_sdtypes.return_value = {
            'name': 'categorical',
            'surname': 'categorical',
        }
        conditions = pd.DataFrame({'name': ['Johanna'], 'surname': ['Doe']})

        # Run
        BaseSingleTableSynthesizer._validate_conditions(instance, conditions)

        # Assert
        instance._data_processor.get_sdtypes.assert_called()

    def test__validate_conditions_raises_error(self):
        """Test that conditions are not in the ``data_processor`` fields."""
        # Setup
        instance = Mock()
        instance._data_processor.get_sdtypes.return_value = {
            'name': 'categorical',
            'surname': 'categorical',
        }
        conditions = pd.DataFrame({'names': ['Johanna'], 'surname': ['Doe']})

        # Run and Assert
        error_msg = re.escape(
            "Unexpected column name 'names'. Use a column name that was present in the "
            'original data.'
        )
        with pytest.raises(ValueError, match=error_msg):
            BaseSingleTableSynthesizer._validate_conditions(instance, conditions)

    def test__sample_with_conditions_constraints_not_met(self):
        """Test when conditions are not met."""
        # Setup
        conditions = pd.DataFrame({
            'name': ['Johanna', 'Doe'],
            'salary': [100., 90.]
        })
        instance = Mock()
        instance._data_processor.transform.side_effect = [ConstraintsNotMetError]

        # Run and Assert
        error_msg = 'Provided conditions are not valid for the given constraints.'
        with pytest.raises(ConstraintsNotMetError, match=error_msg):
            BaseSingleTableSynthesizer._sample_with_conditions(
                instance,
                conditions,
                10,
                10,
            )

    def test__sample_with_conditions_transformed_whith_transformed_data(self):
        """Test when the condition is transformed, this is being used to conditionally sample."""
        # Setup
        conditions = pd.DataFrame({'name': ['Johanna', 'Doe']})
        instance = Mock()
        instance._data_processor.transform.side_effect = [
            pd.DataFrame({'name': [0.25]}),
            pd.DataFrame({'name': [0.90]}),
        ]
        instance._conditionally_sample_rows.side_effect = [
            pd.DataFrame({'name': ['Johanna'], COND_IDX: [1]}),
            pd.DataFrame({'name': ['Doe'], COND_IDX: [0]}),
        ]

        # Run
        result = BaseSingleTableSynthesizer._sample_with_conditions(
            instance,
            conditions,
            10,
            10,
        )

        # Assert
        sample_calls = instance._conditionally_sample_rows.call_args_list
        first_call_kwargs = sample_calls[0][1]
        second_call_kwargs = sample_calls[1][1]
        first_df = first_call_kwargs.pop('dataframe')
        second_df = second_call_kwargs.pop('dataframe')

        pd.testing.assert_frame_equal(result, pd.DataFrame({'name': ['Doe', 'Johanna']}))

        pd.testing.assert_frame_equal(
            first_df,
            pd.DataFrame({'name': [0.25], COND_IDX: [1]}, index=[1])
        )
        pd.testing.assert_frame_equal(
            second_df,
            pd.DataFrame({'name': [0.90], COND_IDX: [0]})
        )
        assert first_call_kwargs == {
            'condition': {
                'name': 'Doe'
            },
            'transformed_condition': {
                'name': 0.25
            },
            'max_tries_per_batch': 10,
            'batch_size': 10,
            'progress_bar': None,
            'output_file_path': None,
        }
        assert second_call_kwargs == {
            'condition': {
                'name': 'Johanna'
            },
            'transformed_condition': {
                'name': 0.90
            },
            'max_tries_per_batch': 10,
            'batch_size': 10,
            'progress_bar': None,
            'output_file_path': None,
        }

    def test__sample_with_conditions_no_transformed_conditions(self):
        """Test when the conditions are not being transformed.

        Test that when the conditions are not transformable, this calls the conditional
        sampling with the expected values and makes the ``transformed_condition`` to be ``None``.
        """
        # Setup
        conditions = pd.DataFrame({'name': ['Johanna']})
        instance = Mock()
        instance._data_processor.transform.side_effect = [
            pd.DataFrame(),
            pd.DataFrame(),
        ]
        instance._conditionally_sample_rows.return_value = pd.DataFrame()

        # Run
        result = BaseSingleTableSynthesizer._sample_with_conditions(
            instance,
            conditions,
            10,
            10,
        )

        # Assert
        pd.testing.assert_frame_equal(result, pd.DataFrame())
        sample_call = instance._conditionally_sample_rows.call_args_list[0][1]
        call_dataframe = sample_call.pop('dataframe')
        pd.testing.assert_frame_equal(
            call_dataframe,
            pd.DataFrame({COND_IDX: [0], 'name': ['Johanna']})
        )
        assert sample_call == {
            'condition': {
                'name': 'Johanna'
            },
            'transformed_condition': None,
            'max_tries_per_batch': 10,
            'batch_size': 10,
            'progress_bar': None,
            'output_file_path': None,
        }

    @patch('sdv.single_table.base.os')
    @patch('sdv.single_table.base.check_num_rows')
    @patch('sdv.single_table.base.DataProcessor')
    @patch('sdv.single_table.base.tqdm')
    @patch('sdv.single_table.base.validate_file_path')
    def test_sample_from_conditions(self, mock_validate_file_path, mock_tqdm,
                                    mock_data_processor, mock_check_num_rows, mock_os):
        """Test sample conditions with sampled data and reject sampling.

        An instance of ``BaseSingleTableSynthesizer`` is created and it's utility functions are
        being mocked to reach the point of calling ```_sample_with_conditions``. After the
        sampling is done, when the ``temp_file`` is the ``.sample.csv.temp``, this is being
        removed.
        """
        # Setup
        metadata = Mock()
        instance = BaseSingleTableSynthesizer(metadata)
        conditions = [Condition({'name': 'John Doe'})]
        mock_validate_file_path.return_value = '.sample.csv.temp'

        instance._validate_conditions = Mock()
        instance._sample_with_conditions = Mock()
        instance._model = GaussianMultivariate()
        instance._sample_with_conditions.return_value = pd.DataFrame({'name': ['John Doe']})

        progress_bar = MagicMock()
        mock_tqdm.tqdm.return_value = progress_bar

        # Run
        result = instance.sample_from_conditions(conditions, 10, 10, '.sample.csv.temp')

        # Assert
        pd.testing.assert_frame_equal(result, pd.DataFrame({'name': ['John Doe']}))
        mock_os.remove.assert_called_once_with('.sample.csv.temp')
        mock_os.path.exists.assert_called_once_with('.sample.csv.temp')

    @patch('sdv.single_table.base.handle_sampling_error')
    @patch('sdv.single_table.base.tqdm')
    @patch('sdv.single_table.base.validate_file_path')
    def test_sample_from_conditions_handle_sampling_error(self,
                                                          mock_validate_file_path,
                                                          mock_tqdm, mock_handle_sampling_error):
        """Test the error handling when we are using ``sample_from_conditions``."""
        # Setup
        progress_bar = MagicMock()
        mock_tqdm.tqdm.return_value = progress_bar
        instance = Mock()
        instance._make_condition_dfs.side_effect = lambda x: x
        conditions = [Condition({'name': 'John Doe'})]
        keyboard_error = KeyboardInterrupt()
        instance._sample_with_conditions.side_effect = [keyboard_error]
        mock_validate_file_path.return_value = 'temp_file'

        # Run
        result = BaseSingleTableSynthesizer.sample_from_conditions(
            instance,
            conditions,
            10,
            10,
            '.sample.csv.temp'
        )

        # Assert
        expected_result = pd.DataFrame()
        pd.testing.assert_frame_equal(result, expected_result)
        mock_tqdm.tqdm.assert_called_once_with(total=1)
        progress_bar.__enter__.return_value.set_description.assert_called_once_with(
            'Sampling conditions'
        )
        mock_handle_sampling_error.assert_called_once_with(False, 'temp_file', keyboard_error)

    @patch('sdv.single_table.base.os')
    @patch('sdv.single_table.base.check_num_rows')
    @patch('sdv.single_table.base.DataProcessor')
    @patch('sdv.single_table.base.tqdm')
    @patch('sdv.single_table.base.validate_file_path')
    def test_sample_remaining_columns(self, mock_validate_file_path, mock_tqdm,
                                      mock_data_processor, mock_check_num_rows, mock_os):
        """Test the this method calls ``_sample_with_conditions`` with the ``known_column."""
        # Setup
        metadata = Mock()
        instance = BaseSingleTableSynthesizer(metadata)
        known_columns = pd.DataFrame({'name': ['Johanna Doe']})

        instance._validate_conditions = Mock()
        instance._sample_with_conditions = Mock()
        instance._model = GaussianMultivariate()
        instance._sample_with_conditions.return_value = pd.DataFrame({'name': ['John Doe']})
        mock_validate_file_path.return_value = '.sample.csv.temp'

        progress_bar = MagicMock()
        mock_tqdm.tqdm.return_value = progress_bar

        # Run
        result = instance.sample_remaining_columns(
            known_columns,
            10,
            10,
            '.sample.csv.temp'
        )

        # Assert
        pd.testing.assert_frame_equal(result, pd.DataFrame({'name': ['John Doe']}))
        mock_os.remove.assert_called_once_with('.sample.csv.temp')
        mock_os.path.exists.assert_called_once_with('.sample.csv.temp')

    @patch('sdv.single_table.base.handle_sampling_error')
    @patch('sdv.single_table.base.check_num_rows')
    @patch('sdv.single_table.base.DataProcessor')
    @patch('sdv.single_table.base.tqdm')
    @patch('sdv.single_table.base.validate_file_path')
    def test_sample_remaining_columns_handles_sampling_error(
        self, mock_validate_file_path, mock_tqdm, mock_data_processor,
        mock_check_num_rows, mock_handle_sampling_error
    ):
        """Test when sample remaining is being interrupted.

        This should properly handle the errors with the ``handle_sampling_error`` function.
        """
        # Setup
        metadata = Mock()
        instance = BaseSingleTableSynthesizer(metadata)
        known_columns = pd.DataFrame({'name': ['Johanna Doe']})

        instance._validate_conditions = Mock()
        instance._sample_with_conditions = Mock()
        instance._model = GaussianMultivariate()
        keyboard_error = KeyboardInterrupt()
        instance._sample_with_conditions.side_effect = [keyboard_error]
        mock_validate_file_path.return_value = 'temp_file'

        progress_bar = MagicMock()
        mock_tqdm.tqdm.return_value = progress_bar

        # Run
        result = instance.sample_remaining_columns(
            known_columns,
            10,
            10,
            'temp_file'
        )

        # Assert
        pd.testing.assert_frame_equal(result, pd.DataFrame())
        mock_handle_sampling_error.assert_called_once_with(False, 'temp_file', keyboard_error)

    @patch('sdv.single_table.base.cloudpickle')
    def test_save(self, cloudpickle_mock):
        """Test that the synthesizer is saved correctly."""
        # Setup
        synthesizer = Mock()

        # Run
        BaseSingleTableSynthesizer.save(synthesizer, 'output.pkl')

        # Assert
        cloudpickle_mock.dump.assert_called_once_with(synthesizer, ANY)

    @patch('sdv.single_table.base.cloudpickle')
    @patch('builtins.open', new_callable=mock_open)
    def test_load(self, mock_file, cloudpickle_mock):
        """Test that the ``load`` method loads a stored synthesizer."""
        # Setup
        synthesizer_mock = Mock()
        cloudpickle_mock.load.return_value = synthesizer_mock

        # Run
        loaded_instance = BaseSingleTableSynthesizer.load('synth.pkl')

        # Assert
        mock_file.assert_called_once_with('synth.pkl', 'rb')
        cloudpickle_mock.load.assert_called_once_with(mock_file.return_value)
        assert loaded_instance == synthesizer_mock

    def test_load_custom_constraint_classes(self):
        """Test that ``load_custom_constraint_classes`` calls the ``DataProcessor``'s method."""
        # Setup
        instance = Mock()

        # Run
        BaseSingleTableSynthesizer.load_custom_constraint_classes(
            instance,
            'path/to/file.py',
            ['Custom', 'Constr', 'UpperPlus']
        )

        # Assert
        instance._data_processor.load_custom_constraint_classes.assert_called_once_with(
            'path/to/file.py',
            ['Custom', 'Constr', 'UpperPlus']
        )

    def test_add_custom_constraint_class(self):
        """Test that this method calls the ``DataProcessor``'s method."""
        # Setup
        instance = Mock()
        constraint_mock = Mock()

        # Run
        BaseSingleTableSynthesizer.add_custom_constraint_class(
            instance,
            constraint_mock,
            'custom'
        )

        # Assert
        instance._data_processor.add_custom_constraint_class.assert_called_once_with(
            constraint_mock,
            'custom'
        )

    def test_add_constraint_warning(self):
        """Test a warning is raised when the synthesizer had already been fitted."""
        # Setup
        metadata = SingleTableMetadata()
        instance = BaseSingleTableSynthesizer(metadata)
        instance._fitted = True

        # Run and Assert
        warn_msg = (
            "For these constraints to take effect, please refit the synthesizer using 'fit'."
        )
        with pytest.warns(UserWarning, match=warn_msg):
            instance.add_constraints([])

    def test_add_constraints(self):
        """Test a list of constraints can be added to the synthesizer."""
        # Setup
        metadata = SingleTableMetadata()
        metadata.add_column('col', sdtype='numerical')
        instance = BaseSingleTableSynthesizer(metadata)
        positive_constraint = {
            'constraint_class': 'Positive',
            'constraint_parameters': {
                'column_name': 'col',
                'strict_boundaries': True
            }
        }
        negative_constraint = {
            'constraint_class': 'Negative',
            'constraint_parameters': {
                'column_name': 'col',
                'strict_boundaries': False
            }
        }

        # Run
        instance.add_constraints([positive_constraint, negative_constraint])

        # Assert
        positive_constraint = {
            'constraint_class': 'Positive',
            'constraint_parameters': {
                'column_name': 'col',
                'strict_boundaries': True
            }
        }
        negative_constraint = {
            'constraint_class': 'Negative',
            'constraint_parameters': {
                'column_name': 'col',
                'strict_boundaries': False
            }
        }
        assert instance.get_constraints() == [positive_constraint, negative_constraint]

    def test_get_constraints(self):
        """Test a list of constraints is returned by the method."""
        # Setup
        metadata = SingleTableMetadata()
        metadata.add_column('col', sdtype='numerical')
        instance = BaseSingleTableSynthesizer(metadata)
        positive_constraint = {
            'constraint_class': 'Positive',
            'constraint_parameters': {
                'column_name': 'col',
                'strict_boundaries': True
            }
        }
        negative_constraint = {
            'constraint_class': 'Negative',
            'constraint_parameters': {
                'column_name': 'col',
                'strict_boundaries': False
            }
        }
        constraints = [positive_constraint, negative_constraint]
        instance.add_constraints(constraints)

        # Run
        output = instance.get_constraints()

        # Assert
        assert output == constraints

    def _pkg_mock(self, lib):
        if lib == 'sdv':
            class Distribution:
                version = '1.0.0'

            return Distribution

    @patch('pkg_resources.get_distribution')
    def test_get_info(self, pkg_mock):
        """Test the correct dictionary is returned.

        Check the return dictionary is valid both before and after fitting the synthesizer.

        Mocks:
            * Mock ``pkg_resources`` so we don't have to rewrite this test for every new release.
            * Unfortunately, ``datetime`` can't be mocked directly. This link explains how to
            do it: https://docs.python.org/3/library/unittest.mock-examples.html#partial-mocking
        """
        # Setup
        data = pd.DataFrame({'col': [1, 2, 3]})
        pkg_mock.side_effect = self._pkg_mock
        metadata = SingleTableMetadata()
        metadata.add_column('col', sdtype='numerical')

        with patch('sdv.single_table.base.datetime.datetime') as mock_date:
            mock_date.today.return_value = datetime(2023, 1, 23)
            mock_date.side_effect = lambda *args, **kw: date(*args, **kw)
            synthesizer = GaussianCopulaSynthesizer(metadata)

            # Run
            info = synthesizer.get_info()

            # Assert
            assert info == {
                'class_name': 'GaussianCopulaSynthesizer',
                'creation_date': '2023-01-23',
                'is_fit': False,
                'last_fit_date': None,
                'fitted_sdv_version': None
            }

            # Run
            synthesizer.fit(data)
            info = synthesizer.get_info()

            # Assert
            assert info == {
                'class_name': 'GaussianCopulaSynthesizer',
                'creation_date': '2023-01-23',
                'is_fit': True,
                'last_fit_date': '2023-01-23',
                'fitted_sdv_version': '1.0.0'
            }
