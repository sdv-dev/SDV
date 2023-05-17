"""Test Single Table Metadata."""

import json
import re
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from sdv.metadata.errors import InvalidMetadataError
from sdv.metadata.single_table import SingleTableMetadata


class TestSingleTableMetadata:
    """Test ``SingleTableMetadata`` class."""

    VALID_KWARGS = [
        ('age', 'numerical', {}),
        ('age', 'numerical', {'computer_representation': 'Int8'}),
        ('start_date', 'datetime', {}),
        ('start_date', 'datetime', {'datetime_format': '%Y-%d'}),
        ('name', 'categorical', {}),
        ('name', 'categorical', {'order_by': 'alphabetical'}),
        ('name', 'categorical', {'order': ['a', 'b', 'c']}),
        ('synthetic', 'boolean', {}),
        ('phrase', 'id', {}),
        ('phrase', 'id', {'regex_format': '[A-z]'}),
        ('phone', 'phone_number', {}),
        ('phone', 'phone_number', {'pii': True}),
    ]

    INVALID_KWARGS = [
        (
            'age',
            'numerical',
            {
                'computer_representation': 'Int8',
                'datetime_format': None,
                'pii': True
            },
            re.escape("Invalid values '(datetime_format, pii)' for numerical column 'age'."),
        ),
        (
            'start_date', 'datetime', {'datetime_format': '%Y-%d', 'pii': True},
            re.escape("Invalid values '(pii)' for datetime column 'start_date'.")
        ),
        (
            'name', 'categorical',
            {'pii': True, 'ordering': ['a', 'b'], 'ordered': 'numerical_values'},
            re.escape("Invalid values '(ordered, ordering, pii)' for categorical column 'name'.")
        ),
        (
            'synthetic', 'boolean', {'pii': True},
            re.escape("Invalid values '(pii)' for boolean column 'synthetic'.")
        ),
        (
            'phrase', 'id', {'regex_format': '[A-z]', 'pii': True, 'anonymization': True},
            re.escape("Invalid values '(anonymization, pii)' for id column 'phrase'.")
        ),
        (
            'phone', 'phone_number', {'anonymization': True, 'order_by': 'phone_number'},
            re.escape(
                "Invalid values '(anonymization, order_by)' for phone_number column 'phone'."
            )
        )
    ]  # noqa: JS102

    def test___init__(self):
        """Test creating an instance of ``SingleTableMetadata``."""
        # Run
        instance = SingleTableMetadata()

        # Assert
        assert instance.columns == {}
        assert instance.primary_key is None
        assert instance.sequence_key is None
        assert instance.alternate_keys == []
        assert instance.sequence_index is None
        assert instance._version == 'SINGLE_TABLE_V1'

    def test__validate_numerical_default_and_invalid(self):
        """Test the ``_validate_numerical`` method.

        Setup:
            - instance of ``SingleTableMetadata``
            - list of accepted computer representations.

        Input:
            - Column name.
            - sdtype numerical
            - computer representation

        Side Effects:
            - Passes when no ``computer_representation`` is provided
            - ``InvalidMetadataError`` is raised stating that the ``computer_representation`` is
              not supported.
        """
        # Setup
        instance = SingleTableMetadata()

        # Run / Assert
        instance._validate_numerical('age')

        error_msg = re.escape("Invalid value for 'computer_representation' '36' for column 'age'.")
        with pytest.raises(InvalidMetadataError, match=error_msg):
            instance._validate_numerical('age', computer_representation=36)

    @pytest.mark.parametrize(
        'computer_representation', SingleTableMetadata._NUMERICAL_REPRESENTATIONS)
    def test__validate_numerical_computer_representations(self, computer_representation):
        """Test the ``_validate_numerical`` method.

        Setup:
            - instance of ``SingleTableMetadata``
            - list of accepted computer representations.

        Input:
            - Column name.
            - sdtype numerical
            - computer representation

        Side Effects:
            - Passes with the correct ``computer_representation``
            - ``InvalidMetadataError`` is raised stating that the ``computer_representation`` is
              wrong.
        """
        # Setup
        instance = SingleTableMetadata()

        # Run / Assert
        instance._validate_numerical('age', computer_representation=computer_representation)

    def test__validate_datetime(self):
        """Test the ``_validate_datetime`` method.

        Setup:
            - instance of ``SingleTableMetadata``

        Input:
            - Column name.
            - sdtype datetime
            - Valid ``datetime_format``.
            - Invalid ``datetime_format``.

        Side Effects:
            - ``InvalidMetadataError`` indicating the format ``%`` that has not been formatted.
        """
        # Setup
        instance = SingleTableMetadata()

        # Run / Assert
        instance._validate_datetime('start_date', datetime_format='%Y-%m-%d')
        instance._validate_datetime('start_date', datetime_format='%Y-%m-%d - Synthetic')

        error_msg = re.escape(
            "Invalid datetime format string '%1-%Y-%m-%d-%' for datetime column 'start_date'.")
        with pytest.raises(InvalidMetadataError, match=error_msg):
            instance._validate_datetime('start_date', datetime_format='%1-%Y-%m-%d-%')

    def test__validate_categorical(self):
        """Test the ``_validate_categorical`` method.

        Setup:
            - instance of ``SingleTableMetadata``

        Input:
            - Column name.
            - sdtype categorical.
            - A valid ``order_by``.
            - A valid ``order``.
            - An invalid ``order_by`` and ``order``.

        Side Effects:
            - ``InvalidMetadataError`` when both ``order`` and ``order_by`` are present.
            - ``InvalidMetadataError`` when ``order`` is an empty list or a random string.
            - ``InvalidMetadataError`` when ``order_by`` is not ``numerical_value`` or
              ``alphabetical``.
        """
        # Setup
        instance = SingleTableMetadata()

        # Run / Assert
        instance._validate_categorical('name')
        instance._validate_categorical('name', order_by='alphabetical')
        instance._validate_categorical('name', order_by='numerical_value')
        instance._validate_categorical('name', order=['a', 'b', 'c'])

        error_msg = re.escape(
            "Categorical column 'name' has both an 'order' and 'order_by' "
            'attribute. Only 1 is allowed.'
        )
        with pytest.raises(InvalidMetadataError, match=error_msg):
            instance._validate_categorical('name', order_by='alphabetical', order=['a', 'b', 'c'])

        error_msg_order_by = re.escape(
            "Unknown ordering method 'my_ordering' provided for categorical column "
            "'name'. Ordering method must be 'numerical_value' or 'alphabetical'."
        )
        with pytest.raises(InvalidMetadataError, match=error_msg_order_by):
            instance._validate_categorical('name', order_by='my_ordering')

        error_msg_order = re.escape(
            "Invalid order value provided for categorical column 'name'. "
            "The 'order' must be a list with 1 or more elements."
        )
        with pytest.raises(InvalidMetadataError, match=error_msg_order):
            instance._validate_categorical('name', order='my_ordering')

        with pytest.raises(InvalidMetadataError, match=error_msg_order):
            instance._validate_categorical('name', order=[])

    def test__validate_id(self):
        """Test the ``_validate_id`` method.

        Setup:
            - instance of ``SingleTableMetadata``

        Input:
            - Column name.
            - sdtype id
            - Valid ``regex_format``.
            - Invalid ``regex_format``.

        Side Effects:
            - ``InvalidMetadataError``
        """
        # Setup
        instance = SingleTableMetadata()

        # Run / Assert
        instance._validate_id('phrase', regex_format='[A-z]')
        error_msg = re.escape("Invalid regex format string '[A-z{' for id column 'phrase'.")
        with pytest.raises(InvalidMetadataError, match=error_msg):
            instance._validate_id('phrase', regex_format='[A-z{')

    def test__validate_column_exists(self):
        """Test the ``_validate_column_exists`` method.

        Setup:
            - instance of ``SingleTableMetadata``
            - A list of ``_columns``.

        Input:
            - Column name.

        Side Effects:
            - ``InvalidMetadataError`` when the column is not in the ``instance.columns``.
        """
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {
            'name': {'sdtype': 'categorical'},
            'age': {'sdtype': 'numerical'},
            'start_date': {'sdtype': 'datetime'},
            'phrase': {'sdtype': 'id'},
        }

        # Run / Assert
        instance._validate_column_exists('age')
        error_msg = re.escape(
            "Column name ('synthetic') does not exist in the table. "
            "Use 'add_column' to add new column."
        )
        with pytest.raises(InvalidMetadataError, match=error_msg):
            instance._validate_column_exists('synthetic')

    @pytest.mark.parametrize(('column_name', 'sdtype', 'kwargs'), VALID_KWARGS)
    def test__validate_unexpected_kwargs_valid(self, column_name, sdtype, kwargs):
        """Test the ``_validate_unexpected_kwargs`` method.

        Setup:
            - instance of ``SingleTableMetadata``

        Input:
            - Column name.
            - sdtype
            - valid kwargs
        """
        # Setup
        instance = SingleTableMetadata()

        # Run / Assert
        instance._validate_unexpected_kwargs(column_name, sdtype, **kwargs)

    @pytest.mark.parametrize(('column_name', 'sdtype', 'kwargs', 'error_msg'), INVALID_KWARGS)
    def test__validate_unexpected_kwargs_invalid(self, column_name, sdtype, kwargs, error_msg):
        """Test the ``_validate_unexpected_kwargs`` method.

        Setup:
            - instance of ``SingleTableMetadata``

        Input:
            - Column name.
            - sdtype
            - unexpected kwargs

        Side Effects:
            - ``InvalidMetadataError`` is being raised for each sdtype.
        """
        # Setup
        instance = SingleTableMetadata()

        # Run / Assert
        with pytest.raises(InvalidMetadataError, match=error_msg):
            instance._validate_unexpected_kwargs(column_name, sdtype, **kwargs)

    def test__validate_column_invalid_sdtype(self):
        """Test the method with an invalid sdtype.

        If the sdtype isn't one of the supported types, anonymized types or Faker functions,
        then an error should be raised.
        """
        # Setup
        instance = SingleTableMetadata()

        # Run and Assert
        error_msg = re.escape(
            "Invalid sdtype: 'fake_type' is not recognized. Please use one of the "
            'supported SDV sdtypes.'
        )
        with pytest.raises(InvalidMetadataError, match=error_msg):
            instance._validate_column('column', 'fake_type')

        error_msg = re.escape(
            'Invalid sdtype: None is not a string. Please use one of the '
            'supported SDV sdtypes.'
        )
        with pytest.raises(InvalidMetadataError, match=error_msg):
            instance._validate_column('column', None)

    @patch('sdv.metadata.single_table.SingleTableMetadata._validate_unexpected_kwargs')
    @patch('sdv.metadata.single_table.SingleTableMetadata._validate_numerical')
    def test__validate_column_numerical(self, mock__validate_numerical, mock__validate_kwargs):
        """Test ``_validate_column`` method.

        Test the ``_validate_column`` method when a ``numerical`` sdtype is passed.

        Setup:
            - Instance of ``SingleTableMetadata``.

        Input:
            - ``column_name`` - a string.
            - ``sdtype`` - a string 'numerical'.
            - kwargs - any additional key word arguments.

        Mock:
            - ``_validate_unexpected_kwargs`` function from ``SingleTableMetadata``.
            - ``_validate_numerical`` function from ``SingleTableMetadata``.

        Side effects:
            - ``_validate_numerical`` has been called once.
        """
        # Setup
        instance = SingleTableMetadata()

        # Run
        instance._validate_column('age', 'numerical', computer_representation='Int8')

        # Assert
        mock__validate_kwargs.assert_called_once_with(
            'age', 'numerical', computer_representation='Int8')
        mock__validate_numerical.assert_called_once_with('age', computer_representation='Int8')

    @patch('sdv.metadata.single_table.SingleTableMetadata._validate_unexpected_kwargs')
    @patch('sdv.metadata.single_table.SingleTableMetadata._validate_categorical')
    def test__validate_column_categorical(self, mock__validate_categorical, mock__validate_kwargs):
        """Test ``_validate_column`` method.

        Test the ``_validate_column`` method when a ``categorical`` sdtype is passed.

        Setup:
            - Instance of ``SingleTableMetadata``.

        Input:
            - ``column_name`` - a string.
            - ``sdtype`` - a string 'categorical'.
            - kwargs - any additional key word arguments.

        Mock:
            - ``_validate_unexpected_kwargs``
            - ``_validate_categorical`` function from ``SingleTableMetadata``.

        Side effects:
            - ``_validate_categorical`` has been called once.
        """
        # Setup
        instance = SingleTableMetadata()

        # Run
        instance._validate_column('name', 'categorical', order=['a', 'b', 'c'])

        # Assert
        mock__validate_kwargs.assert_called_once_with(
            'name', 'categorical', order=['a', 'b', 'c'])
        mock__validate_categorical.assert_called_once_with('name', order=['a', 'b', 'c'])

    @patch('sdv.metadata.single_table.SingleTableMetadata._validate_unexpected_kwargs')
    def test__validate_column_boolean(self, mock__validate_kwargs):
        """Test ``_validate_column`` method.

        Test the ``_validate_column`` method when a ``boolean`` sdtype is passed.

        Setup:
            - Instance of ``SingleTableMetadata``.

        Input:
            - ``column_name`` - a string.
            - ``sdtype`` - a string 'boolean'.
            - kwargs - any additional key word arguments.

        Mock:
            - ``_validate_unexpected_kwargs``

        Side effects:
        """
        # Setup
        instance = SingleTableMetadata()

        # Run
        instance._validate_column('snythetic', 'boolean')

        # Assert
        mock__validate_kwargs.assert_called_once_with('snythetic', 'boolean')

    @patch('sdv.metadata.single_table.SingleTableMetadata._validate_unexpected_kwargs')
    @patch('sdv.metadata.single_table.SingleTableMetadata._validate_datetime')
    def test__validate_column_datetime(self, mock__validate_datetime, mock__validate_kwargs):
        """Test ``_validate_column`` method.

        Test the ``_validate_column`` method when a ``datetime`` sdtype is passed.

        Setup:
            - Instance of ``SingleTableMetadata``.

        Input:
            - ``column_name`` - a string.
            - ``sdtype`` - a string 'datetime'.
            - kwargs - any additional key word arguments.

        Mock:
            - ``_validate_unexpected_kwargs``
            - ``_validate_datetime`` function from ``SingleTableMetadata``.

        Side effects:
            - ``_validate_datetime`` has been called once.
        """
        # Setup
        instance = SingleTableMetadata()

        # Run
        instance._validate_column('start', 'datetime')

        # Assert
        mock__validate_kwargs.assert_called_once_with('start', 'datetime')
        mock__validate_datetime.assert_called_once_with('start')

    @patch('sdv.metadata.single_table.SingleTableMetadata._validate_unexpected_kwargs')
    @patch('sdv.metadata.single_table.SingleTableMetadata._validate_id')
    def test__validate_column_id(self, mock__validate_id, mock__validate_kwargs):
        """Test ``_validate_column`` method.

        Test the ``_validate_column`` method when a ``id`` sdtype is passed.

        Setup:
            - Instance of ``SingleTableMetadata``.

        Input:
            - ``column_name`` - a string.
            - ``sdtype`` - a string 'id'.
            - kwargs - any additional key word arguments.

        Mock:
            - ``_validate_unexpected_kwargs``
            - ``_validate_id`` function from ``SingleTableMetadata``.

        Side effects:
            - ``_validate_id`` has been called once.
        """
        # Setup
        instance = SingleTableMetadata()

        # Run
        instance._validate_column('phrase', 'id', regex_format='[A-z0-9]', pii=True)

        # Assert
        mock__validate_kwargs.assert_called_once_with(
            'phrase', 'id', regex_format='[A-z0-9]', pii=True)
        mock__validate_id.assert_called_once_with('phrase', regex_format='[A-z0-9]', pii=True)

    @patch('sdv.metadata.single_table.SingleTableMetadata._validate_unexpected_kwargs')
    def test__validate_pii_not_true_or_false(self, mock__validate_kwargs):
        """Test ``_validate_column`` method when ``pii`` is not ``True``or ``False``."""
        # Run and Assert
        error_msg = re.escape(
            "Parameter 'pii' is set to an invalid attribute ('some_text') for column 'address'. "
            'Expected a value of True or False.'
        )
        with pytest.raises(InvalidMetadataError, match=error_msg):
            SingleTableMetadata._validate_pii('address', pii='some_text')

    @patch('sdv.metadata.single_table.SingleTableMetadata._validate_unexpected_kwargs')
    def test__validate_pii(self, mock__validate_kwargs):
        """Test ``_validate_column`` method when ``pii`` is ``True``or ``False``."""
        # Run and Assert
        SingleTableMetadata._validate_pii('address', pii=True)

    def test_update_column_sdtype(self):
        """Test that ``update_column`` updates the sdtype and keyword args for the given column."""
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {'a': {'sdtype': 'numerical'}}

        # Run
        instance.update_column('a', sdtype='categorical', order_by='alphabetical')

        # Assert
        assert instance.columns == {'a': {'sdtype': 'categorical', 'order_by': 'alphabetical'}}

    def test_update_column_add_extra_value(self):
        """Test that ``update_column`` updates only the keyword args for the given column."""
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {'a': {'sdtype': 'numerical'}}

        # Run
        instance.update_column('a', computer_representation='Int64')

        # Assert
        assert instance.columns == {
            'a': {
                'sdtype': 'numerical',
                'computer_representation': 'Int64'
            }
        }

    def test_add_column_column_name_in_columns(self):
        """Test ``add_column`` method.

        Test that when calling ``add_column`` with a column that is already in
        ``instance.columns`` raises an ``InvalidMetadataError`` stating to use the
        ``update_column`` instead.

        Setup:
            - Instance of ``SingleTableMetadata``.
            - ``_columns`` with some values.

        Input:
            - A column name that is already in ``instance.columns``.

        Side Effects:
            - ``InvalidMetadataError`` is being raised stating that the column exists.
        """
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {'age': {'sdtype': 'numerical'}}

        # Run / Assert
        error_msg = re.escape(
            "Column name 'age' already exists. Use 'update_column' to update an existing column.")
        with pytest.raises(InvalidMetadataError, match=error_msg):
            instance.add_column('age')

    def test_add_column_sdtype_not_in_kwargs(self):
        """Test ``add_column`` method.

        Test that when calling ``add_column`` without an sdtype an ``InvalidMetadataError`` stating
        that it must be provided is raised.

        Setup:
            - Instance of ``SingleTableMetadata``.

        Input:
            - A column name.

        Side Effects:
            - ``InvalidMetadataError`` is being raised stating that sdtype must be provided.
        """
        # Setup
        instance = SingleTableMetadata()

        # Run / Assert
        error_msg = re.escape("Please provide a 'sdtype' for column 'synthetic'.")
        with pytest.raises(InvalidMetadataError, match=error_msg):
            instance.add_column('synthetic')

    def test_add_column_invalid_sdtype(self):
        """Test the method with an invalid sdtype.

        If the sdtype isn't one of the supported types, anonymized types or Faker functions,
        then an error should be raised.
        """
        # Setup
        instance = SingleTableMetadata()

        # Run and Assert
        error_msg = re.escape(
            "Invalid sdtype: 'fake_type' is not recognized. Please use one of the "
            'supported SDV sdtypes.'
        )
        with pytest.raises(InvalidMetadataError, match=error_msg):
            instance.add_column('column', sdtype='fake_type')

    def test_add_column(self):
        """Test ``add_column`` method.

        Test that when calling ``add_column`` method with a ``sdtype`` and the proper ``kwargs``
        this is being added to the ``instance.columns``.

        Setup:
            - Instance of ``SingleTableMetadata``.

        Input:
            - A column name.
            - An ``sdtype``.

        Side Effects:
            - ``instance.columns[column_name]`` now exists.
        """
        # Setup
        instance = SingleTableMetadata()

        # Run
        instance.add_column('age', sdtype='numerical', computer_representation='Int8')

        # Assert
        assert instance.columns['age'] == {
            'sdtype': 'numerical',
            'computer_representation': 'Int8'
        }

    def test_add_column_other_sdtype(self):
        """Test ``add_column`` with an ``sdtype`` that isn't in our base ``sdtypes``..

        If the column is an ``sdtype`` outside of our base ones, it should have the ``pii``
        attribute set to True.

        Run:
            Pass a column with an ``sdtype`` of ``phone_number``.
        """
        # Setup
        instance = SingleTableMetadata()

        # Run
        instance.add_column('number', sdtype='phone_number')

        # Assert
        assert instance.columns['number'] == {'sdtype': 'phone_number', 'pii': True}

    @patch('sdv.metadata.single_table.SingleTableMetadata._validate_column')
    @patch('sdv.metadata.single_table.SingleTableMetadata._validate_column_exists')
    def test_upate_column_sdtype_in_kwargs(self,
                                           mock__validate_column_exists, mock__validate_column):
        """Test the ``update_column`` method.

        Test that when calling ``update_column`` with an ``sdtype`` this is being updated as well
        as any additional.

        Setup:
            - Instance of ``SingleTableMetadata``.
            - A column already in ``_columns``.

        Mock:
            - ``_validate_column_exists``.
            - ``_validate_column``.

        Side Effects:
            - The column has been updated with the new ``sdtype`` and ``kwargs``.
        """
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {'age': {'sdtype': 'numerical'}}

        # Run
        instance.update_column('age', sdtype='categorical', order_by='numerical_value')

        # Assert
        assert instance.columns['age'] == {
            'sdtype': 'categorical',
            'order_by': 'numerical_value'
        }
        mock__validate_column_exists.assert_called_once_with('age')
        mock__validate_column.assert_called_once_with(
            'age', 'categorical', order_by='numerical_value')

    @patch('sdv.metadata.single_table.SingleTableMetadata._validate_column')
    @patch('sdv.metadata.single_table.SingleTableMetadata._validate_column_exists')
    def test_upate_column_no_sdtype(self, mock__validate_column_exists, mock__validate_column):
        """Test the ``update_column`` method.

        Test that when calling ``update_column`` without an ``sdtype`` is updating the other
        ``kwargs``.

        Setup:
            - Instance of ``SingleTableMetadata``.
            - A column already in ``_columns``.

        Mock:
            - ``_validate_column_exists``.
            - ``_validate_column``.

        Side Effects:
            - The column has been updated with the new ``kwargs``.
        """
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {'age': {'sdtype': 'numerical'}}

        # Run
        instance.update_column('age', computer_representation='Float')

        # Assert
        assert instance.columns['age'] == {
            'sdtype': 'numerical',
            'computer_representation': 'Float'
        }
        mock__validate_column_exists.assert_called_once_with('age')
        mock__validate_column.assert_called_once_with(
            'age', 'numerical', computer_representation='Float')

    def test_detect_from_dataframe_raises_error(self):
        """Test the ``detect_from_dataframe`` method.

        Test that if there are existing columns in the metadata, this raises an
        ``InvalidMetadataError``.

        Setup:
            - instance of ``SingleTableMetadata``.
            - Add some value to ``instance.columns``.

        Side Effects:
            Raises an ``InvalidMetadataError`` stating that ``metadata`` already exists.
        """
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {'column': {'sdtype': 'categorical'}}

        # Run / Assert
        err_msg = (
            'Metadata already exists. Create a new ``SingleTableMetadata`` '
            'object to detect from other data sources.'
        )

        with pytest.raises(InvalidMetadataError, match=err_msg):
            instance.detect_from_dataframe('dataframe')

    @patch('sdv.metadata.single_table.LOGGER')
    def test_detect_from_dataframe(self, mock_log):
        """Test the ``dectect_from_dataframe`` method.

        Test that when given a ``pandas.DataFrame``, the current instance of
        ``SingleTableMetadata`` is being updated with the ``sdtypes`` of each
        column in the ``pandas.DataFrame``.

        Setup:
            - Instance of ``SingleTableMetadata``.

        Input:
            - ``pandas.DataFrame`` with multiple data types.

        Side Effects:
            - ``instance.columns`` has been updated with the expected ``sdtypes``.
            - A message is being printed.
        """
        # Setup
        instance = SingleTableMetadata()
        data = pd.DataFrame({
            'categorical': ['cat', 'dog', 'tiger', np.nan],
            'date': pd.to_datetime(['2021-02-02', np.nan, '2021-03-05', '2022-12-09']),
            'int': [1, 2, 3, 4],
            'float': [1., 2., 3., 4],
            'bool': [np.nan, True, False, True]
        })

        # Run
        instance.detect_from_dataframe(data)

        # Assert
        assert instance.columns == {
            'categorical': {'sdtype': 'categorical'},
            'date': {'sdtype': 'datetime'},
            'int': {'sdtype': 'numerical'},
            'float': {'sdtype': 'numerical'},
            'bool': {'sdtype': 'boolean'}
        }

        expected_log_calls = [
            call('Detected metadata:'),
            call(json.dumps(instance.to_dict(), indent=4))
        ]
        mock_log.info.assert_has_calls(expected_log_calls)

    def test_detect_from_csv_raises_error(self):
        """Test the ``detect_from_csv`` method.

        Test that if there are existing columns in the metadata, this raises an
        ``InvalidMetadataError``.

        Setup:
            - instance of ``SingleTableMetadata``.
            - Add some value to ``instance.columns``.

        Side Effects:
            Raises an ``InvalidMetadataError`` stating that ``metadata`` already exists.
        """
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {'column': {'sdtype': 'categorical'}}

        # Run / Assert
        err_msg = (
            'Metadata already exists. Create a new ``SingleTableMetadata`` '
            'object to detect from other data sources.'
        )

        with pytest.raises(InvalidMetadataError, match=err_msg):
            instance.detect_from_csv('filepath')

    @patch('sdv.metadata.single_table.LOGGER')
    def test_detect_from_csv(self, mock_log, tmp_path):
        """Test the ``dectect_from_csv`` method.

        Test that when given a file path to a ``csv`` file, the current instance of
        ``SingleTableMetadata`` is being updated with the ``sdtypes`` of each
        column from the read data that is contained within the ``pandas.DataFrame`` from
        that ``csv`` file.

        Setup:
            - Instance of ``SingleTableMetadata``.

        Input:
            - String that represents the ``path`` to the ``csv`` file.

        Side Effects:
            - ``instance.columns`` has been updated with the expected ``sdtypes``.
            - A message is being printed.
        """
        # Setup
        instance = SingleTableMetadata()
        data = pd.DataFrame({
            'categorical': ['cat', 'dog', 'tiger', np.nan],
            'date': pd.to_datetime(['2021-02-02', np.nan, '2021-03-05', '2022-12-09']),
            'int': [1, 2, 3, 4],
            'float': [1., 2., 3., 4],
            'bool': [np.nan, True, False, True]
        })

        # Run
        filepath = tmp_path / 'mydata.csv'
        data.to_csv(filepath, index=False)
        instance.detect_from_csv(filepath)

        # Assert
        assert instance.columns == {
            'categorical': {'sdtype': 'categorical'},
            'date': {'sdtype': 'categorical'},
            'int': {'sdtype': 'numerical'},
            'float': {'sdtype': 'numerical'},
            'bool': {'sdtype': 'boolean'}
        }

        expected_log_calls = [
            call('Detected metadata:'),
            call(json.dumps(instance.to_dict(), indent=4))
        ]
        mock_log.info.assert_has_calls(expected_log_calls)

    @patch('sdv.metadata.single_table.LOGGER')
    def test_detect_from_csv_with_kwargs(self, mock_log, tmp_path):
        """Test the ``dectect_from_csv`` method.

        Test that when given a file path to a ``csv`` file, the current instance of
        ``SingleTableMetadata`` is being updated with the ``sdtypes`` of each
        column from the read data that is contained within the ``pandas.DataFrame`` from
        that ``csv`` file, having in consideration the ``kwargs`` that are passed.

        Setup:
            - Instance of ``SingleTableMetadata``.

        Input:
            - String that represents the ``path`` to the ``csv`` file.

        Side Effects:
            - ``instance.columns`` has been updated with the expected ``sdtypes``.
            - one of the columns must be datetime
            - A message is being printed.
        """
        # Setup
        instance = SingleTableMetadata()
        data = pd.DataFrame({
            'categorical': ['cat', 'dog', 'tiger', np.nan],
            'date': pd.to_datetime(['2021-02-02', np.nan, '2021-03-05', '2022-12-09']),
            'int': [1, 2, 3, 4],
            'float': [1., 2., 3., 4],
            'bool': [np.nan, True, False, True]
        })

        # Run
        filepath = tmp_path / 'mydata.csv'
        data.to_csv(filepath, index=False)
        instance.detect_from_csv(filepath, pandas_kwargs={'parse_dates': ['date']})

        # Assert
        assert instance.columns == {
            'categorical': {'sdtype': 'categorical'},
            'date': {'sdtype': 'datetime'},
            'int': {'sdtype': 'numerical'},
            'float': {'sdtype': 'numerical'},
            'bool': {'sdtype': 'boolean'}
        }

        expected_log_calls = [
            call('Detected metadata:'),
            call(json.dumps(instance.to_dict(), indent=4))
        ]
        mock_log.info.assert_has_calls(expected_log_calls)

    def test__validate_dataype_strings(self):
        """Test ``_validate_dataype`` for strings.

        Input:
            - A string

        Output:
            - True
        """
        # Setup
        instance = SingleTableMetadata()

        # Run
        out = instance._validate_datatype('10')

        # Assert
        assert out is True

    def test__validate_dataype_int(self):
        """Test ``_validate_dataype`` for invalid datatypes.

        Input:
            - A non-string and non-tuple

        Output:
            - False
        """
        # Setup
        instance = SingleTableMetadata()

        # Run
        out = instance._validate_datatype(10)

        # Assert
        assert out is False

    def test__validate_dataype_tuple(self):
        """Test ``_validate_dataype`` for tuples.

        Input:
            - A tuple of strings

        Output:
            - True
        """
        # Setup
        instance = SingleTableMetadata()

        # Run
        out = instance._validate_datatype(('10', '20'))

        # Assert
        assert out is True

    def test__validate_dataype_invalid_tuple(self):
        """Test ``_validate_dataype`` for tuples of non-strings.

        Input:
            - A tuple with some non-strings

        Output:
            - False
        """
        # Setup
        instance = SingleTableMetadata()

        # Run
        out = instance._validate_datatype(('10', '20', 30))

        # Assert
        assert out is False

    def test_set_primary_key_validation_dtype(self):
        """Test that ``set_primary_key`` crashes for invalid arguments.

        Input:
            - A tuple with non-string values.

        Side Effect:
            - An ``InvalidMetadataError`` should be raised.
        """
        # Setup
        instance = SingleTableMetadata()

        err_msg = (
            "'primary_key' must be a string or tuple of strings."
        )
        # Run / Assert
        with pytest.raises(InvalidMetadataError, match=err_msg):
            instance.set_primary_key(('1', 2, '3'))

    def test_set_primary_key_validation_columns(self):
        """Test that ``set_primary_key`` crashes for invalid arguments.

        Setup:
            - A ``SingleTableMetadata`` instance with ``_columns`` set.

        Input:
            - A tuple with columns not present in ``_columns``.

        Side Effect:
            - An ``InvalidMetadataError`` should be raised.
        """
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {'a', 'd'}

        err_msg = (
            "Unknown primary key values {'b'}."
            ' Keys should be columns that exist in the table.'
        )
        # Run / Assert
        with pytest.raises(InvalidMetadataError, match=err_msg):
            instance.set_primary_key(('a', 'b', 'd'))
            # NOTE: used to be ('a', 'b', 'd', 'c')

    def test_set_primary_key_validation_categorical(self):
        """Test that ``set_primary_key`` crashes when its sdtype is categorical.

        Input:
            - A tuple of keys, some of which have sdtype categorical.

        Side Effect:
            - An ``InvalidMetadataError`` should be raised.
        """
        # Setup
        instance = SingleTableMetadata()
        instance.add_column('column1', sdtype='categorical')
        instance.add_column('column2', sdtype='categorical')
        instance.add_column('column3', sdtype='id')

        err_msg = re.escape(
            "The primary_keys ['column1', 'column2'] must be type 'id' or another PII type."
        )
        # Run / Assert
        with pytest.raises(InvalidMetadataError, match=err_msg):
            instance.set_primary_key(('column1', 'column2', 'column3'))

    def test_set_primary_key(self):
        """Test that ``set_primary_key`` sets the ``_primary_key`` value."""
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {'column': {'sdtype': 'id'}}

        # Run
        instance.set_primary_key('column')

        # Assert
        assert instance.primary_key == 'column'

    def test_set_primary_key_tuple(self):
        """Test that ``set_primary_key`` sets the ``_primary_key`` value for tuples."""
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {'col1': {'sdtype': 'id'}, 'col2': {'sdtype': 'id'}}

        # Run
        instance.set_primary_key(('col1', 'col2'))

        # Assert
        assert instance.primary_key == ('col1', 'col2')

    @patch('sdv.metadata.single_table.warnings')
    def test_set_primary_key_already_exists_warning(self, warning_mock):
        """Test that ``set_primary_key`` raises a warning when a primary key already exists.

        Setup:
            - An instance of ``SingleTableMetadata`` with ``_primary_key`` set.

        Input:
            - String.

        Side Effect:
            - A warning should be raised.
        """
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {'column1': {'sdtype': 'id'}}
        instance.primary_key = 'column0'

        # Run
        instance.set_primary_key('column1')

        # Assert
        warning_msg = "There is an existing primary key 'column0'. This key will be removed."
        assert warning_mock.warn.called_once_with(warning_msg)
        assert instance.primary_key == 'column1'

    @patch('sdv.metadata.single_table.warnings')
    def test_set_primary_key_in_alternate_keys_warning(self, warning_mock):
        """Test that ``set_primary_key`` raises a warning the key is in ``self.alternate_keys``.

        Setup:
            Set the ``self.alternate_keys`` list to contain the key being added.
        """
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {'column1': {'sdtype': 'id'}}
        instance.primary_key = 'column0'
        instance.alternate_keys = ['column1', 'column2']

        # Run
        instance.set_primary_key('column1')

        # Assert
        warning_msg = (
            'column1 is currently set as an alternate key and will be removed from that list.'
        )
        assert warning_mock.warn.called_once_with(warning_msg)
        assert instance.primary_key == 'column1'
        assert instance.alternate_keys == ['column2']

    def test_set_sequence_key_validation_dtype(self):
        """Test that ``set_sequence_key`` crashes for invalid arguments.

        Input:
            - A tuple with non-string values.

        Side Effect:
            - An ``InvalidMetadataError`` should be raised.
        """
        # Setup
        instance = SingleTableMetadata()

        err_msg = "'sequence_key' must be a string or tuple of strings."
        # Run / Assert
        with pytest.raises(InvalidMetadataError, match=err_msg):
            instance.set_sequence_key(('1', 2, '3'))

    def test_set_sequence_key_validation_columns(self):
        """Test that ``set_sequence_key`` crashes for invalid arguments.

        Setup:
            - A ``SingleTableMetadata`` instance with ``_columns`` set.

        Input:
            - A tuple with columns not present in ``_columns``.

        Side Effect:
            - An ``InvalidMetadataError`` should be raised.
        """
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {'a', 'd'}

        err_msg = (
            "Unknown sequence key values {'b'}."
            ' Keys should be columns that exist in the table.'
        )
        # Run / Assert
        with pytest.raises(InvalidMetadataError, match=err_msg):
            instance.set_sequence_key(('a', 'b', 'd'))
            # NOTE: used to be ('a', 'b', 'd', 'c')

    def test_set_sequence_key_validation_categorical(self):
        """Test that ``set_sequence_key`` crashes when its sdtype is categorical.

        Input:
            - A tuple of keys, some of which have sdtype categorical.

        Side Effect:
            - An ``InvalidMetadataError`` should be raised.
        """
        # Setup
        instance = SingleTableMetadata()
        instance.add_column('column1', sdtype='categorical')
        instance.add_column('column2', sdtype='categorical')
        instance.add_column('column3', sdtype='id')

        err_msg = re.escape(
            "The sequence_keys ['column1', 'column2'] must be type 'id' or another PII type."
        )
        # Run / Assert
        with pytest.raises(InvalidMetadataError, match=err_msg):
            instance.set_sequence_key(('column1', 'column2', 'column3'))

    def test_set_sequence_key(self):
        """Test that ``set_sequence_key`` sets the ``_sequence_key`` value."""
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {'column': {'sdtype': 'id'}}

        # Run
        instance.set_sequence_key('column')

        # Assert
        assert instance.sequence_key == 'column'

    def test_set_sequence_key_tuple(self):
        """Test that ``set_sequence_key`` sets ``_sequence_key`` for tuples."""
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {'col1': {'sdtype': 'id'}, 'col2': {'sdtype': 'id'}}

        # Run
        instance.set_sequence_key(('col1', 'col2'))

        # Assert
        assert instance.sequence_key == ('col1', 'col2')

    @patch('sdv.metadata.single_table.warnings')
    def test_set_sequence_key_warning(self, warning_mock):
        """Test that ``set_sequence_key`` raises a warning when a sequence key already exists.

        Setup:
            - An instance of ``SingleTableMetadata`` with ``_sequence_key`` set.

        Input:
            - String.

        Side Effect:
            - A warning should be raised.
        """
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {'column1': {'sdtype': 'id'}}
        instance.sequence_key = 'column0'

        # Run
        instance.set_sequence_key('column1')

        # Assert
        warning_msg = "There is an existing sequence key 'column0'. This key will be removed."
        assert warning_mock.warn.called_once_with(warning_msg)
        assert instance.sequence_key == 'column1'

    def test_add_alternate_keys_validation_dtype(self):
        """Test that ``add_alternate_keys`` crashes for invalid arguments.

        Input:
            - A list with tuples with non-string values.

        Side Effect:
            - An ``InvalidMetadataError`` should be raised.
        """
        # Setup
        instance = SingleTableMetadata()

        err_msg = "'alternate_keys' must be a list of strings or a list of tuples of strings."
        # Run / Assert
        with pytest.raises(InvalidMetadataError, match=err_msg):
            instance.add_alternate_keys(['col1', ('1', 2, '3'), 'col3'])

    def test_add_alternate_keys_validation_columns(self):
        """Test that ``add_alternate_keys`` crashes for invalid arguments.

        Setup:
            - A ``SingleTableMetadata`` instance with ``_columns`` set.

        Input:
            - A tuple with non-string values.

        Side Effect:
            - An ``InvalidMetadataError`` should be raised.
        """
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {'abc', '213', '312'}

        err_msg = (
            "Unknown alternate key values {'123'}."
            ' Keys should be columns that exist in the table.'
        )
        # Run / Assert
        with pytest.raises(InvalidMetadataError, match=err_msg):
            instance.add_alternate_keys(['abc', ('123', '213', '312')])
            # NOTE: used to be ['abc', ('123', '213', '312'), 'bca']

    def test_add_alternate_keys_validation_categorical(self):
        """Test that ``add_alternate_keys`` crashes when its sdtype is categorical.

        Input:
            - A list of keys, some of which have sdtype categorical.

        Side Effect:
            - An ``InvalidMetadataError`` should be raised.
        """
        # Setup
        instance = SingleTableMetadata()
        instance.add_column('column1', sdtype='categorical')
        instance.add_column('column2', sdtype='categorical')
        instance.add_column('column3', sdtype='id')

        err_msg = re.escape(
            "The alternate_keys ['column1', 'column2'] must be type 'id' or another PII type."
        )
        # Run / Assert
        with pytest.raises(InvalidMetadataError, match=err_msg):
            instance.add_alternate_keys([('column1', 'column2'), 'column3'])

    def test_add_alternate_keys_validation_primary_key(self):
        """Test that ``add_alternate_keys`` crashes when the key is a primary key.

        If the ``_primary_key`` is set to be the same as the key being added, an
        ``InvalidMetadataError``
        should be raised.
        """
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {'column1': {'sdtype': 'numerical'}}
        instance.primary_key = 'column1'

        err_msg = re.escape(
            "Invalid alternate key 'column1'. The key is already specified as a primary key."
        )
        # Run / Assert
        with pytest.raises(InvalidMetadataError, match=err_msg):
            instance.add_alternate_keys(['column1'])

    def test_add_alternate_keys(self):
        """Test that ``add_alternate_keys`` adds the columns to the ``_alternate_keys``."""
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {
            'column1': {'sdtype': 'id'},
            'column2': {'sdtype': 'id'},
            'column3': {'sdtype': 'id'}
        }

        # Run
        instance.add_alternate_keys(['column1', ('column2', 'column3')])

        # Assert
        assert instance.alternate_keys == ['column1', ('column2', 'column3')]

    @patch('sdv.metadata.single_table.warnings')
    def test_add_alternate_keys_duplicate(self, warnings_mock):
        """Test that the method does not add columns that are already in ``_alternate_keys``."""
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {
            'column1': {'sdtype': 'id'},
            'column2': {'sdtype': 'id'},
            'column3': {'sdtype': 'id'}
        }
        instance.alternate_keys = ['column3']

        # Run
        instance.add_alternate_keys(['column1', 'column2', 'column3'])

        # Assert
        assert instance.alternate_keys == ['column3', 'column1', 'column2']
        message = 'column3 is already an alternate key.'
        warnings_mock.warn.assert_called_once_with(message)

    def test_set_sequence_index_validation(self):
        """Test that ``set_sequence_index`` crashes for invalid arguments.

        Input:
            - A non-string value.

        Side Effect:
            - An ``InvalidMetadataError`` should be raised.
        """
        # Setup
        instance = SingleTableMetadata()

        err_msg = "'sequence_index' must be a string."
        # Run / Assert
        with pytest.raises(InvalidMetadataError, match=err_msg):
            instance.set_sequence_index(('column1', 'column2'))

    def test_set_sequence_index_validation_columns(self):
        """Test that ``set_sequence_index`` crashes for invalid arguments.

        Setup:
            - A ``SingleTableMetadata`` instance with ``_columns`` set.

        Input:
            - A string not present in ``_columns``.

        Side Effect:
            - An ``InvalidMetadataError`` should be raised.
        """
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {'a', 'd'}

        err_msg = (
            "Unknown sequence index value {'column'}."
            ' Keys should be columns that exist in the table.'
        )
        # Run / Assert
        with pytest.raises(InvalidMetadataError, match=err_msg):
            instance.set_sequence_index('column')

    def test_set_sequence_index_column_not_numerical_or_datetime(self):
        """Test that the method errors if the column is not numerical or datetime."""
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {
            'a': {'sdtype': 'numerical'},
            'd': {'sdtype': 'categorical'}
        }

        # Run / Assert
        error_message = "The sequence_index must be of type 'datetime' or 'numerical'."
        with pytest.raises(InvalidMetadataError, match=error_message):
            instance.set_sequence_index('d')

    def test_set_sequence_index(self):
        """Test that ``set_sequence_index`` sets the ``_sequence_index`` value."""
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {'column': {'sdtype': 'numerical'}}

        # Run
        instance.set_sequence_index('column')

        # Assert
        assert instance.sequence_index == 'column'

    def test_validate_sequence_index_not_in_sequence_key(self):
        """Test the ``_validate_sequence_index_not_in_sequence_key`` method."""
        # Setup
        instance = SingleTableMetadata()
        instance.sequence_key = ('abc', 'def')
        instance.sequence_index = 'abc'

        err_msg = (
            "'sequence_index' and 'sequence_key' have the same value {'abc'}."
            ' These columns must be different.'
        )
        # Run / Assert
        with pytest.raises(InvalidMetadataError, match=err_msg):
            instance._validate_sequence_index_not_in_sequence_key()

    def test_validate(self):
        """Test the ``validate`` method.

        Ensure the method calls the correct methods with the correct parameters.

        Setup:
            - A ``SingleTableMetadata`` instance with:
                - ``_columns``, ``_primary_key``, ``_alternate_keys``,
                  ``_sequence_key`` and ``_sequence_index`` defined.
                - ``_validate_key``, ``_validate_alternate_keys``, ``_validate_sequence_index``
                  and ``_validate_sequence_index_not_in_sequence_key`` mocked.
        """
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {'col1': {'sdtype': 'numerical'}, 'col2': {'sdtype': 'numerical'}}
        instance.primary_key = 'col1'
        instance.alternate_keys = ['col2']
        instance.sequence_key = 'col1'
        instance.sequence_index = 'col2'
        instance._validate_key = Mock()
        instance._validate_alternate_keys = Mock()
        instance._validate_sequence_index = Mock()
        instance._validate_sequence_index_not_in_sequence_key = Mock()
        instance._validate_column = Mock(side_effect=InvalidMetadataError('column_error'))

        err_msg = re.escape(
            'The following errors were found in the metadata:'
            '\n\ncolumn_error'
            '\ncolumn_error'
        )
        # Run
        with pytest.raises(InvalidMetadataError, match=err_msg):
            instance.validate()

        # Assert
        instance._validate_key.assert_has_calls(
            [call(instance.primary_key, 'primary'), call(instance.sequence_key, 'sequence')]
        )
        instance._validate_column.assert_has_calls(
            [call('col1', sdtype='numerical'), call('col2', sdtype='numerical')]
        )
        instance._validate_alternate_keys.assert_called_once_with(instance.alternate_keys)
        instance._validate_sequence_index.assert_called_once_with(instance.sequence_index)
        instance._validate_sequence_index_not_in_sequence_key.assert_called_once()

    def test_to_dict(self):
        """Test the ``to_dict`` method from ``SingleTableMetadata``.

        Setup:
            - Instance of ``SingleTableMetadata`` and modify the ``instance.columns`` to ensure
            that ``to_dict`` works properly.
        Output:
            - A dictionary representation of the ``instance`` that does not modify the
              internal dictionaries.
        """
        # Setup
        instance = SingleTableMetadata()
        instance.columns['my_column'] = 'value'

        # Run
        result = instance.to_dict()

        # Assert
        assert result == {
            'columns': {'my_column': 'value'},
            'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1'
        }

        # Ensure that the output object does not alterate the inside object
        result['columns']['my_column'] = 1
        assert instance.columns['my_column'] == 'value'

    def test_load_from_dict(self):
        """Test that ``load_from_dict`` returns a instance with the ``dict`` updated objects."""
        # Setup
        my_metadata = {
            'columns': {'my_column': 'value'},
            'primary_key': 'pk',
            'alternate_keys': [],
            'sequence_key': None,
            'sequence_index': None,
            'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1'
        }

        # Run
        instance = SingleTableMetadata.load_from_dict(my_metadata)

        # Assert
        assert instance.columns == {'my_column': 'value'}
        assert instance.primary_key == 'pk'
        assert instance.sequence_key is None
        assert instance.alternate_keys == []
        assert instance.sequence_index is None
        assert instance._version == 'SINGLE_TABLE_V1'

    @patch('sdv.metadata.utils.Path')
    def test_load_from_json_path_does_not_exist(self, mock_path):
        """Test the ``load_from_json`` method.

        Test that the method raises a ``ValueError`` when the specified path does not
        exist.

        Mock:
            - Mock the ``Path`` library in order to return ``False``, that the file does not exist.

        Input:
            - String representing a filepath.

        Side Effects:
            - A ``ValueError`` is raised pointing that the ``file`` does not exist.
        """
        # Setup
        mock_path.return_value.exists.return_value = False
        mock_path.return_value.name = 'filepath.json'

        # Run / Assert
        error_msg = (
            "A file named 'filepath.json' does not exist. Please specify a different filename."
        )
        with pytest.raises(ValueError, match=error_msg):
            SingleTableMetadata.load_from_json('filepath.json')

    @patch('sdv.metadata.utils.open')
    @patch('sdv.metadata.utils.Path')
    @patch('sdv.metadata.utils.json')
    def test_load_from_json_schema_not_present(self, mock_json, mock_path, mock_open):
        """Test the ``load_from_json`` method.

        Test that the method raises an ``InvalidMetadataError`` when the specified ``json`` file
        does not contain a ``METADATA_SPEC_VERSION`` in it.

        Mock:
            - Mock the ``Path`` library in order to return ``True``, so the file exists.
            - Mock the ``json`` library in order to use a custom return.
            - Mock the ``open`` in order to avoid loading a binary file.

        Input:
            - String representing a filepath.

        Side Effects:
            - An ``InvalidMetadataError`` is raised pointing that the given metadata configuration
              is not compatible with the current version.
        """
        # Setup
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.name = 'filepath.json'
        mock_json.load.return_value = {
            'columns': {
                'animals': {
                    'type': 'categorical'
                }
            },
            'primary_key': 'animals',
        }

        # Run / Assert
        error_msg = (
            'This metadata file is incompatible with the ``SingleTableMetadata`` '
            'class and version.'
        )
        with pytest.raises(InvalidMetadataError, match=error_msg):
            SingleTableMetadata.load_from_json('filepath.json')

    @patch('sdv.metadata.utils.open')
    @patch('sdv.metadata.utils.Path')
    @patch('sdv.metadata.utils.json')
    def test_load_from_json(self, mock_json, mock_path, mock_open):
        """Test the ``load_from_json`` method.

        Test that ``load_from_json`` function creates an instance with the contents returned by the
        ``json`` load function.

        Mock:
            - Mock the ``Path`` library in order to return ``True``.
            - Mock the ``json`` library in order to use a custom return.
            - Mock the ``open`` in order to avoid loading a binary file.

        Input:
            - String representing a filepath.

        Output:
            - ``SingleTableMetadata`` instance with the custom configuration from the ``json``
              file (``json.load`` return value)
        """
        # Setup
        instance = SingleTableMetadata()
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.name = 'filepath.json'
        mock_json.load.return_value = {
            'columns': {
                'animals': {
                    'type': 'categorical'
                }
            },
            'primary_key': 'animals',
            'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1'
        }

        # Run
        instance = SingleTableMetadata.load_from_json('filepath.json')

        # Assert
        assert instance.columns == {'animals': {'type': 'categorical'}}
        assert instance.primary_key == 'animals'
        assert instance.sequence_key is None
        assert instance.alternate_keys == []
        assert instance.sequence_index is None
        assert instance._version == 'SINGLE_TABLE_V1'

    @patch('sdv.metadata.utils.Path')
    def test_save_to_json_file_exists(self, mock_path):
        """Test the ``save_to_json`` method.

        Test that when attempting to write over a file that already exists, the method
        raises a ``ValueError``.

        Setup:
            - instance of ``SingleTableMetadata``.
        Mock:
            - Mock ``Path`` in order to point that the file does exist.

        Side Effects:
            - Raise ``ValueError`` pointing that the file does exist.
        """
        # Setup
        instance = SingleTableMetadata()
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.name = 'filepath.json'

        # Run / Assert
        error_msg = (
            "A file named 'filepath.json' already exists in this folder. Please specify "
            'a different filename.'
        )
        with pytest.raises(ValueError, match=error_msg):
            instance.save_to_json('filepath.json')

    def test_save_to_json(self, tmp_path):
        """Test the ``save_to_json`` method.

        Test that ``save_to_json`` stores a ``json`` file and dumps the instance dict into
        it.

        Setup:
            - instance of ``SingleTableMetadata``.
            - Use ``TemporaryDirectory`` to store the file in order to read it afterwards and
              assert it's contents.

        Side Effects:
            - Creates a json representation of the instance.
        """
        # Setup
        instance = SingleTableMetadata()

        # Run / Assert
        file_name = tmp_path / 'singletable.json'
        instance.save_to_json(file_name)

        with open(file_name, 'rb') as single_table_file:
            saved_metadata = json.load(single_table_file)
            assert saved_metadata == instance.to_dict()

    @patch('sdv.metadata.single_table.json')
    def test___repr__(self, mock_json):
        """Test that the ``__repr__`` method.

        Test that the ``__repr__`` method calls the ``json.dumps``  method and
        returns its output.

        Setup:
            - Instance of ``SingleTableMetadata``.
        Mock:
            - ``json`` from ``sdv.metadata.single_table``.

        Output:
            - ``json.dumps`` return value.
        """
        # Setup
        instance = SingleTableMetadata()

        # Run
        res = instance.__repr__()

        # Assert
        mock_json.dumps.assert_called_once_with(instance.to_dict(), indent=4)
        assert res == mock_json.dumps.return_value

    @patch('sdv.metadata.single_table.read_json')
    @patch('sdv.metadata.single_table.convert_metadata')
    @patch('sdv.metadata.single_table.SingleTableMetadata.load_from_dict')
    def test_upgrade_metadata(self, from_dict_mock, convert_mock, read_json_mock):
        """Test the ``upgrade_metadata`` method.

        The method should validate that the ``new_filepath`` does not exist, read the old metadata
        from a file, convert it and save it to the ``new_filepath``.

        Setup:
            - Mock ``read_json``.
            - Mock ``validate_file_does_not_exist``.
            - Mock the ``convert_metadata`` method to return something.
            - Mock the ``from_dict`` method to return a mock.

        Input:
            - A fake old filepath.
            - A fake new filepath.

        Side effect:
            - The mock should call ``save_to_json`` and ``validate``.
        """
        # Setup
        convert_mock.return_value = {}
        new_metadata = Mock()
        from_dict_mock.return_value = new_metadata

        # Run
        SingleTableMetadata.upgrade_metadata('old')

        # Assert
        convert_mock.assert_called_once()
        read_json_mock.assert_called_once_with('old')
        new_metadata.validate.assert_called_once()

    @patch('sdv.metadata.single_table.read_json')
    @patch('sdv.metadata.single_table.convert_metadata')
    @patch('sdv.metadata.single_table.SingleTableMetadata.load_from_dict')
    def test_upgrade_metadata_multiple_tables(
            self, from_dict_mock, convert_mock, read_json_mock):
        """Test the ``upgrade_metadata`` method.

        If the old metadata is in the multi-table format (has 'tables'), but it only contains one
        table, then it should still get converted.

        Setup:
            - Mock ``read_json`` to return a multi-table metadata dict with one table.
            - Mock ``validate_file_does_not_exist``.
            - Mock the ``convert_metadata`` method to return something.
            - Mock the ``from_dict`` method to return a mock.

        Input:
            - A fake old filepath.
            - A fake new filepath.

        Side effect:
            - The conversion should be done on the nested table.
        """
        # Setup
        convert_mock.return_value = {}
        new_metadata = Mock()
        from_dict_mock.return_value = new_metadata
        read_json_mock.return_value = {
            'tables': {'table': {'columns': {}}}
        }

        # Run
        SingleTableMetadata.upgrade_metadata('old')

        # Assert
        convert_mock.assert_called_once_with({'columns': {}})
        new_metadata.validate.assert_called_once()

    @patch('sdv.metadata.single_table.read_json')
    @patch('sdv.metadata.single_table.convert_metadata')
    @patch('sdv.metadata.single_table.SingleTableMetadata.load_from_dict')
    def test_upgrade_metadata_multiple_tables_fails(
            self, from_dict_mock, convert_mock, read_json_mock):
        """Test the ``upgrade_metadata`` method.

        If the old metadata is in the multi-table format (has 'tables'), but contains multiple
        tables, then an error should be raised.

        Setup:
            - Mock ``read_json`` to return a multi-table metadata dict.
            - Mock ``validate_file_does_not_exist``.
            - Mock the ``convert_metadata`` method to return something.
            - Mock the ``from_dict`` method to return a mock.

        Input:
            - A fake old filepath.
            - A fake new filepath.

        Side effect:
            - An ``InvalidMetadataError`` should be raised.
        """
        # Setup
        convert_mock.return_value = {}
        new_metadata = Mock()
        from_dict_mock.return_value = new_metadata
        read_json_mock.return_value = {
            'tables': {'table1': {'columns': {}}, 'table2': {}}
        }

        # Run
        message = (
            'There are multiple tables specified in the JSON. '
            'Try using the MultiTableMetadata class to upgrade this file.'
        )
        with pytest.raises(InvalidMetadataError, match=message):
            SingleTableMetadata.upgrade_metadata('old')

    @patch('sdv.metadata.single_table.warnings')
    @patch('sdv.metadata.single_table.read_json')
    @patch('sdv.metadata.single_table.convert_metadata')
    @patch('sdv.metadata.single_table.SingleTableMetadata.load_from_dict')
    def test_upgrade_metadata_validate_error(
            self, from_dict_mock, convert_mock, read_json_mock, warnings_mock):
        """Test the ``upgrade_metadata`` method.

        The method should raise a warning with any validation errors after the metadata is
        converted.

        Setup:
            - Mock ``read_json``.
            - Mock ``validate_file_does_not_exist``.
            - Mock the ``convert_metadata`` method to return something.
            - Mock the ``from_dict`` method to return a mock.

        Input:
            - A fake old filepath.
            - A fake new filepath.

        Side effect:
            - The mock should call ``save_to_json`` and ``validate``.
        """
        # Setup
        convert_mock.return_value = {}
        new_metadata = Mock()
        from_dict_mock.return_value = new_metadata
        new_metadata.validate.side_effect = InvalidMetadataError('blah')

        # Run
        SingleTableMetadata.upgrade_metadata('old')

        # Assert
        convert_mock.assert_called_once()
        read_json_mock.assert_called_once_with('old')
        new_metadata.validate.assert_called_once()
        warnings_mock.warn.assert_called_once_with(
            'Successfully converted the old metadata, but the metadata was not valid. '
            'To use this with the SDV, please fix the following errors.\n blah'
        )
