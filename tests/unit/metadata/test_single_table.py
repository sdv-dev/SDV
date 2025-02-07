"""Test Single Table Metadata."""

import json
import logging
import re
import warnings
from datetime import datetime
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from sdv.errors import InvalidDataError
from sdv.metadata.errors import InvalidMetadataError
from sdv.metadata.single_table import SingleTableMetadata
from tests.utils import catch_sdv_logs


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
            {'computer_representation': 'Int8', 'datetime_format': None, 'pii': True},
            re.escape("Invalid values '(datetime_format, pii)' for numerical column 'age'."),
        ),
        (
            'start_date',
            'datetime',
            {'datetime_format': '%Y-%d', 'pii': True},
            re.escape("Invalid values '(pii)' for datetime column 'start_date'."),
        ),
        (
            'name',
            'categorical',
            {'pii': True, 'ordering': ['a', 'b'], 'ordered': 'numerical_values'},
            re.escape("Invalid values '(ordered, ordering, pii)' for categorical column 'name'."),
        ),
        (
            'synthetic',
            'boolean',
            {'pii': True},
            re.escape("Invalid values '(pii)' for boolean column 'synthetic'."),
        ),
        (
            'phrase',
            'id',
            {'regex_format': '[A-z]', 'pii': True, 'anonymization': True},
            re.escape("Invalid values '(anonymization, pii)' for id column 'phrase'."),
        ),
        (
            'phone',
            'phone_number',
            {'anonymization': True, 'order_by': 'phone_number'},
            re.escape(
                "Invalid values '(anonymization, order_by)' for phone_number column 'phone'."
            ),
        ),
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
        assert instance._updated is False

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
        'computer_representation', SingleTableMetadata._NUMERICAL_REPRESENTATIONS
    )
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
            "Invalid datetime format string '%1-%Y-%m-%d-%' for datetime column 'start_date'."
        )
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
        instance._get_unexpected_kwargs = Mock(return_value=None)

        # Run
        instance._validate_unexpected_kwargs(column_name, sdtype, **kwargs)

        # Assert
        instance._get_unexpected_kwargs.assert_called_once_with(sdtype, **kwargs)

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

    @patch('sdv.metadata.single_table.is_faker_function')
    def test__validate_column_invalid_sdtype(self, mock_is_faker_function):
        """Test the method with an invalid sdtype.

        If the sdtype isn't one of the supported types, anonymized types or Faker functions,
        then an error should be raised.
        """
        # Setup
        instance = SingleTableMetadata()
        mock_is_faker_function.return_value = False

        # Run and Assert
        error_msg = re.escape(
            "Invalid sdtype: 'fake_type' is not recognized. Please use one of the "
            'supported SDV sdtypes.'
        )
        with pytest.raises(InvalidMetadataError, match=error_msg):
            instance._validate_column_args('column', 'fake_type')

        error_msg = re.escape(
            'Invalid sdtype: None is not a string. Please use one of the supported SDV sdtypes.'
        )
        with pytest.raises(InvalidMetadataError, match=error_msg):
            instance._validate_column_args('column', None)

        mock_is_faker_function.assert_called_once_with('fake_type')

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
        instance._validate_column_args('age', 'numerical', computer_representation='Int8')

        # Assert
        mock__validate_kwargs.assert_called_once_with(
            'age', 'numerical', computer_representation='Int8'
        )
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
        instance._validate_column_args('name', 'categorical', order=['a', 'b', 'c'])

        # Assert
        mock__validate_kwargs.assert_called_once_with('name', 'categorical', order=['a', 'b', 'c'])
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
        instance._validate_column_args('snythetic', 'boolean')

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
        instance._validate_column_args('start', 'datetime')

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
        instance._validate_column_args('phrase', 'id', regex_format='[A-z0-9]', pii=True)

        # Assert
        mock__validate_kwargs.assert_called_once_with(
            'phrase', 'id', regex_format='[A-z0-9]', pii=True
        )
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
            'a': {'sdtype': 'numerical', 'computer_representation': 'Int64'}
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
            "Column name 'age' already exists. Use 'update_column' to update an existing column."
        )
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

    @patch('sdv.metadata.single_table.is_faker_function')
    def test_add_column_invalid_sdtype(self, mock_is_faker_function):
        """Test the method with an invalid sdtype.

        If the sdtype isn't one of the supported types, anonymized types or Faker functions,
        then an error should be raised.
        """
        # Setup
        instance = SingleTableMetadata()
        mock_is_faker_function.return_value = False

        # Run and Assert
        error_msg = re.escape(
            "Invalid sdtype: 'fake_type' is not recognized. Please use one of the "
            'supported SDV sdtypes.'
        )
        with pytest.raises(InvalidMetadataError, match=error_msg):
            instance.add_column('column', sdtype='fake_type')

        mock_is_faker_function.assert_called_once_with('fake_type')

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
        assert instance.columns['age'] == {'sdtype': 'numerical', 'computer_representation': 'Int8'}

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

    def test__get_unexpected_kwargs(self):
        """Test the ``_get_unexpected_kwargs`` method."""
        # Setup
        instance = SingleTableMetadata()
        instance._validate_unexpected_kwargs = Mock()

        # Run
        with_unexpected_kwargs = instance._get_unexpected_kwargs('numerical', pii=True)
        without_unexpected_kwargs = instance._get_unexpected_kwargs('latitude', pii=True)

        # Assert
        assert with_unexpected_kwargs == 'pii'
        assert without_unexpected_kwargs == set()

    def test__validate_update_column_kwargs_with_sdtype(self):
        """Test the ``_validate_update_column`` when kwargs has the sdtype key."""
        # Setup
        instance = SingleTableMetadata()
        instance._validate_column_exists = Mock()
        instance._validate_column_args = Mock()
        instance.columns = {'age': {'sdtype': 'categorical'}}
        kwargs = {'sdtype': 'numerical', 'computer_representation': 'Int8'}

        # Run
        instance._validate_update_column('age', **kwargs)

        # Assert
        instance._validate_column_exists.assert_called_once_with('age')
        expected_kwargs = {'computer_representation': 'Int8'}
        instance._validate_column_args.assert_called_once_with(
            'age', 'numerical', **expected_kwargs
        )

    def test_update_column(self):
        """Test the ``update_column`` method."""
        # Setup
        instance = SingleTableMetadata()
        instance._validate_update_column = Mock()
        instance.columns = {'age': {'sdtype': 'numerical'}}

        # Run
        instance.update_column('age', sdtype='categorical', order_by='numerical_value')

        # Assert
        instance._validate_update_column.assert_called_once_with(
            'age', sdtype='categorical', order_by='numerical_value'
        )
        assert instance.columns['age'] == {'sdtype': 'categorical', 'order_by': 'numerical_value'}

    @patch('sdv.metadata.single_table.SingleTableMetadata._validate_column_args')
    @patch('sdv.metadata.single_table.SingleTableMetadata._validate_column_exists')
    def test_update_column_sdtype_in_kwargs(
        self, mock__validate_column_exists, mock__validate_column
    ):
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
        assert instance.columns['age'] == {'sdtype': 'categorical', 'order_by': 'numerical_value'}
        mock__validate_column_exists.assert_called_once_with('age')
        mock__validate_column.assert_called_once_with(
            'age', 'categorical', order_by='numerical_value'
        )

    @patch('sdv.metadata.single_table.SingleTableMetadata._validate_column_args')
    @patch('sdv.metadata.single_table.SingleTableMetadata._validate_column_exists')
    def test_update_column_no_sdtype(self, mock__validate_column_exists, mock__validate_column):
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
            'computer_representation': 'Float',
        }
        mock__validate_column_exists.assert_called_once_with('age')
        mock__validate_column.assert_called_once_with(
            'age', 'numerical', computer_representation='Float'
        )

    def test_update_columns_sdtype_in_kwargs_error(self):
        """Test the ``update_columns`` method.

        Test that ``update_columns`` with invalid ``sdtype`` and other ``kwargs`` combination
        raises an ``InvalidMetadataError``.
        """
        # Setup
        instance = SingleTableMetadata()

        # Run / Assert
        error_msg = re.escape("Invalid values '(pii)' for 'numerical' sdtype.")

        with pytest.raises(InvalidMetadataError, match=error_msg):
            instance.update_columns(['col_1', 'col_2'], sdtype='numerical', pii=True)

    @patch('sdv.metadata.single_table.is_faker_function')
    def test_update_columns_multiple_errors(self, mock_is_faker_function):
        """Test the ``update_columns`` method.

        Test that ``update_columns`` with multiple errors.
        Should raise an ``InvalidMetadataError`` with a summary of all the errors.
        """
        # Setup
        mock_is_faker_function.return_value = True
        instance = SingleTableMetadata()
        instance.columns = {
            'col_1': {'sdtype': 'country_code'},
            'col_2': {'sdtype': 'numerical'},
            'col_3': {'sdtype': 'categorical'},
        }

        # Run / Assert
        error_msg = re.escape(
            'The following errors were found when updating columns:\n\n'
            "Invalid values '(pii)' for numerical column 'col_2'.\n"
            "Invalid values '(pii)' for categorical column 'col_3'."
        )
        with pytest.raises(InvalidMetadataError, match=error_msg):
            instance.update_columns(['col_1', 'col_2', 'col_3'], pii=True)

        mock_is_faker_function.assert_called_once_with('country_code')

    def test_update_columns(self):
        """Test the ``update_columns`` method."""
        # Setup
        instance = SingleTableMetadata()
        instance._validate_update_column = Mock()
        instance._get_unexpected_kwargs = Mock(return_value=None)
        instance.columns = {'age': {'sdtype': 'numerical'}, 'salary': {'sdtype': 'numerical'}}

        # Run
        instance.update_columns(['age', 'salary'], sdtype='categorical')

        # Assert
        instance._get_unexpected_kwargs.assert_called_once_with('categorical')
        instance._validate_update_column.assert_has_calls([
            call('age', sdtype='categorical'),
            call('salary', sdtype='categorical'),
        ])
        assert instance.columns == {
            'age': {'sdtype': 'categorical'},
            'salary': {'sdtype': 'categorical'},
        }

    @patch('sdv.metadata.single_table.is_faker_function')
    def test_update_columns_kwargs_without_sdtype(self, mock_is_faker_function):
        """Test the ``update_columns`` method when there is no ``sdtype`` in the kwargs."""
        # Setup
        mock_is_faker_function.return_value = True
        instance = SingleTableMetadata()
        instance.columns = {
            'col_1': {'sdtype': 'country_code'},
            'col_2': {'sdtype': 'latitude'},
            'col_3': {'sdtype': 'longitude'},
        }

        # Run
        instance.update_columns(['col_1', 'col_2', 'col_3'], pii=True)

        # Assert
        assert instance.columns == {
            'col_1': {'sdtype': 'country_code', 'pii': True},
            'col_2': {'sdtype': 'latitude', 'pii': True},
            'col_3': {'sdtype': 'longitude', 'pii': True},
        }
        assert instance._updated is True
        mock_is_faker_function.assert_has_calls([
            call('country_code'),
            call('latitude'),
            call('longitude'),
        ])

    def test_update_columns_metadata(self):
        """Test the ``update_columns_metadata`` method."""
        # Setup
        instance = SingleTableMetadata()
        instance._validate_update_column = Mock()
        instance.columns = {'age': {'sdtype': 'numerical'}, 'salary': {'sdtype': 'numerical'}}

        # Run
        instance.update_columns_metadata({
            'age': {'sdtype': 'categorical'},
            'salary': {'computer_representation': 'Int64'},
        })

        # Assert
        instance._validate_update_column.assert_has_calls([
            call('age', sdtype='categorical'),
            call('salary', computer_representation='Int64'),
        ])
        assert instance.columns == {
            'age': {'sdtype': 'categorical'},
            'salary': {'sdtype': 'numerical', 'computer_representation': 'Int64'},
        }

    def test_update_columns_metadata_multiple_error(self):
        """Test the ``update_columns_metadata`` method with multiple error."""
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {'age': {'sdtype': 'numerical'}, 'hours': {'sdtype': 'numerical'}}

        # Run / Assert
        error_msg = re.escape(
            'The following errors were found when updating columns:\n\n'
            "Invalid values '(pii)' for numerical column 'age'.\n"
            "Invalid values '(datetime_format)' for categorical column 'hours'.\n"
            "Column name ('salary') does not exist in the table. Use 'add_column' to"
            ' add new column.'
        )
        with pytest.raises(InvalidMetadataError, match=error_msg):
            instance.update_columns_metadata({
                'age': {'pii': True},
                'hours': {'sdtype': 'categorical', 'datetime_format': '%Y-%m-%d'},
                'salary': {'sdtype': 'numerical'},
            })

    def test_get_column_names(self):
        """Test the ``get_column_names`` method filters for matching columns."""
        # Setup
        metadata = SingleTableMetadata()
        metadata.columns = {
            'id': {'sdtype': 'id'},
            'value1': {'sdtype': 'numerical'},
            'value2': {'sdtype': 'numerical', 'computer_representation': 'Float'},
        }

        # Run
        matches_no_filter = metadata.get_column_names()
        matches_numerical = metadata.get_column_names(sdtype='numerical')
        matches_extra = metadata.get_column_names(
            sdtype='numerical', computer_representation='Float'
        )

        # Assert
        assert set(matches_no_filter) == {'id', 'value1', 'value2'}
        assert set(matches_numerical) == {'value1', 'value2'}
        assert set(matches_extra) == {'value2'}

    def test__detect_pii_columns(self):
        """Test the ``_detect_pii_column`` method."""
        # Setup
        metadata = SingleTableMetadata()

        # Run and Assert
        assert metadata._detect_pii_column('user_first_name') == 'first_name'
        assert metadata._detect_pii_column('USER_FIRST_NAME') == 'first_name'
        assert metadata._detect_pii_column('User_Last_Name') == 'last_name'
        assert metadata._detect_pii_column('countrycode') == 'country_code'
        assert metadata._detect_pii_column('city') == 'city'
        assert metadata._detect_pii_column('non_cIty') == 'city'
        assert metadata._detect_pii_column('nonCity') == 'city'
        assert metadata._detect_pii_column('cIty') is None
        assert metadata._detect_pii_column('address') is None
        assert metadata._detect_pii_column('first_name_last_name') == 'first_name'
        assert metadata._detect_pii_column('license') == 'license_plate'
        assert metadata._detect_pii_column('resolving_loans') is None
        assert metadata._detect_pii_column('vin_loans') == 'vin'
        assert metadata._detect_pii_column('VinLoans') == 'vin'
        assert metadata._detect_pii_column('VINLOANSVIN') is None
        assert metadata._detect_pii_column('VIN') == 'vin'
        assert metadata._detect_pii_column('StateDepartment') == 'administrative_unit'
        assert metadata._detect_pii_column('STATEDEPARTMENT') is None

    def test__determine_sdtype_for_numbers(self):
        """Test the ``determine_sdtype_for_numbers`` method.

        Setup:
            - Instance of ``SingleTableMetadata``.
            - A series of numbers with less than 5 rows. Should be detected as numerical sdtype
            - A series of numbers with less than 10% unique values. Should be detected as
              categorical sdtype
            - A series of numbers with all unique values. Should be detected as id sdtype
            - A series of integers. Should be detected as numerical sdtype
        - A series of floats. Should be detected as numerical sdtype
        """
        # Setup
        instance = SingleTableMetadata()

        data_less_than_5_rows = pd.Series([1, np.nan, 3, 4, 5])
        data_less_than_10_percent_unique_values = pd.Series([
            1,
            2,
            2,
            None,
            2,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            np.nan,
            2,
            1,
            1,
            2,
            1,
            2,
            2,
        ])
        large_numerical_series = pd.Series(
            [400, 401, 402, 403, 404, 405, 406, 500, 501, 502, 503, 504, 505, 506] * 1000
        )

        large_categorical_series = pd.Series(
            [400, 401, 402, 403, 404, 500, 501, 502, 503, 504] * 1000
        )
        data_all_unique = pd.Series([1, 2, 3, 4, 5, 6])
        data_numerical_int = pd.Series([1, np.nan, 3, 4, 5, 6, 7, 8, 9, 10])
        data_numerical_float = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])

        # Run
        sdtype_less_than_5_rows = instance._determine_sdtype_for_numbers(data_less_than_5_rows)
        sdtype_less_than_10_percent_unique_values = instance._determine_sdtype_for_numbers(
            data_less_than_10_percent_unique_values
        )
        sdtype_all_unique = instance._determine_sdtype_for_numbers(data_all_unique)
        sdtype_numerical_int = instance._determine_sdtype_for_numbers(data_numerical_int)
        sdtype_numerical_float = instance._determine_sdtype_for_numbers(data_numerical_float)
        sdtype_large_numerical_series = instance._determine_sdtype_for_numbers(
            large_numerical_series
        )
        sdtype_large_categorical_series = instance._determine_sdtype_for_numbers(
            large_categorical_series
        )

        # Assert
        assert sdtype_less_than_5_rows == 'numerical'
        assert sdtype_less_than_10_percent_unique_values == 'categorical'
        assert sdtype_all_unique == 'id'
        assert sdtype_numerical_int == 'numerical'
        assert sdtype_numerical_float == 'numerical'
        assert sdtype_large_numerical_series == 'numerical'
        assert sdtype_large_categorical_series == 'categorical'

    def test__determine_sdtype_for_objects(self):
        """Test the ``_determine_sdtype_for_objects`` method."""
        # Setup
        instance = SingleTableMetadata()

        data_datetime = pd.Series(['2022-01-01', '2022-02-01', '2022-03-01'])
        wrong_datetime = pd.Series(['2022-01-01', '01-02-2022', '2022-03-01', '2022-03-01'])
        data_categorical_small = pd.Series(['a', 'b', 'c', 'd', 'e'])
        data_all_unique = pd.Series(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k'])
        data_categorical_large = pd.Series(['a'] * 10 + ['b'] * 4)
        data_unknown = pd.Series(['a', 'b', 'c', 'c', 1, 2.2, np.nan, None, 'd', 'e', 'f'])

        # Run
        sdtype_datetime = instance._determine_sdtype_for_objects(data_datetime)
        sdtype_wrong_datetime = instance._determine_sdtype_for_objects(wrong_datetime)
        sdtype_categorical_small = instance._determine_sdtype_for_objects(data_categorical_small)
        sdtype_all_unique = instance._determine_sdtype_for_objects(data_all_unique)
        sdtype_categorical_large = instance._determine_sdtype_for_objects(data_categorical_large)
        sdtype_unknown = instance._determine_sdtype_for_objects(data_unknown)

        # Assert
        assert sdtype_datetime == 'datetime'
        assert sdtype_wrong_datetime == 'categorical'
        assert sdtype_categorical_small == 'categorical'
        assert sdtype_all_unique == 'id'
        assert sdtype_categorical_large == 'categorical'
        assert sdtype_unknown == 'unknown'

    @patch.object(pd.Series, 'sample')
    def test__determine_sdtype_for_objects_subsample_datetime(self, mock_sample):
        """Test the ``_determine_sdtype_for_objects`` method with large a datetime column."""
        # Setup
        instance = SingleTableMetadata()

        data_datetime = pd.Series(['2022-01-01'] * 15000)

        # Run
        instance._determine_sdtype_for_objects(data_datetime)

        # Assert
        mock_sample.assert_called_once_with(10000)

    def test__determine_sdtype_for_objects_silence_warning(self):
        """Test that UserWarning are silenced for ``_determine_sdtype_for_objects``."""
        # Setup
        instance = SingleTableMetadata()
        data = pd.Series(['warning1', 'warning2'])

        # Run
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            instance._determine_sdtype_for_objects(data)

        # Assert
        assert len(w) == 0

    def test__determine_sdtype_for_objects_with_none(self):
        """Test ``_determine_sdtype_for_objects`` with ``None`` in it."""
        # Setup
        instance = SingleTableMetadata()
        data = pd.Series([None] * 100)

        # Run
        sdtype = instance._determine_sdtype_for_objects(data)

        # Assert
        assert sdtype == 'categorical'

    def test__detect_columns(self):
        """Test the ``_detect_columns`` method."""
        # Setup
        instance = SingleTableMetadata()
        data = pd.DataFrame({
            'id': ['id1', 'id2', 'id3', 'id4', 'id5', 'id6', 'id7', 'id8', 'id9', 'id10', 'id11'],
            'numerical': [1, 2, 3, 2, 5, 6, 7, 8, 9, 10, 11],
            'datetime': [
                '2022-01-01',
                '2022-02-01',
                '2022-03-01',
                '2022-04-01',
                '2022-05-01',
                '2022-06-01',
                '2022-07-01',
                '2022-08-01',
                '2022-09-01',
                '2022-10-01',
                '2022-11-01',
            ],
            'alternate_id': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'alternate_id_string': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
            'categorical': ['a', 'b', 'a', 'a', 'b', 'b', 'a', 'b', 'a', 'b', 'a'],
            'bool': [True, False, True, False, True, False, True, False, True, False, True],
            'unknown': ['a', 'b', 'c', 'c', 1, 2.2, np.nan, None, 'd', 'e', 'f'],
            'first_name': [
                'John',
                'Jane',
                'John',
                'Jane',
                'John',
                'Jane',
                'John',
                'Jane',
                'John',
                'Jane',
                'John',
            ],
        })

        expected_datetime_format = '%Y-%m-%d'

        # Run
        instance._detect_columns(data)

        # Assert
        assert instance.columns['id']['sdtype'] == 'id'
        assert instance.columns['numerical']['sdtype'] == 'numerical'
        assert instance.columns['datetime']['sdtype'] == 'datetime'
        assert instance.columns['datetime']['datetime_format'] == expected_datetime_format
        assert instance.columns['alternate_id']['sdtype'] == 'unknown'
        assert instance.columns['alternate_id']['pii'] is True
        assert instance.columns['alternate_id_string']['sdtype'] == 'unknown'
        assert instance.columns['alternate_id_string']['pii'] is True
        assert instance.columns['categorical']['sdtype'] == 'categorical'
        assert instance.columns['unknown']['sdtype'] == 'unknown'
        assert instance.columns['unknown']['pii'] is True
        assert instance.columns['bool']['sdtype'] == 'categorical'
        assert instance.columns['first_name']['sdtype'] == 'first_name'
        assert instance.columns['first_name']['pii'] is True

        assert instance.primary_key == 'id'
        assert instance._updated is True

    def test__detect_columns_numerical_dtypes(self):
        """Test the ``_detect_columns`` method with numerical dtypes."""
        # Setup
        instance = SingleTableMetadata()
        data = pd.DataFrame({
            'Int8': pd.Series([1, 2, -3, pd.NA], dtype='Int8'),
            'Int16': pd.Series([1, 2, -3, pd.NA], dtype='Int16'),
            'Int32': pd.Series([1, 2, -3, pd.NA], dtype='Int32'),
            'Int64': pd.Series([1, 2, -3, pd.NA], dtype='Int64'),
            'UInt8': pd.Series([1, 2, 3, pd.NA], dtype='UInt8'),
            'UInt16': pd.Series([1, 2, 3, pd.NA], dtype='UInt16'),
            'UInt32': pd.Series([1, 2, 3, pd.NA], dtype='UInt32'),
            'UInt64': pd.Series([1, 2, 3, pd.NA], dtype='UInt64'),
            'Float32': pd.Series([1.1, 2.2, 3.3, pd.NA], dtype='Float32'),
            'Float64': pd.Series([1.1, 2.2, 3.3, pd.NA], dtype='Float64'),
            'uint8': np.array([1, 2, 3, 4], dtype='uint8'),
            'uint16': np.array([1, 2, 3, 4], dtype='uint16'),
            'uint32': np.array([1, 2, 3, 4], dtype='uint32'),
            'uint64': np.array([1, 2, 3, 4], dtype='uint64'),
        })

        # Run
        instance._detect_columns(data)

        # Assert
        for column in data.columns:
            assert instance.columns[column]['sdtype'] == 'numerical'

    def test__detect_columns_primary_key_detection(self):
        """Test the ``_detect_columns`` primary key detection."""
        # Setup
        metadata_without_primary_key_1 = SingleTableMetadata()
        metadata_without_primary_key_2 = SingleTableMetadata()
        metadata_with_primary_key = SingleTableMetadata()
        data_without_primary_key_1 = pd.DataFrame({
            'email': ['sdv@sdv.dev', 'info@datacebo.com', 'info@gmail.co.uk', None],
            'numerical': [0, 1, 2, 1],
        })  # Not primary key because has NaNs.

        data_without_primary_key_2 = pd.DataFrame({
            'email': ['sdv@sdv.dev', 'info@datacebo.com', 'info@gmail.co.uk', 'sdv@sdv.dev'],
            'numerical': [0, 1, 2, 1],
        })  # Not primary key because not unique.

        data_with_primary_key = pd.DataFrame({
            'email': ['sdv@sdv.dev', 'info@datacebo.com', 'info@gmail.co.uk'],
            'numerical': [0, 1, 2],
        })

        # Run
        metadata_with_primary_key._detect_columns(data_with_primary_key)
        metadata_without_primary_key_1._detect_columns(data_without_primary_key_1)
        metadata_without_primary_key_2._detect_columns(data_without_primary_key_2)

        # Assert
        assert metadata_with_primary_key.primary_key == 'email'
        assert metadata_without_primary_key_1.primary_key is None
        assert metadata_without_primary_key_2.primary_key is None

    def test__detect_columns_with_nans_nones_and_nats(self):
        """Test the ``_detect_columns`` with ``None``, ``np.nan`` and ``pd.NaT``."""
        # Setup
        data = pd.DataFrame({
            'cat': [None] * 100,
            'num': [np.nan] * 100,
            'num2': [float('nan')] * 100,
            'date': [pd.NaT] * 100,
        })
        stm = SingleTableMetadata()

        # Run
        stm._detect_columns(data)

        # Assert
        stm.columns['cat']['sdtype'] == 'categorical'
        stm.columns['num']['sdtype'] == 'numerical'
        stm.columns['num2']['sdtype'] == 'numerical'
        stm.columns['date']['sdtype'] == 'datetime'

    @patch('sdv.metadata.single_table._get_datetime_format')
    def test__detect_columns_with_error(self, mock__get_datetime_format):
        """Test the ``_detect_columns`` method with unsupported dtype."""
        # Setup
        instance = SingleTableMetadata()
        data = pd.DataFrame({
            'numerical': [1, 2, 3],
            'datetime': ['2022-01-01', '2022-02-01', '2022-03-01'],
        })

        non_supported_data = pd.DataFrame({
            'complex_dtype': [1 + 2j, 3 + 4j, 5 + 6j],
            'numerical': [1, 2, 3],
        })

        instance._determine_sdtype_for_numbers = Mock(return_value='numerical')
        instance._determine_sdtype_for_objects = Mock(return_value='datetime')

        # Run
        instance._detect_columns(data)

        expected_error_message = re.escape(
            "Unsupported data type for column 'complex_dtype' (kind: c)."
            " The valid data types are: 'object', 'int', 'float', 'datetime', 'bool'."
        )
        with pytest.raises(InvalidMetadataError, match=expected_error_message):
            instance._detect_columns(non_supported_data)

        # Assert
        args_numerical, _ = instance._determine_sdtype_for_numbers.call_args
        args_datetime, _ = instance._determine_sdtype_for_objects.call_args
        args_datetime_format, _ = mock__get_datetime_format.call_args

        pd.testing.assert_series_equal(args_numerical[0], data['numerical'])
        pd.testing.assert_series_equal(args_datetime[0], data['datetime'])
        pd.testing.assert_series_equal(args_datetime_format[0], data['datetime'])

        instance._determine_sdtype_for_numbers.assert_called_once()
        instance._determine_sdtype_for_objects.assert_called_once()
        mock__get_datetime_format.assert_called_once()

    def test__detect_columns_invalid_data_format(self):
        """Test the ``_detect_columns`` method with an invalid data format."""
        # Setup
        instance = SingleTableMetadata()
        dict_data = [
            {
                'key1': i,
                'key2': f'string_{i}',
                'key3': np.random.random(),  # random float
            }
            for i in range(100)
        ]
        data = pd.DataFrame({
            'dict_column': dict_data,
            'numerical': [1.2] * 100,
        })
        expected_error_message = re.escape(
            "Unable to detect metadata for column 'dict_column' due to an invalid data format."
            "\n TypeError: unhashable type: 'dict'"
        )

        # Run and Assert
        with pytest.raises(InvalidMetadataError, match=expected_error_message):
            instance._detect_columns(data)

    def test__detect_columns_without_infer_sdtypes(self):
        """Test the _detect_columns when infer_sdtypes is False."""
        # Setup
        instance = SingleTableMetadata()
        data = pd.DataFrame({
            'id': ['id1', 'id2', 'id3', 'id4', 'id5', 'id6', 'id7', 'id8', 'id9', 'id10', 'id11'],
            'numerical': [1, 2, 3, 2, 5, 6, 7, 8, 9, 10, 11],
            'datetime': [
                '2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01', '2022-05-01', '2022-06-01',
                '2022-07-01', '2022-08-01', '2022-09-01', '2022-10-01', '2022-11-01'
            ],
            'alternate_id': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'alternate_id_string': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
            'categorical': ['a', 'b', 'a', 'a', 'b', 'b', 'a', 'b', 'a', 'b', 'a'],
            'bool': [True, False, True, False, True, False, True, False, True, False, True],
            'unknown': ['a', 'b', 'c', 'c', 1, 2.2, np.nan, None, 'd', 'e', 'f'],
            'first_name': [
                'John', 'Jane', 'John', 'Jane', 'John', 'Jane',
                'John', 'Jane', 'John', 'Jane', 'John'
            ],
        })

        # Run
        instance._detect_columns(data, infer_sdtypes=False)

        # Assert
        assert instance.columns['id']['sdtype'] == 'unknown'
        assert instance.columns['numerical']['sdtype'] == 'unknown'
        assert instance.columns['datetime']['sdtype'] == 'unknown'
        assert instance.columns['alternate_id']['sdtype'] == 'unknown'
        assert instance.columns['alternate_id']['pii'] is True
        assert instance.columns['alternate_id_string']['sdtype'] == 'unknown'
        assert instance.columns['alternate_id_string']['pii'] is True
        assert instance.columns['categorical']['sdtype'] == 'unknown'
        assert instance.columns['unknown']['sdtype'] == 'unknown'
        assert instance.columns['unknown']['pii'] is True
        assert instance.columns['bool']['sdtype'] == 'unknown'
        assert instance.columns['first_name']['sdtype'] == 'unknown'
        assert instance.columns['first_name']['pii'] is True

        assert instance.primary_key is None
        assert instance._updated is True

    def test__detect_primary_key_missing_sdtypes(self):
        """The method should raise an error if not all sdtypes were detected."""
        # Setup
        data = pd.DataFrame({
            'string_id': ['1', '2', '3', '4', '5', '6'],
            'num_id': [1, 2, 3, 4, 5, 6],
        })
        metadata = SingleTableMetadata()
        metadata.columns = {'string_id': {'sdtype': 'id'}}

        # Run and Assert
        message = (
            'All columns must have sdtypes detected or set manually to detect the primary key.'
        )
        with pytest.raises(RuntimeError, match=message):
            metadata._detect_primary_key(data)

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
            'categorical': ['cat', 'dog', 'cat', np.nan],
            'date': pd.to_datetime(['2021-02-02', np.nan, '2021-03-05', '2022-12-09']),
            'int': [1, 2, 3, 4],
            'float': [1.0, 2.0, 3.0, 4],
            'bool': [np.nan, True, False, True],
        })

        # Run
        instance.detect_from_dataframe(data)

        # Assert
        assert instance.columns == {
            'categorical': {'sdtype': 'categorical'},
            'date': {'sdtype': 'datetime'},
            'int': {'sdtype': 'numerical'},
            'float': {'sdtype': 'numerical'},
            'bool': {'sdtype': 'categorical'},
        }

        expected_log_calls = [
            call('Detected metadata:'),
            call(json.dumps(instance.to_dict(), indent=4)),
        ]
        mock_log.info.assert_has_calls(expected_log_calls)

    @patch('sdv.metadata.single_table.LOGGER')
    def test_detect_from_dataframe_numerical_columns(self, mock_log):
        """Test the detect from dataframe with columns that are integers"""
        # Setup
        num_rows = 100
        num_cols = 20
        values = {i + 1: np.random.randint(0, 100, size=num_rows) for i in range(num_cols)}
        data = pd.DataFrame(values)
        correct_metadata = {
            'columns': {
                '1': {'sdtype': 'numerical'},
                '2': {'sdtype': 'numerical'},
                '3': {'sdtype': 'numerical'},
                '4': {'sdtype': 'numerical'},
                '5': {'sdtype': 'numerical'},
                '6': {'sdtype': 'numerical'},
                '7': {'sdtype': 'numerical'},
                '8': {'sdtype': 'numerical'},
                '9': {'sdtype': 'numerical'},
                '10': {'sdtype': 'numerical'},
                '11': {'sdtype': 'numerical'},
                '12': {'sdtype': 'numerical'},
                '13': {'sdtype': 'numerical'},
                '14': {'sdtype': 'numerical'},
                '15': {'sdtype': 'numerical'},
                '16': {'sdtype': 'numerical'},
                '17': {'sdtype': 'numerical'},
                '18': {'sdtype': 'numerical'},
                '19': {'sdtype': 'numerical'},
                '20': {'sdtype': 'numerical'},
            },
            'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1',
        }

        # Run
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data)

        # Assert
        assert correct_metadata == metadata.to_dict()

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
            'float': [1.0, 2.0, 3.0, 4],
            'bool': [np.nan, True, False, True],
        })

        # Run
        filepath = tmp_path / 'mydata.csv'
        data.to_csv(filepath, index=False)
        instance.detect_from_csv(filepath)

        # Assert
        assert instance.columns == {
            'categorical': {'sdtype': 'categorical'},
            'date': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
            'int': {'sdtype': 'numerical'},
            'float': {'sdtype': 'numerical'},
            'bool': {'sdtype': 'categorical'},
        }

        expected_log_calls = [
            call('Detected metadata:'),
            call(json.dumps(instance.to_dict(), indent=4)),
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
            'float': [1.0, 2.0, 3.0, 4],
            'bool': [np.nan, True, False, True],
        })

        # Run
        filepath = tmp_path / 'mydata.csv'
        data.to_csv(filepath, index=False)
        instance.detect_from_csv(filepath, read_csv_parameters={'parse_dates': ['date']})

        # Assert
        assert instance.columns == {
            'categorical': {'sdtype': 'categorical'},
            'date': {'sdtype': 'datetime'},
            'int': {'sdtype': 'numerical'},
            'float': {'sdtype': 'numerical'},
            'bool': {'sdtype': 'categorical'},
        }

        expected_log_calls = [
            call('Detected metadata:'),
            call(json.dumps(instance.to_dict(), indent=4)),
        ]
        mock_log.info.assert_has_calls(expected_log_calls)

    def test__validate_key_dataype_strings(self):
        """Test ``_validate_key_dataype`` for strings.

        Input:
            - A string

        Output:
            - True
        """
        # Setup
        instance = SingleTableMetadata()

        # Run
        out = instance._validate_key_datatype('10')

        # Assert
        assert out is True

    def test__validate_key_dataype_int(self):
        """Test ``_validate_key_dataype`` for invalid datatypes.

        Input:
            - A non-string

        Output:
            - False
        """
        # Setup
        instance = SingleTableMetadata()

        # Run
        out = instance._validate_key_datatype(10)

        # Assert
        assert out is False

    def test__validate_key_dataype_invalid_tuple(self):
        """Test ``_validate_key_dataype`` for tuples.

        Input:
            - A tuple with some strings

        Output:
            - False
        """
        # Setup
        instance = SingleTableMetadata()

        # Run
        out = instance._validate_key_datatype(('10', '20', '30'))

        # Assert
        assert out is False

    def test__validate_key_sequence_and_primary_key_same(self):
        """Test ``_validate_key`` for a column used as both sequence and primary keys."""
        # Setup
        instance_primary = SingleTableMetadata()
        instance_primary.primary_key = 'A'
        error_msg_primary = re.escape(
            'The column (A) cannot be set as sequence_key as it is already set as the primary_key.'
        )

        instance_sequence = SingleTableMetadata()
        instance_sequence.sequence_key = 'A'
        error_msg_sequence = re.escape(
            'The column (A) cannot be set as primary_key as it is already set as the sequence_key.'
        )

        # Run and Assert
        with pytest.raises(InvalidMetadataError, match=error_msg_primary):
            instance_primary._validate_key('A', 'sequence')
        with pytest.raises(InvalidMetadataError, match=error_msg_sequence):
            instance_sequence._validate_key('A', 'primary')

    def test_set_primary_key_validation_dtype(self):
        """Test that ``set_primary_key`` crashes for invalid arguments.

        Input:
            - A non-string value.

        Side Effect:
            - An ``InvalidMetadataError`` should be raised.
        """
        # Setup
        instance = SingleTableMetadata()

        err_msg = "'primary_key' must be a string."
        # Run / Assert
        with pytest.raises(InvalidMetadataError, match=err_msg):
            instance.set_primary_key(1)

    def test_set_primary_key_validation_columns(self):
        """Test that ``set_primary_key`` crashes for invalid arguments.

        Setup:
            - A ``SingleTableMetadata`` instance with ``_columns`` set.

        Input:
            - A column not present in ``_columns``.

        Side Effect:
            - An ``InvalidMetadataError`` should be raised.
        """
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {'a', 'd'}

        err_msg = (
            "Unknown primary key values {'b'}. Keys should be columns that exist in the table."
        )
        # Run / Assert
        with pytest.raises(InvalidMetadataError, match=err_msg):
            instance.set_primary_key('b')
            # NOTE: used to be ('a', 'b', 'd', 'c')

    @patch('sdv.metadata.single_table.is_faker_function')
    def test_set_primary_key_validation_categorical(self, mock_is_faker_function):
        """Test that ``set_primary_key`` crashes when its sdtype is categorical.

        Input:
            - A a key with a categorical sdtype.

        Side Effect:
            - An ``InvalidMetadataError`` should be raised.
        """
        # Setup
        mock_is_faker_function.return_value = False
        instance = SingleTableMetadata()
        instance.add_column('column1', sdtype='categorical')
        instance.add_column('column2', sdtype='categorical')
        instance.add_column('column3', sdtype='id')

        err_msg = re.escape("The primary_keys ['column1'] must be type 'id' or another PII type.")
        # Run / Assert
        with pytest.raises(InvalidMetadataError, match=err_msg):
            instance.set_primary_key('column1')

        mock_is_faker_function.assert_called_once_with('categorical')

    def test_set_primary_key(self):
        """Test that ``set_primary_key`` sets the ``_primary_key`` value."""
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {'column': {'sdtype': 'id'}}

        # Run
        instance.set_primary_key('column')

        # Assert
        assert instance.primary_key == 'column'

    def test_remove_primary_key(self):
        """Test that ``remove_primary_key`` removes the ``primary_key`` value."""
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {'id': {'sdtype': 'id'}}
        instance.primary_key = 'id'

        # Run
        instance.remove_primary_key()

        # Assert
        assert instance.primary_key is None

    @patch('sdv.metadata.single_table.warnings')
    def test_remove_primary_key_warns_no_key_set(self, warning_mock):
        """Test that ``remove_primary_key`` removes the ``primary_key`` value."""
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {'id': {'sdtype': 'id'}}

        # Run
        instance.remove_primary_key()

        # Assert
        assert instance.primary_key is None
        warning_mock.warn.assert_called_once_with('No primary key exists to remove.')

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
        warning_mock.warn.assert_called_once_with(warning_msg)
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
        alternate_key_warning_msg = (
            "'column1' is currently set as an alternate key and will be removed from that list."
        )
        primary_key_warning_msg = (
            "There is an existing primary key 'column0'. This key will be removed."
        )
        warning_mock.warn.assert_has_calls([
            call(alternate_key_warning_msg),
            call(primary_key_warning_msg),
        ])
        assert instance.primary_key == 'column1'
        assert instance.alternate_keys == ['column2']

    def test_set_sequence_key_validation_dtype(self):
        """Test that ``set_sequence_key`` crashes for invalid arguments.

        Input:
            - A non-string value.

        Side Effect:
            - An ``InvalidMetadataError`` should be raised.
        """
        # Setup
        instance = SingleTableMetadata()

        err_msg = "'sequence_key' must be a string."
        # Run / Assert
        with pytest.raises(InvalidMetadataError, match=err_msg):
            instance.set_sequence_key(1)

    def test_set_sequence_key_validation_columns(self):
        """Test that ``set_sequence_key`` crashes for invalid arguments.

        Setup:
            - A ``SingleTableMetadata`` instance with ``_columns`` set.

        Input:
            - A column not present in ``_columns``.

        Side Effect:
            - An ``InvalidMetadataError`` should be raised.
        """
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {'a', 'd'}

        err_msg = (
            "Unknown sequence key values {'b'}. Keys should be columns that exist in the table."
        )
        # Run / Assert
        with pytest.raises(InvalidMetadataError, match=err_msg):
            instance.set_sequence_key('b')
            # NOTE: used to be ('a', 'b', 'd', 'c')

    @patch('sdv.metadata.single_table.is_faker_function')
    def test_set_sequence_key_validation_categorical(self, mock_is_faker_function):
        """Test that ``set_sequence_key`` crashes when its sdtype is categorical.

        Input:
            - A key with categorical sdtype.

        Side Effect:
            - An ``InvalidMetadataError`` should be raised.
        """
        # Setup
        mock_is_faker_function.return_value = False
        instance = SingleTableMetadata()
        instance.add_column('column1', sdtype='categorical')
        instance.add_column('column2', sdtype='categorical')
        instance.add_column('column3', sdtype='id')

        err_msg = re.escape("The sequence_keys ['column1'] must be type 'id' or another PII type.")
        # Run / Assert
        with pytest.raises(InvalidMetadataError, match=err_msg):
            instance.set_sequence_key('column1')

        mock_is_faker_function.assert_called_once_with('categorical')

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
        """Test that ``set_sequence_key`` errors for tuples."""
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {'col1': {'sdtype': 'id'}, 'col2': {'sdtype': 'id'}}

        # Run and Assert
        msg = "'sequence_key' must be a string."
        with pytest.raises(InvalidMetadataError, match=msg):
            instance.set_sequence_key(('col1', 'col2'))

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
        warning_mock.warn.assert_called_once_with(warning_msg)
        assert instance.sequence_key == 'column1'

    def test_add_alternate_keys_validation_dtype(self):
        """Test that ``add_alternate_keys`` crashes for invalid arguments.

        Input:
            - A list with non-string values.

        Side Effect:
            - An ``InvalidMetadataError`` should be raised.
        """
        # Setup
        instance = SingleTableMetadata()

        err_msg = "'alternate_keys' must be a list of strings."
        # Run / Assert
        with pytest.raises(InvalidMetadataError, match=err_msg):
            instance.add_alternate_keys(['col1', ('1', 2, '3'), 'col3'])

    def test_add_alternate_keys_validation_columns(self):
        """Test that ``add_alternate_keys`` crashes for invalid arguments.

        Setup:
            - A ``SingleTableMetadata`` instance with ``_columns`` set.

        Input:
            - A list with unknown key column.

        Side Effect:
            - An ``InvalidMetadataError`` should be raised.
        """
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {'abc', '213', '312'}

        err_msg = (
            "Unknown alternate key values {'123'}. Keys should be columns that exist in the table."
        )
        # Run / Assert
        with pytest.raises(InvalidMetadataError, match=err_msg):
            instance.add_alternate_keys(['abc', '123'])
            # NOTE: used to be ['abc', ('123', '213', '312'), 'bca']

    @patch('sdv.metadata.single_table.is_faker_function')
    def test_add_alternate_keys_validation_categorical(self, mock_is_faker_function):
        """Test that ``add_alternate_keys`` crashes when its sdtype is categorical.

        Input:
            - A list of keys, some of which have sdtype categorical.

        Side Effect:
            - An ``InvalidMetadataError`` should be raised.
        """
        # Setup
        mock_is_faker_function.return_value = False
        instance = SingleTableMetadata()
        instance.add_column('column1', sdtype='categorical')
        instance.add_column('column2', sdtype='categorical')
        instance.add_column('column3', sdtype='id')

        err_msg = re.escape(
            "The alternate_keys ['column1', 'column2'] must be type 'id' or another PII type."
        )
        # Run / Assert
        with pytest.raises(InvalidMetadataError, match=err_msg):
            instance.add_alternate_keys(['column1', 'column2', 'column3'])

        mock_is_faker_function.assert_has_calls([call('categorical'), call('categorical')])

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
            'column3': {'sdtype': 'id'},
        }

        # Run
        instance.add_alternate_keys(['column1', 'column2', 'column3'])

        # Assert
        assert instance.alternate_keys == ['column1', 'column2', 'column3']

    @patch('sdv.metadata.single_table.warnings')
    def test_add_alternate_keys_duplicate(self, warnings_mock):
        """Test that the method does not add columns that are already in ``_alternate_keys``."""
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {
            'column1': {'sdtype': 'id'},
            'column2': {'sdtype': 'id'},
            'column3': {'sdtype': 'id'},
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
        instance.columns = {'a': {'sdtype': 'numerical'}, 'd': {'sdtype': 'categorical'}}

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

    def test__validate_column_relationship(self):
        """Test the ``_validate_column_relationship`` method."""
        # Setup
        instance = SingleTableMetadata()
        mock_relationship_validation = Mock()
        instance._COLUMN_RELATIONSHIP_TYPES = {'mock_relationship': mock_relationship_validation}
        relationship = {'type': 'mock_relationship', 'column_names': ['a', 'b']}
        instance.columns = {
            'a': {'sdtype': 'categorical'},
            'b': {'sdtype': 'numerical'},
            'c': {'sdtype': 'datetime'},
        }

        # Run
        instance._validate_column_relationship(relationship)

        # Assert
        expected_columns_to_sdtypes = {
            'a': 'categorical',
            'b': 'numerical',
        }
        mock_relationship_validation.assert_called_once_with(expected_columns_to_sdtypes)

    def test__validate_column_relationship_bad_relationship_type(self):
        """Test validation fails for an unknown relationship type."""
        # Setup
        instance = SingleTableMetadata()
        instance._COLUMN_RELATIONSHIP_TYPES = {'mock_relationship': Mock()}
        relationship = {'type': 'bad_relationship_type', 'column_names': ['a', 'b']}

        # Run and Assert
        msg = re.escape(
            "Unknown column relationship type 'bad_relationship_type'. "
            "Must be one of ['mock_relationship']."
        )
        with pytest.raises(InvalidMetadataError, match=msg):
            instance._validate_column_relationship(relationship)

    def test__validate_column_relationship_bad_columns(self):
        """Test validation fails for invalid columns."""

        # Setup
        def validation_side_effect(*args, **kwargs):
            raise InvalidMetadataError("Columns ['a', 'b'] have unsupported sdtype.")

        instance = SingleTableMetadata()
        mock_relationship_validation = Mock()
        mock_relationship_validation.side_effect = validation_side_effect
        instance._COLUMN_RELATIONSHIP_TYPES = {'mock_relationship': mock_relationship_validation}
        relationship = {'type': 'mock_relationship', 'column_names': ['a', 'b', 'c', 'x']}
        instance.columns = {
            'a': {'sdtype': 'id'},
            'b': {'sdtype': 'categorical'},
            'c': {'sdtype': 'numerical'},
        }
        instance.primary_key = 'a'

        # Run
        err_msg = re.escape(
            "Cannot use primary key 'a' in column relationship.\n"
            "Column 'x' not in metadata.\n"
            "Columns ['a', 'b'] have unsupported sdtype."
        )
        with pytest.raises(InvalidMetadataError, match=err_msg):
            instance._validate_column_relationship(relationship)

        # Assert
        expected_columns_to_sdtypes = {'a': 'id', 'b': 'categorical', 'c': 'numerical', 'x': None}
        mock_relationship_validation.assert_called_once_with(expected_columns_to_sdtypes)

    def test__validate_column_relationship_with_other_relationships(self):
        """Test ``_validate_column_relationship_with_others``."""
        # Setup
        instance = SingleTableMetadata()
        column_relationships = [
            {'type': 'relationship_one', 'column_names': ['a', 'b']},
        ]
        relationship_valid = {'type': 'relationship_two', 'column_names': ['c', 'd']}
        relationship_invalid = {'type': 'relationship_two', 'column_names': ['b', 'e']}

        # Run and Assert
        instance._validate_column_relationship_with_others(relationship_valid, column_relationships)
        expected_message = re.escape(
            "Columns 'b' is already part of a relationship of type"
            " 'relationship_one'. Columns cannot be part of multiple relationships."
        )
        with pytest.raises(InvalidMetadataError, match=expected_message):
            instance._validate_column_relationship_with_others(
                relationship_invalid, column_relationships
            )

    def test__validate_all_column_relationships(self):
        """Test ``_validate_all_column_relationships`` method."""
        # Setup
        instance = SingleTableMetadata()
        mock_validate_relationship = Mock()
        instance._validate_column_relationship = mock_validate_relationship
        relationship_one = {'type': 'relationship_one', 'column_names': ['a', 'b']}
        relationship_two = {'type': 'relationship_two', 'column_names': ['c', 'd']}
        column_relationships = [relationship_one, relationship_two]

        # Run
        instance._validate_all_column_relationships(column_relationships)

        # Assert
        mock_validate_relationship.assert_has_calls([
            call(relationship_one),
            call(relationship_two),
        ])

    def test__validate_all_column_relationships_invalid_relationship_structure(self):
        """Test validation fails if relationship is malformed."""
        # Setup
        instance = SingleTableMetadata()
        mock_validate_relationship = Mock()
        instance._validate_column_relationship = mock_validate_relationship
        column_relationships = [
            {'type': 'relationship_one', 'column_names': ['a', 'b']},
            {'type': 'relationship_two', 'bad_key': ['c', 'd']},
        ]

        # Run and Assert
        err_msg = re.escape("Relationship has invalid keys {'bad_key'}.")
        with pytest.raises(InvalidMetadataError, match=err_msg):
            instance._validate_all_column_relationships(column_relationships)

    def test__validate_all_column_relationships_repeated_column(self):
        """Test validation fails if columns are repeated across column relationships."""
        # Setup
        instance = SingleTableMetadata()
        mock_validate_relationship = Mock()
        instance._validate_column_relationship = mock_validate_relationship
        column_relationships = [
            {'type': 'relationship_one', 'column_names': ['a', 'b']},
            {'type': 'relationship_two', 'column_names': ['b', 'c']},
        ]
        instance.column_relationships = column_relationships
        # Run and Assert
        expected_message = re.escape(
            "Columns 'b' is already part of a relationship of type 'relationship_two'."
            ' Columns cannot be part of multiple relationships.'
        )
        with pytest.raises(InvalidMetadataError, match=expected_message):
            instance._validate_all_column_relationships(column_relationships)

    def test__validate_all_column_relationships_bad_relationship(self):
        """Test validation fails if individual relationship validation fails."""

        # Setup
        def mock_relationship_validate(relationship):
            raise InvalidMetadataError(f"Error in '{relationship['type']}' relationship.")

        instance = SingleTableMetadata()
        mock_validate_relationship = Mock()
        mock_validate_relationship.side_effect = mock_relationship_validate
        instance._validate_column_relationship = mock_validate_relationship
        column_relationships = [
            {'type': 'relationship_one', 'column_names': ['a', 'b']},
            {'type': 'relationship_two', 'column_names': ['c', 'd']},
        ]

        # Run and Assert
        err_msg = re.escape(
            "Error in 'relationship_one' relationship.\nError in 'relationship_two' relationship."
        )
        with pytest.raises(InvalidMetadataError, match=err_msg):
            instance._validate_all_column_relationships(column_relationships)

    def test_add_column_relationships(self):
        """Test ``add_column_relationship`` adds a column relationship."""
        # Setup
        instance = SingleTableMetadata()
        mock_validate_column_relationships = Mock()
        instance._validate_all_column_relationships = mock_validate_column_relationships

        # Run
        instance.add_column_relationship(
            relationship_type='relationship_A', column_names=['colA', 'colB']
        )
        instance.add_column_relationship(
            relationship_type='relationship_B', column_names=['col1', 'col2', 'col3']
        )
        # Assert
        mock_validate_column_relationships.assert_has_calls([
            call([{'type': 'relationship_A', 'column_names': ['colA', 'colB']}]),
            call([
                {'type': 'relationship_B', 'column_names': ['col1', 'col2', 'col3']},
                {'type': 'relationship_A', 'column_names': ['colA', 'colB']},
            ]),
        ])
        assert instance.column_relationships == [
            {'type': 'relationship_A', 'column_names': ['colA', 'colB']},
            {'type': 'relationship_B', 'column_names': ['col1', 'col2', 'col3']},
        ]

    def test_add_column_relationships_silence_warnings(self):
        """Test ``add_column_relationship`` silences UserWarnings."""

        # Setup
        def raise_user_warning(*args, **kwargs):
            warnings.warn('This is a warning', UserWarning)

        instance = SingleTableMetadata()
        mock_validate_column_relationships = Mock(side_effect=raise_user_warning)
        instance._validate_all_column_relationships = mock_validate_column_relationships

        # Run
        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter('always')
            instance.add_column_relationship(
                relationship_type='relationship_A', column_names=['colA', 'colB']
            )

        # Assert
        assert len(captured_warnings) == 0

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
        instance.column_relationships = [
            {'type': 'relationship_one', 'column_names': ['col1', 'col2']},
        ]
        instance._validate_key = Mock()
        instance._validate_alternate_keys = Mock()
        instance._validate_sequence_index = Mock()
        instance._validate_sequence_index_not_in_sequence_key = Mock()
        instance._validate_all_column_relationships = Mock()
        instance._validate_column_args = Mock(side_effect=InvalidMetadataError('column_error'))

        err_msg = re.escape(
            'The following errors were found in the metadata:\n\ncolumn_error\ncolumn_error'
        )
        # Run
        with pytest.raises(InvalidMetadataError, match=err_msg):
            instance.validate()

        # Assert
        instance._validate_key.assert_has_calls([
            call(instance.primary_key, 'primary'),
            call(instance.sequence_key, 'sequence'),
        ])
        instance._validate_column_args.assert_has_calls([
            call('col1', sdtype='numerical'),
            call('col2', sdtype='numerical'),
        ])
        instance._validate_alternate_keys.assert_called_once_with(instance.alternate_keys)
        instance._validate_sequence_index.assert_called_once_with(instance.sequence_index)
        instance._validate_sequence_index_not_in_sequence_key.assert_called_once()
        instance._validate_all_column_relationships.assert_called_once_with([
            {'type': 'relationship_one', 'column_names': ['col1', 'col2']}
        ])

    def test_validate_data_wrong_type(self):
        """Test error is raised if data is not ``pd.DataFrame``."""
        # Setup
        data = np.ndarray([])
        metadata = SingleTableMetadata()

        # Run and Assert
        err_msg = "Data must be a DataFrame, not a <class 'numpy.ndarray'>."
        with pytest.raises(ValueError, match=err_msg):
            metadata.validate_data(data)

    def test_validate_data_data_columns_in_empty_metadata(self):
        """Test error is raised if data is passed and metadata is empty."""
        # Setup
        data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
        })
        metadata = SingleTableMetadata()

        # Run and Assert
        err_msg = re.escape(
            'The provided data does not match the metadata:'
            "\nThe columns ['col1', 'col2'] are not present in the metadata."
        )
        with pytest.raises(InvalidDataError, match=err_msg):
            metadata.validate_data(data)

    def test_validate_data_data_columns_in_metadata(self):
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

        # Run and Assert
        err_msg = re.escape(
            'The provided data does not match the metadata:'
            "\nThe columns ['col2', 'col3'] are not present in the metadata."
            '\n'
            "\nThe metadata columns ['col4', 'col5'] are not present in the data."
        )
        with pytest.raises(InvalidDataError, match=err_msg):
            metadata.validate_data(data)

    def test_validate_data_keys_with_missing_values(self):
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
            'ak_col3': [0, 1, 2],
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
        metadata.set_sequence_key('sk_col1')
        metadata.add_alternate_keys(['ak_col1', 'ak_col2', 'ak_col3'])

        # Run and Assert
        err_msg = re.escape(
            'The provided data does not match the metadata:'
            "\nKey column 'ak_col1' contains missing values."
            '\n'
            "\nKey column 'ak_col2' contains missing values."
            '\n'
            "\nKey column 'pk_col' contains missing values."
            '\n'
        )
        with pytest.raises(InvalidDataError, match=err_msg):
            metadata.validate_data(data)

    def test_validate_data_keys_with_missing_with_single_sequence_key(self):
        """Test error is raised if keys contain missing values.

        Test the case with a single sequence key.
        """
        data = pd.DataFrame({'pk_col': [1], 'sk_col': [None]})
        metadata = SingleTableMetadata()
        metadata.add_column('pk_col', sdtype='id')
        metadata.add_column('sk_col', sdtype='id')
        metadata.set_primary_key('pk_col')
        metadata.set_sequence_key('sk_col')

        # Run and Assert
        err_msg = re.escape(
            'The provided data does not match the metadata:'
            "\nKey column 'sk_col' contains missing values."
        )
        with pytest.raises(InvalidDataError, match=err_msg):
            metadata.validate_data(data)

    def test_validate_data_keys_not_unique(self):
        """Test error is raised if primary or alternate keys are not unique."""
        data = pd.DataFrame({
            'pk_col': [0, 1, 1, 0, 2],
            'ak_col1': [0, 1, 0, 3, 3],
            'ak_col2': [2, 2, 2, 2, 2],
            'ak_col3': [0, 1, 2, 3, 4],
        })
        metadata = SingleTableMetadata()
        metadata.add_column('pk_col', sdtype='id')
        metadata.add_column('ak_col1', sdtype='id')
        metadata.add_column('ak_col2', sdtype='id')
        metadata.add_column('ak_col3', sdtype='id')
        metadata.set_primary_key('pk_col')
        metadata.add_alternate_keys(['ak_col1', 'ak_col2', 'ak_col3'])

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
            metadata.validate_data(data)

    def test_validate_data_empty(self):
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

        # Run
        metadata.validate_data(data)

    def test_validate_data_no_keys(self):
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

        # Run
        metadata.validate_data(data)

    def test_validate_data_empty_dataframe(self):
        """Test method doesn't raise when data is an empty dataframe."""
        data = pd.DataFrame()
        metadata = SingleTableMetadata()

        # Run
        metadata.validate_data(data)

    def test_validate_data_sdtypes(self):
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
            metadata.validate_data(data)

    def test_validate_data_datetime_sdtype(self):
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
                '20220929111311000000',
            ],
            'date_int': [
                20220902110443000000,
                20220916230356000000,
                20220826173917000000,
                20220826212135000000,
                20220929111311000000,
            ],
            'bad_date': [
                2022090,
                20220916230356000000,
                2022,
                20220826212135000000,
                20220929111311000000,
            ],
        })
        metadata = SingleTableMetadata()
        metadata.add_column('date_str', sdtype='datetime', datetime_format='%Y%m%d%H%M%S%f')
        metadata.add_column('date_int', sdtype='datetime', datetime_format='%Y%m%d%H%M%S%f')
        metadata.add_column('bad_date', sdtype='datetime', datetime_format='%Y%m%d%H%M%S%f')

        # Run and Assert
        err_msg = re.escape(
            'The provided data does not match the metadata:'
            "\nInvalid values found for datetime column 'bad_date': [2022, 2022090]."
        )
        with pytest.raises(InvalidDataError, match=err_msg):
            metadata.validate_data(data)

    def test_validate_data_datetime_warning(self):
        """Test validation for columns with datetime.

        If the datetime format is not provided, a warning should be shwon if the ``dtype`` is
        object.
        """
        # Setup
        data = pd.DataFrame({
            'warning_date_str': [
                '2022-09-02',
                '2022-09-16',
                '2022-08-26',
                '2022-08-26',
                '2022-09-29',
            ],
            'valid_date': [
                '20220902110443000000',
                '20220916230356000000',
                '20220826173917000000',
                '20220826212135000000',
                '20220929111311000000',
            ],
            'datetime': pd.to_datetime([
                '20220902',
                '20220916',
                '20220826',
                '20220826',
                '20220929',
            ]),
        })
        metadata = SingleTableMetadata()
        metadata.add_column('warning_date_str', sdtype='datetime')
        metadata.add_column('valid_date', sdtype='datetime', datetime_format='%Y%m%d%H%M%S%f')
        metadata.add_column('datetime', sdtype='datetime')

        # Run and Assert
        warning_frame = pd.DataFrame({
            'Column Name': ['warning_date_str'],
            'sdtype': ['datetime'],
            'datetime_format': [None],
        })
        warning_msg = (
            "No 'datetime_format' is present in the metadata for the following columns:\n"
            f'{warning_frame.to_string(index=False)}\n'
            'Without this specification, SDV may not be able to accurately parse the data. '
            "We recommend adding datetime formats using 'update_column'."
        )
        with pytest.warns(UserWarning, match=warning_msg):
            metadata.validate_data(data)

    def test_validate_data(self):
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
        metadata.set_sequence_key('sk_col1')
        metadata.add_alternate_keys(['ak_col1', 'ak_col2'])

        # Run
        metadata.validate_data(data)

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
            'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1',
        }

        # Ensure that the output object does not alterate the inside object
        result['columns']['my_column'] = 1
        assert instance.columns['my_column'] == 'value'

    def test_to_dict_missing_attributes(self):
        """Test when the class is missing a new attribute.

        If the metadata class was saved on previous versions, it may
        be missing attributes so we should still be able to convert
        that old metadata to a dict.
        """
        # Setup
        instance = SingleTableMetadata()
        instance.columns['my_column'] = 'value'
        del instance.column_relationships

        # Run
        result = instance.to_dict()

        # Assert
        assert result == {
            'columns': {'my_column': 'value'},
            'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1',
        }

    def test_load_from_dict(self):
        """Test that ``load_from_dict`` returns a instance with the ``dict`` updated objects."""
        # Setup
        my_metadata = {
            'columns': {'my_column': 'value'},
            'primary_key': 'pk',
            'alternate_keys': [],
            'sequence_key': None,
            'sequence_index': None,
            'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1',
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

    def test_load_from_dict_integer(self):
        """Test that ``load_from_dict`` returns a instance with the ``dict`` updated objects.

        If the metadata dict contains columns with integers for certain reasons
        (e.g. due to missing column names from CSV) make sure they are correctly typed
        to strings to ensure metadata is parsed properly.
        """
        # Setup
        my_metadata = {
            'columns': {1: 'value'},
            'primary_key': 'pk',
            'alternate_keys': [],
            'sequence_key': None,
            'sequence_index': None,
            'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1',
        }

        # Run
        instance = SingleTableMetadata.load_from_dict(my_metadata)

        # Assert
        assert instance.columns == {'1': 'value'}
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
            'columns': {'animals': {'type': 'categorical'}},
            'primary_key': 'animals',
        }

        # Run / Assert
        error_msg = (
            'This metadata file is incompatible with the ``SingleTableMetadata`` class and version.'
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
            'columns': {'animals': {'type': 'categorical'}},
            'primary_key': 'animals',
            'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1',
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

    @patch('sdv.metadata.single_table.datetime')
    def test_save_to_json(self, mock_datetime, tmp_path, caplog):
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
        mock_datetime.now.return_value = '2024-04-19 16:20:10.037183'
        instance = SingleTableMetadata()

        # Run
        file_name = tmp_path / 'singletable.json'
        with catch_sdv_logs(caplog, logging.INFO, logger='SingleTableMetadata'):
            instance.save_to_json(file_name)

        # Assert
        assert caplog.messages[0] == (
            '\nMetadata Save:\n'
            '  Timestamp: 2024-04-19 16:20:10.037183\n'
            '  Statistics about the metadata:\n'
            '    Total number of tables: 1'
            '    Total number of columns: 0'
            '    Total number of relationships: 0'
        )
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

    def test_visualize_with_invalid_input(self):
        """Test that a ``ValueError`` is being raised when ``show_table_details`` is incorrect."""
        # Setup
        instance = SingleTableMetadata()

        # Run and Assert
        error_msg = "'show_table_details' should be 'full' or 'summarized'."
        with pytest.raises(ValueError, match=error_msg):
            instance.visualize(None)

    @patch('sdv.metadata.single_table.visualize_graph')
    def test_visualize_metadata_full(self, mock_visualize_graph):
        """Test the ``visualize`` method when ``show_table_details`` is 'full'."""
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {
            'name': {'sdtype': 'categorical'},
            'age': {'sdtype': 'numerical'},
            'start_date': {'sdtype': 'datetime'},
            'phrase': {'sdtype': 'id'},
        }

        # Run
        result = instance.visualize('full')

        # Assert
        assert result == mock_visualize_graph.return_value
        expected_node = {
            '': '{name : categorical\\lage : numerical\\lstart_date : datetime\\lphrase : id\\l}'
        }
        mock_visualize_graph.assert_called_once_with(expected_node, [], None)

    @patch('sdv.metadata.single_table.visualize_graph')
    def test_visualize_metadata_summarized(self, mock_visualize_graph):
        """Test the ``visualize`` method when ``show_table_details`` is 'summarized'."""
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {
            'name': {'sdtype': 'categorical'},
            'age': {'sdtype': 'numerical'},
            'start_date': {'sdtype': 'datetime'},
            'phrase': {'sdtype': 'id'},
        }

        # Run
        result = instance.visualize('summarized')

        # Assert
        assert result == mock_visualize_graph.return_value
        node = (
            '{Columns\\l&nbsp; &nbsp; • categorical : 1\\l&nbsp; &nbsp; • datetime : 1\\l&nbsp; '
            '&nbsp; • id : 1\\l&nbsp; &nbsp; • numerical : 1\\l}'
        )
        expected_node = {'': node}
        mock_visualize_graph.assert_called_once_with(expected_node, [], None)

    @patch('sdv.metadata.single_table.visualize_graph')
    def test_visualize_metadata_with_primary_alternate_and_sequence_keys(
        self, mock_visualize_graph
    ):
        """Test the ``visualize`` method when there are primary, alternate and sequence keys."""
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {
            'name': {'sdtype': 'categorical'},
            'timestamp': {'sdtype': 'datetime'},
            'age': {'sdtype': 'numerical'},
            'start_date': {'sdtype': 'datetime'},
            'phrase': {'sdtype': 'id'},
            'passport': {'sdtype': 'id'},
        }
        instance.primary_key = 'passport'
        instance.alternate_keys = ['phrase', 'name']
        instance.sequence_key = 'timestamp'
        instance.sequence_index = 'start_date'

        # Run
        result = instance.visualize('full')

        # Assert
        assert result == mock_visualize_graph.return_value
        node = (
            '{name : categorical\\ltimestamp : datetime\\lage : numerical\\l'
            'start_date : datetime\\lphrase : id\\lpassport : id\\l|'
            'Primary key: passport\\lSequence key: timestamp\\lSequence index: start_date\\l'
            'Alternate keys:\\l &nbsp; &nbsp; • phrase\\l&nbsp; &nbsp; • name\\l}'
        )
        expected_node = {'': node}
        mock_visualize_graph.assert_called_once_with(expected_node, [], None)

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
    def test_upgrade_metadata_multiple_tables(self, from_dict_mock, convert_mock, read_json_mock):
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
        read_json_mock.return_value = {'tables': {'table': {'columns': {}}}}

        # Run
        SingleTableMetadata.upgrade_metadata('old')

        # Assert
        convert_mock.assert_called_once_with({'columns': {}})
        new_metadata.validate.assert_called_once()

    @patch('sdv.metadata.single_table.read_json')
    @patch('sdv.metadata.single_table.convert_metadata')
    @patch('sdv.metadata.single_table.SingleTableMetadata.load_from_dict')
    def test_upgrade_metadata_multiple_tables_fails(
        self, from_dict_mock, convert_mock, read_json_mock
    ):
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
        read_json_mock.return_value = {'tables': {'table1': {'columns': {}}, 'table2': {}}}

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
        self, from_dict_mock, convert_mock, read_json_mock, warnings_mock
    ):
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

    def test_anonymize(self):
        """Test the ``anonymize`` method."""
        # Setup
        instance = SingleTableMetadata()
        instance.columns = {
            'real_column1': {'sdtype': 'id', 'regex_format': r'\d{30}'},
            'real_column2': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
            'real_column3': {'sdtype': 'numerical'},
            'real_column4': {'sdtype': 'id'},
        }
        instance.primary_key = 'real_column1'
        instance.alternate_keys = ['real_column4']
        instance.sequence_index = 'real_column2'
        instance.sequence_key = 'real_column4'

        # Run
        anonymized = instance.anonymize()

        # Assert
        anonymized.validate()

        assert all(original_col not in anonymized.columns for original_col in instance.columns)
        for original_col, anonymized_col in instance._anonymized_column_map.items():
            assert instance.columns[original_col] == anonymized.columns[anonymized_col]

        anon_primary_key = anonymized.primary_key
        assert anonymized.columns[anon_primary_key] == instance.columns['real_column1']

        anon_alternate_keys = anonymized.alternate_keys
        assert anonymized.columns[anon_alternate_keys[0]] == instance.columns['real_column4']

        anon_sequence_index = anonymized.sequence_index
        assert anonymized.columns[anon_sequence_index] == instance.columns['real_column2']

        anon_sequence_key = anonymized.sequence_key
        assert anonymized.columns[anon_sequence_key] == instance.columns['real_column4']

        assert anon_alternate_keys[0] == anon_sequence_key
