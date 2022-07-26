"""Test Single Table Metadata."""

import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from sdv.metadata.single_table import SingleTableMetadata


class TestSingleTableMetadata:
    """Test ``SingleTableMetadata`` class."""

    VALID_KWARGS = [
        ('age', 'numerical', {}),
        ('age', 'numerical', {'representation': 'int'}),
        ('start_date', 'datetime', {}),
        ('start_date', 'datetime', {'datetime_format': '%Y-%d'}),
        ('name', 'categorical', {}),
        ('name', 'categorical', {'order_by': 'alphabetical'}),
        ('name', 'categorical', {'order': ['a', 'b', 'c']}),
        ('synthetic', 'boolean', {}),
        ('phrase', 'text', {}),
        ('phrase', 'text', {'regex_format': '[A-z]'}),
        ('phone', 'phone_number', {}),
        ('phone', 'phone_number', {'pii': True}),
    ]

    INVALID_KWARGS = [
        ('age', 'numerical', {'representation': 'int', 'datetime_format': None, 'pii': True},
         re.escape("Invalid values '(datetime_format, pii)' for numerical column 'age'."),),
        ('start_date', 'datetime', {'datetime_format': '%Y-%d', 'pii': True},
         re.escape("Invalid values '(pii)' for datetime column 'start_date'.")),
        ('name', 'categorical',
         {'pii': True, 'ordering': ['a', 'b'], 'ordered': 'numerical_values'},
         re.escape("Invalid values '(ordered, ordering, pii)' for categorical column 'name'.")),
        ('synthetic', 'boolean', {'pii': True},
         re.escape("Invalid values '(pii)' for boolean column 'synthetic'.")),
        ('phrase', 'text', {'regex_format': '[A-z]', 'pii': True, 'anonymization': True},
         re.escape("Invalid values '(anonymization, pii)' for text column 'phrase'.")),
        ('phone', 'phone_number', {'anonymization': True, 'order_by': 'phone_number'},
         re.escape("Invalid values '(anonymization, order_by)' for phone_number column 'phone'."))
    ]

    def test___init__(self):
        """Test creating an instance of ``SingleTableMetadata``."""
        # Run
        instance = SingleTableMetadata()

        # Assert
        assert instance._columns == {}
        assert instance._constraints == []
        assert instance._version == 'SINGLE_TABLE_V1'
        assert instance._metadata == {
            'columns': {},
            'primary_key': None,
            'alternate_keys': [],
            'sequence_key': None,
            'sequence_index': None,
            'constraints': [],
            'SCHEMA_VERSION': 'SINGLE_TABLE_V1'
        }

    def test__validate_numerical_default_and_invalid(self):
        """Test the ``_validate_numerical`` method.

        Setup:
            - instance of ``SingleTableMetadata``
            - list of accepted representations.

        Input:
            - Column name.
            - sdtype numerical
            - representation

        Side Effects:
            - Passes when no ``representation`` is provided
            - ``ValueError`` is raised stating that the ``representation`` is not supported.
        """
        # Setup
        instance = SingleTableMetadata()

        # Run / Assert
        instance._validate_numerical('age')

        error_msg = re.escape("Invalid value for 'representation' 36 for column 'age'.")
        with pytest.raises(ValueError, match=error_msg):
            instance._validate_numerical('age', representation=36)

    @pytest.mark.parametrize('representation', SingleTableMetadata._NUMERICAL_REPRESENTATIONS)
    def test__validate_numerical_representations(self, representation):
        """Test the ``_validate_numerical`` method.

        Setup:
            - instance of ``SingleTableMetadata``
            - list of accepted representations.

        Input:
            - Column name.
            - sdtype numerical
            - representation

        Side Effects:
            - Passes with the correct ``representation``
            - ``ValueError`` is raised stating that the ``representation`` is wrong.
        """
        # Setup
        instance = SingleTableMetadata()

        # Run / Assert
        instance._validate_numerical('age', representation=representation)

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
            - ``ValueError`` indicating the format ``%`` that has not been formatted.
        """
        # Setup
        instance = SingleTableMetadata()

        # Run / Assert
        instance._validate_datetime('start_date', datetime_format='%Y-%m-%d')
        instance._validate_datetime('start_date', datetime_format='%Y-%m-%d - Synthetic')

        error_msg = re.escape(
            "Invalid datetime format string '%1-%Y-%m-%d-%' for datetime column 'start_date'.")
        with pytest.raises(ValueError, match=error_msg):
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
            - ``ValueError`` when both ``order`` and ``order_by`` are present.
            - ``ValueError`` when ``order`` is an empty list or a random string.
            - ``ValueError`` when ``order_by`` is not ``numerical_value`` or ``alphabetical``.
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
        with pytest.raises(ValueError, match=error_msg):
            instance._validate_categorical('name', order_by='alphabetical', order=['a', 'b', 'c'])

        error_msg_order_by = re.escape(
            "Unknown ordering method 'my_ordering' provided for categorical column "
            "'name'. Ordering method must be 'numerical_value' or 'alphabetical'."
        )
        with pytest.raises(ValueError, match=error_msg_order_by):
            instance._validate_categorical('name', order_by='my_ordering')

        error_msg_order = re.escape(
            "Invalid order value provided for categorical column 'name'. "
            "The 'order' must be a list with 1 or more elements."
        )
        with pytest.raises(ValueError, match=error_msg_order):
            instance._validate_categorical('name', order='my_ordering')

        with pytest.raises(ValueError, match=error_msg_order):
            instance._validate_categorical('name', order=[])

    def test__validate_text(self):
        """Test the ``_validate_text`` method.

        Setup:
            - instance of ``SingleTableMetadata``

        Input:
            - Column name.
            - sdtype text
            - Valid ``regex_format``.
            - Invalid ``regex_format``.

        Side Effects:
            - ``ValueError``
        """
        # Setup
        instance = SingleTableMetadata()

        # Run / Assert
        instance._validate_text('phrase', regex_format='[A-z]')
        error_msg = re.escape("Invalid regex format string '[A-z{' for text column 'phrase'.")
        with pytest.raises(ValueError, match=error_msg):
            instance._validate_text('phrase', regex_format='[A-z{')

    def test__validate_column_exists(self):
        """Test the ``_validate_column_exists`` method.

        Setup:
            - instance of ``SingleTableMetadata``
            - A list of ``_columns``.

        Input:
            - Column name.

        Side Effects:
            - ``ValueError`` when the column is not in the ``instance._columns``.
        """
        # Setup
        instance = SingleTableMetadata()
        instance._columns = {
            'name': {'sdtype': 'categorical'},
            'age': {'sdtype': 'numerical'},
            'start_date': {'sdtype': 'datetime'},
            'phrase': {'sdtype': 'text'},
        }

        # Run / Assert
        instance._validate_column_exists('age')
        error_msg = re.escape(
            "Column name ('synthetic') does not exist in the table. "
            "Use 'add_column' to add new column."
        )
        with pytest.raises(ValueError, match=error_msg):
            instance._validate_column_exists('synthetic')

    @pytest.mark.parametrize('column_name, sdtype, kwargs', VALID_KWARGS)
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

    @pytest.mark.parametrize('column_name, sdtype, kwargs, error_msg', INVALID_KWARGS)
    def test__validate_unexpected_kwargs_invalid(self, column_name, sdtype, kwargs, error_msg):
        """Test the ``_validate_unexpected_kwargs`` method.

        Setup:
            - instance of ``SingleTableMetadata``

        Input:
            - Column name.
            - sdtype
            - unexpected kwargs

        Side Effects:
            - ``ValueError`` is being raised for each sdtype.
        """
        # Setup
        instance = SingleTableMetadata()

        # Run / Assert
        with pytest.raises(ValueError, match=error_msg):
            instance._validate_unexpected_kwargs(column_name, sdtype, **kwargs)

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
        instance._validate_column('age', 'numerical', representation='int')

        # Assert
        mock__validate_kwargs.assert_called_once_with('age', 'numerical', representation='int')
        mock__validate_numerical.assert_called_once_with('age', representation='int')

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
    @patch('sdv.metadata.single_table.SingleTableMetadata._validate_text')
    def test__validate_column_text(self, mock__validate_text, mock__validate_kwargs):
        """Test ``_validate_column`` method.

        Test the ``_validate_column`` method when a ``text`` sdtype is passed.

        Setup:
            - Instance of ``SingleTableMetadata``.

        Input:
            - ``column_name`` - a string.
            - ``sdtype`` - a string 'text'.
            - kwargs - any additional key word arguments.

        Mock:
            - ``_validate_unexpected_kwargs``
            - ``_validate_text`` function from ``SingleTableMetadata``.

        Side effects:
            - ``_validate_text`` has been called once.
        """
        # Setup
        instance = SingleTableMetadata()

        # Run
        instance._validate_column('phrase', 'text', regex_format='[A-z0-9]', pii=True)

        # Assert
        mock__validate_kwargs.assert_called_once_with(
            'phrase', 'text', regex_format='[A-z0-9]', pii=True)
        mock__validate_text.assert_called_once_with('phrase', regex_format='[A-z0-9]', pii=True)

    def test_add_column_column_name_in_columns(self):
        """Test ``add_column`` method.

        Test that when calling ``add_column`` with a column that is already in
        ``instance._columns`` raises a ``ValueError`` stating to use the ``update_column`` instead.

        Setup:
            - Instance of ``SingleTableMetadata``.
            - ``_columns`` with some values.

        Input:
            - A column name that is already in ``instance._columns``.

        Side Effects:
            - ``ValueError`` is being raised stating that the column exists.
        """
        # Setup
        instance = SingleTableMetadata()
        instance._columns = {'age': {'sdtype': 'numerical'}}

        # Run / Assert
        error_msg = re.escape(
            "Column name 'age' already exists. Use 'update_column' to update an existing column.")
        with pytest.raises(ValueError, match=error_msg):
            instance.add_column('age')

    def test_add_column_sdtype_not_in_kwargs(self):
        """Test ``add_column`` method.

        Test that when calling ``add_column`` without an sdtype a ``ValueError`` stating that
        it must be provided is raised.

        Setup:
            - Instance of ``SingleTableMetadata``.

        Input:
            - A column name.

        Side Effects:
            - ``ValueError`` is being raised stating that sdtype must be provided.
        """
        # Setup
        instance = SingleTableMetadata()

        # Run / Assert
        error_msg = re.escape("Please provide a 'sdtype' for column 'synthetic'.")
        with pytest.raises(ValueError, match=error_msg):
            instance.add_column('synthetic')

    def test_add_column(self):
        """Test ``add_column`` method.

        Test that when calling ``add_column`` method with a ``sdtype`` and the proper ``kwargs``
        this is being added to the ``instance._columns``.

        Setup:
            - Instance of ``SingleTableMetadata``.

        Input:
            - A column name.
            - An ``sdtype``.

        Side Effects:
            - ``instance._columns[column_name]`` now exists.
        """
        # Setup
        instance = SingleTableMetadata()

        # Run
        instance.add_column('age', sdtype='numerical', representation='int')

        # Assert
        assert instance._columns['age'] == {'sdtype': 'numerical', 'representation': 'int'}

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
        instance._columns = {'age': {'sdtype': 'numerical'}}

        # Run
        instance.update_column('age', sdtype='categorical', order_by='numerical_value')

        # Assert
        assert instance._columns['age'] == {
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
        instance._columns = {'age': {'sdtype': 'numerical'}}

        # Run
        instance.update_column('age', representation='float')

        # Assert
        assert instance._columns['age'] == {
            'sdtype': 'numerical',
            'representation': 'float'
        }
        mock__validate_column_exists.assert_called_once_with('age')
        mock__validate_column.assert_called_once_with('age', 'numerical', representation='float')

    def test_detect_from_dataframe_raises_value_error(self):
        """Test the ``detect_from_dataframe`` method.

        Test that if there are existing columns in the metadata, this raises a ``ValueError``.

        Setup:
            - instance of ``SingleTableMetadata``.
            - Add some value to ``instance._columns``.

        Side Effects:
            Raises a ``ValueError`` stating that ``metadata`` already exists.
        """
        # Setup
        instance = SingleTableMetadata()
        instance._columns = {'column': {'sdtype': 'categorical'}}

        # Run / Assert
        err_msg = (
            'Metadata already exists. Create a new ``SingleTableMetadata`` '
            'object to detect from other data sources.'
        )

        with pytest.raises(ValueError, match=err_msg):
            instance.detect_from_dataframe('dataframe')

    @patch('sdv.metadata.single_table.print')
    def test_detect_from_dataframe(self, mock_print):
        """Test the ``dectect_from_dataframe`` method.

        Test that when given a ``pandas.DataFrame``, the current instance of
        ``SingleTableMetadata`` is being updated with the ``sdtypes`` of each
        column in the ``pandas.DataFrame``.

        Setup:
            - Instance of ``SingleTableMetadata``.

        Input:
            - ``pandas.DataFrame`` with multiple data types.

        Side Effects:
            - ``instance._columns`` has been updated with the expected ``sdtypes``.
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
        assert instance._columns == {
            'categorical': {'sdtype': 'categorical'},
            'date': {'sdtype': 'datetime'},
            'int': {'sdtype': 'numerical'},
            'float': {'sdtype': 'numerical'},
            'bool': {'sdtype': 'boolean'}
        }

        expected_print_calls = [
            call('Detected metadata:'),
            call(json.dumps(instance.to_dict(), indent=4))
        ]
        assert mock_print.call_args_list == expected_print_calls

    def test_detect_from_csv_raises_value_error(self):
        """Test the ``detect_from_csv`` method.

        Test that if there are existing columns in the metadata, this raises a ``ValueError``.

        Setup:
            - instance of ``SingleTableMetadata``.
            - Add some value to ``instance._columns``.

        Side Effects:
            Raises a ``ValueError`` stating that ``metadata`` already exists.
        """
        # Setup
        instance = SingleTableMetadata()
        instance._columns = {'column': {'sdtype': 'categorical'}}

        # Run / Assert
        err_msg = (
            'Metadata already exists. Create a new ``SingleTableMetadata`` '
            'object to detect from other data sources.'
        )

        with pytest.raises(ValueError, match=err_msg):
            instance.detect_from_csv('filepath')

    @patch('sdv.metadata.single_table.print')
    def test_detect_from_csv(self, mock_print):
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
            - ``instance._columns`` has been updated with the expected ``sdtypes``.
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
        with TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / 'mydata.csv'
            data.to_csv(filepath, index=False)
            instance.detect_from_csv(filepath)

        # Assert
        assert instance._columns == {
            'categorical': {'sdtype': 'categorical'},
            'date': {'sdtype': 'categorical'},
            'int': {'sdtype': 'numerical'},
            'float': {'sdtype': 'numerical'},
            'bool': {'sdtype': 'boolean'}
        }

        expected_print_calls = [
            call('Detected metadata:'),
            call(json.dumps(instance.to_dict(), indent=4))
        ]
        assert mock_print.call_args_list == expected_print_calls

    @patch('sdv.metadata.single_table.print')
    def test_detect_from_csv_with_kwargs(self, mock_print):
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
            - ``instance._columns`` has been updated with the expected ``sdtypes``.
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
        with TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / 'mydata.csv'
            data.to_csv(filepath, index=False)
            instance.detect_from_csv(filepath, pandas_kwargs={'parse_dates': ['date']})

        # Assert
        assert instance._columns == {
            'categorical': {'sdtype': 'categorical'},
            'date': {'sdtype': 'datetime'},
            'int': {'sdtype': 'numerical'},
            'float': {'sdtype': 'numerical'},
            'bool': {'sdtype': 'boolean'}
        }

        expected_print_calls = [
            call('Detected metadata:'),
            call(json.dumps(instance.to_dict(), indent=4))
        ]
        assert mock_print.call_args_list == expected_print_calls

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
        out is True

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
        out is False

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
        out is True

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
        out is False

    def test_set_primary_key_validation(self):
        """Test that ``set_primary_key`` crashes for invalid arguments.

        Input:
            - A tuple with non-string values.

        Side Effect:
            - A ``ValueError`` should be raised.
        """
        # Setup
        instance = SingleTableMetadata()

        err_msg = "'primary_key' must be a string or tuple of strings."
        # Run / Assert
        with pytest.raises(ValueError, match=err_msg):
            instance.set_primary_key(('1', 2, '3'))

    def test_set_primary_key(self):
        """Test that ``set_primary_key`` sets the ``_metadata['primary_key']`` value."""
        # Setup
        instance = SingleTableMetadata()

        # Run
        instance.set_primary_key('column')

        # Assert
        instance._metadata['primary_key'] == 'column'

    def test_set_primary_key_tuple(self):
        """Test that ``set_primary_key`` sets the ``_metadata['primary_key']`` value for tuples."""
        # Setup
        instance = SingleTableMetadata()

        # Run
        instance.set_primary_key(('column1', 'column2'))

        # Assert
        instance._metadata['primary_key'] == ('column1', 'column2')

    @patch('sdv.tabular.utils.warnings')
    def test_set_primary_key_warning(self, warning_mock):
        """Test that ``set_primary_key`` raises a warning when a primary key already exists.

        Setup:
            - An instance of ``SingleTableMetadata`` with ``_metadata['primary_key']`` set.

        Input:
            - String.

        Side Effect:
            - A warning should be raised.
        """
        # Setup
        instance = SingleTableMetadata()
        instance._metadata['primary_key'] = 'column0'

        # Run
        instance.set_primary_key('column1')

        # Assert
        warning_msg = "There is an existing primary key 'column0'. This key will be removed."
        warning_mock.warn.called_once_with(warning_msg)
        instance._metadata['primary_key'] == 'column1'

    def test_set_alternate_keys_validation(self):
        """Test that ``set_alternate_keys`` crashes for invalid arguments.

        Input:
            - A list with tuples with non-string values.

        Side Effect:
            - A ``ValueError`` should be raised.
        """
        # Setup
        instance = SingleTableMetadata()

        err_msg = "'alternate_keys' must be a list of strings or a list of tuples of strings."
        # Run / Assert
        with pytest.raises(ValueError, match=err_msg):
            instance.set_alternate_keys(['col1', ('1', 2, '3'), 'col3'])

    def test_set_alternate_keys(self):
        """Test that ``set_alternate_keys`` sets the ``_metadata['alternate_keys']`` value."""
        # Setup
        instance = SingleTableMetadata()

        # Run
        instance.set_alternate_keys(['column1', ('column2', 'column3')])

        # Assert
        instance._metadata['alternate_keys'] == ['column1', ('column2', 'column3')]

    def test_set_sequence_key_validation(self):
        """Test that ``set_sequence_key`` crashes for invalid arguments.

        Input:
            - A tuple with non-string values.

        Side Effect:
            - A ``ValueError`` should be raised.
        """
        # Setup
        instance = SingleTableMetadata()

        err_msg = "'sequence_key' must be a string or tuple of strings."
        # Run / Assert
        with pytest.raises(ValueError, match=err_msg):
            instance.set_sequence_key(('1', 2, '3'))

    def test_set_sequence_key(self):
        """Test that ``set_sequence_key`` sets the ``_metadata['sequence_key']`` value."""
        # Setup
        instance = SingleTableMetadata()

        # Run
        instance.set_sequence_key('column')

        # Assert
        instance._metadata['sequence_key'] == 'column'

    def test_set_sequence_key_tuple(self):
        """Test that ``set_sequence_key`` sets ``_metadata['sequence_key']`` for tuples."""
        # Setup
        instance = SingleTableMetadata()

        # Run
        instance.set_sequence_key(('column1', 'column2'))

        # Assert
        instance._metadata['sequence_key'] == ('column1', 'column2')

    @patch('sdv.tabular.utils.warnings')
    def test_set_sequence_key_warning(self, warning_mock):
        """Test that ``set_sequence_key`` raises a warning when a sequence key already exists.

        Setup:
            - An instance of ``SingleTableMetadata`` with ``_metadata['sequence_key']`` set.

        Input:
            - String.

        Side Effect:
            - A warning should be raised.
        """
        # Setup
        instance = SingleTableMetadata()
        instance._metadata['sequence_key'] = 'column0'

        # Run
        instance.set_primary_key('column1')

        # Assert
        warning_msg = "There is an existing sequence key 'column0'. This key will be removed."
        warning_mock.warn.called_once_with(warning_msg)
        instance._metadata['sequence_key'] == 'column1'

    def test_set_sequence_index_validation(self):
        """Test that ``set_sequence_index`` crashes for invalid arguments.

        Input:
            - A non-string value.

        Side Effect:
            - A ``ValueError`` should be raised.
        """
        # Setup
        instance = SingleTableMetadata()

        err_msg = "'sequence_index' must be a string."
        # Run / Assert
        with pytest.raises(ValueError, match=err_msg):
            instance.set_sequence_index(('column1', 'column2'))

    def test_set_sequence_index(self):
        """Test that ``set_sequence_index`` sets the ``_metadata['sequence_index']`` value."""
        # Setup
        instance = SingleTableMetadata()

        # Run
        instance.set_sequence_index('column')

        # Assert
        instance._metadata['sequence_index'] == 'column'

    def test_to_dict(self):
        """Test the ``to_dict`` method from ``SingleTableMetadata``.

        Setup:
            - Instance of ``SingleTableMetadata`` and modify the ``instance._columns`` to ensure
            that ``to_dict`` works properly.
            - Add constraint Mock and ensure that `to_dict` of the object is being called.
        Output:
            - A dictionary representation of the ``instance`` that does not modify the
              internal dictionaries.
        """
        # Setup
        instance = SingleTableMetadata()
        instance._columns['my_column'] = 'value'
        constraint = Mock()
        constraint.to_dict.return_value = {'column': 'value', 'scalar': 1}
        dict_constraint = {'column': 'value', 'increment_value': 20}
        instance._constraints.extend([constraint, dict_constraint])

        # Run
        result = instance.to_dict()

        # Assert
        assert result == {
            'columns': {'my_column': 'value'},
            'constraints': [
                {'column': 'value', 'scalar': 1},
                {'column': 'value', 'increment_value': 20}
            ],
            'SCHEMA_VERSION': 'SINGLE_TABLE_V1'
        }

        constraint.to_dict.assert_called_once()

        # Ensure that the output object does not alterate the inside object
        result['columns']['my_column'] = 1
        assert instance._columns['my_column'] == 'value'

    def test__set_metadata_dict(self):
        """Test the ``_set_metadata_dict`` to a instance.

        Setup:
            - Instance of ``SingleTableMetadata``.
            - Dictionary representing ``SingleTableMetadata``.

        Output:
            - ``SingleTableMetadata`` instance with the dictionary represented values.
        """
        # Setup
        instance = SingleTableMetadata()
        metadata = {
            'columns': {'my_column': 'value'},
            'SCHEMA_VERSION': 'SINGLE_TABLE_V1'
        }

        # Run
        instance._set_metadata_dict(metadata)

        # Assert
        assert instance._metadata == {
            'columns': {'my_column': 'value'},
            'primary_key': None,
            'alternate_keys': [],
            'constraints': [],
            'SCHEMA_VERSION': 'SINGLE_TABLE_V1'
        }

        assert instance._columns == {'my_column': 'value'}

    def test__load_from_dict(self):
        """Test that ``_load_from_dict`` returns a instance with the ``dict`` updated objects."""
        # Setup
        my_metadata = {
            'columns': {'my_column': 'value'},
            'primary_key': 'pk',
            'alternate_keys': [],
            'sequence_keys': None,
            'sequence_index': None,
            'constraints': [],
            'SCHEMA_VERSION': 'SINGLE_TABLE_V1'
        }

        # Run
        instance = SingleTableMetadata._load_from_dict(my_metadata)

        # Assert
        assert instance._metadata == my_metadata
        assert instance._columns == {'my_column': 'value'}
        assert instance._constraints == []

    @patch('sdv.metadata.single_table.Path')
    def test_load_from_json_path_does_not_exist(self, mock_path):
        """Test the ``load_from_json`` method.

        Test that the method raises a ``ValueError`` when the specified path does not exist.

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

    @patch('sdv.metadata.single_table.open')
    @patch('sdv.metadata.single_table.Path')
    @patch('sdv.metadata.single_table.json')
    def test_load_from_json_schema_not_present(self, mock_json, mock_path, mock_open):
        """Test the ``load_from_json`` method.

        Test that the method raises a ``ValueError`` when the specified ``json`` file does
        not contain a ``SCHEMA_VERSION`` in it.

        Mock:
            - Mock the ``Path`` library in order to return ``True``, so the file exists.
            - Mock the ``json`` library in order to use a custom return.
            - Mock the ``open`` in order to avoid loading a binary file.

        Input:
            - String representing a filepath.

        Side Effects:
            - A ``ValueError`` is raised pointing that the given metadata configuration is not
              compatible with the current version.
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
        with pytest.raises(ValueError, match=error_msg):
            SingleTableMetadata.load_from_json('filepath.json')

    @patch('sdv.metadata.single_table.Constraint')
    @patch('sdv.metadata.single_table.open')
    @patch('sdv.metadata.single_table.Path')
    @patch('sdv.metadata.single_table.json')
    def test_load_from_json(self, mock_json, mock_path, mock_open, mock_constraint):
        """Test the ``load_from_json`` method.

        Test that ``load_from_json`` function creates an instance with the contents returned by the
        ``json`` load function.

        Mock:
            - Mock the ``Path`` library in order to return ``True``.
            - Mock the ``json`` library in order to use a custom return.
            - Mock the ``open`` in order to avoid loading a binary file.
            - Mock the ``Constraint`` to ensure that is being loaded.

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
        mock_constraint.from_dict.return_value = {'my_constraint': 'my_params'}
        mock_json.load.return_value = {
            'columns': {
                'animals': {
                    'type': 'categorical'
                }
            },
            'primary_key': 'animals',
            'constraints': [{
                'my_constraint': 'my_params'
            }],
            'SCHEMA_VERSION': 'SINGLE_TABLE_V1'
        }

        # Run
        instance = SingleTableMetadata.load_from_json('filepath.json')

        # Assert
        expected_metadata = {
            'columns': {'animals': {'type': 'categorical'}},
            'primary_key': 'animals',
            'alternate_keys': [],
            'sequence_keys': None,
            'sequence_index': None,
            'constraints': [{'my_constraint': 'my_params'}],
            'SCHEMA_VERSION': 'SINGLE_TABLE_V1'
        }
        assert instance._columns == {'animals': {'type': 'categorical'}}
        assert instance._constraints == [{'my_constraint': 'my_params'}]
        assert instance._metadata == expected_metadata
        mock_constraint.from_dict.assert_called_once()

    @patch('sdv.metadata.single_table.Path')
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

    def test_save_to_json(self):
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
        with TemporaryDirectory() as temp_dir:
            file_name = Path(temp_dir) / 'singletable.json'
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
